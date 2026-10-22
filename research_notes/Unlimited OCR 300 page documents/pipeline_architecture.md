# Pipeline and API Architecture for OCR of Very Long Documents (300+ pages) with a Page-Level VLM

Research date: 2026-10-22. 16 tool calls (search plus fetch). The primary sources are the olmOCR source code (pipeline.py, work_queue.py), the Marker README, cloud provider limit docs (Azure, AWS, Google), the vLLM DeepSeek-OCR recipe, the pypdfium2 docs and the Celery docs. Some items come from secondary or community sources, and each one is labeled as such.

## 1. PDF handling: streaming rasterization, library choice, DPI, embedded text layer

### Takeaway
Render one page at a time from an open document handle, and parallelize with processes rather than threads, because PDFium is not thread-safe. Size images by the longest side in pixels (olmOCR uses 1288 px) rather than a fixed DPI. I found no independent, current benchmark that compares pypdfium2, PyMuPDF and pdf2image head to head. Where a usable embedded text layer exists, production tools use it either as the fast path (Marker) or as the fallback (olmOCR via pdftotext).

### Cited Findings
- **pypdfium2 thread safety:** the official docs say PDFium is not thread-safe. Calling pdfium functions at the same time from different threads is not allowed, even on different documents. Threads are acceptable only if a mutex serializes every PDFium call. For parallel rendering the docs recommend a multiprocessing pool where each process sets up its own context. — [pypdfium2 Python API docs](https://pypdfium2.readthedocs.io/en/stable/python_api.html)
- **pypdfium2 removed its built-in multiprocess render:** in 4.25.0, `PdfDocument.render()` was dropped in favour of linear rendering. The maintainers judged that sending pixel data back from worker processes was costly, gave no real performance gain and raised memory load. — [pypdfium2 changelog](https://pypdfium2.readthedocs.io/en/stable/changelog.html)
  - Implication: each worker should open the PDF itself, render its own page range and write PNG/JPEG straight to disk or object storage, instead of returning bitmaps to a parent process.
- **pypdfium2 scale:** the scale factor is relative to 72 DPI, so 1.5 is 108 DPI and 2.0 is 144 DPI. This comes from a third-party code comment, not the official API reference. — [equalify-reflow page_image_renderer.py](https://equalify.uic.edu/EqualifyEverything/equalify-reflow/blob/3ed4124ce2ee0005fe52ff940a3fb4734a0e2f97/src/services/page_image_renderer.py)
- **Speed comparisons (vendor-reported only):**
  - SoPDF, a library built on pypdfium2, reports single-page renders 2.2x to 2.8x faster than PyMuPDF at 72, 150 and 300 DPI. Its 50-page parallel run was nearly tied with PyMuPDF (1.07x). — [sopdf on PyPI](https://pypi.org/project/sopdf/)
  - PyMuPDF's own older methodology reports about 367 s to render 7,031 pages, against about 646 s for XPDF. — [PyMuPDF Appendix 4](https://pymupdf.readthedocs.io/en/latest/app4.html)
  - Neither source covers pdf2image/poppler or measures memory.
- **olmOCR rendering:** pages are rendered by `render_pdf_to_base64png(local_pdf_path, page, target_longest_image_dim=...)`, with a default of `target_longest_image_dim = 1288`. Rendering runs in `asyncio.to_thread`, gated by a semaphore sized to CPU count minus 2. Poppler is checked at startup, and rotation corrections are applied afterwards with PIL `transpose`. — [olmOCR pipeline.py](https://raw.githubusercontent.com/allenai/olmocr/main/olmocr/pipeline.py)
- **Marker's use of the text layer:** in fast mode, Marker uses pdftext to "Extract embedded text with pdftext, in the PDF's reading order".
  - `--force_ocr` forces OCR on every page and is recommended when the text layer is garbled.
  - `strip_existing_ocr` removes existing OCR text and re-OCRs.
  - Tables are rebuilt from the text layer, and "low-confidence reconstructions fall back to the VLM".
  - Source: [Marker README](https://github.com/datalab-to/marker)
- **olmOCR's use of the text layer as fallback:** when every retry of a page fails, `make_fallback_result` uses `get_anchor_text(..., pdf_engine="pdftotext")` and flags the page `is_fallback=True`. — [olmOCR pipeline.py](https://raw.githubusercontent.com/allenai/olmocr/main/olmocr/pipeline.py)
- **Docling memory tips (community bot answer, unverified):**
  - Use `pdf_backend="pypdfium2"` to save memory and time.
  - Set `generate_page_images=False` and `generate_picture_images=False`.
  - Use `page_batch_size` (CLI `--page-batch-size`), "Start with 4–16 and profile".
  - Source: [Docling discussion #2180](https://github.com/docling-project/docling/discussions/2180)
- **Minimum text size:** AWS Textract needs text at least 15 px tall, which it says equals 8-point font at 150 DPI. This is a useful lower bound when choosing DPI for small print. — [AWS Textract set quotas](https://docs.aws.amazon.com/textract/latest/dg/limits-document.html)

### Inferences
- For a 300+ page job, separate rendering from inference. CPU workers render pages (pypdfium2 or PyMuPDF, one document handle per process) into object storage keyed `{job_id}/pages/{n:05d}.png`. GPU-side workers then stream those images into vLLM, so the whole document is never held in RAM.
- Size by the model's native resolution, in longest-side pixels, rather than by DPI. Upsample only when the small-text rule above would fail. A typical working range is about 1024 to 1600 px, or roughly 144 to 200 DPI for A4/Letter. This range is an inference, not a cited benchmark.
- Use a cheap pre-pass that extracts the text layer for each page (pypdfium2 `get_textpage()` or pdftext). Treat each page as one of three cases:
  - "born digital, good text": skip the VLM, or use the VLM only for tables and figures.
  - "scanned or no text": VLM.
  - "garbled text": VLM.
  - This both saves GPU time and gives you an anchor and fallback text for every page.

### Gaps
- No independent 2024–2026 benchmark covering speed and peak RSS for pypdfium2, PyMuPDF and pdf2image. Teams should benchmark on their own corpus at the target resolution.
- No official recommended DPI from Marker, MinerU or DeepSeek-OCR was confirmed in this pass.
- olmOCR's exact rasterizer behind `render_pdf_to_base64png` (pdftoppm or something else) was not visible in the fetched file.

## 2. Job architecture: async API, queues, chunking, per-page status, checkpointing, retries, idempotency

### Takeaway
Every reference design uses the same shape: submit, get a job or operation id, then poll (or receive a notification) and fetch paginated results. Work is broken into page groups. Completion is recorded per unit in durable storage. Retries are bounded and escalate (temperature, rotation). Units are content-hashed so re-running is idempotent. olmOCR does all of this with nothing but S3 plus lock and done-flag files.

### Cited Findings
- **olmOCR work-queue design:**
  - Work items are groups of PDFs, each identified by a SHA-1 over its sorted paths.
  - The index is stored as `work_index_list.csv.zstd`, one CSV row per group: the hash, then its paths. New paths are appended and existing groups are kept.
  - A worker claims an item by writing `worker_{hash}.lock`. There is no heartbeat. A lock older than `worker_lock_timeout_secs` (default 1800) counts as stale and can be overwritten.
  - Completion is a `done_flags/done_{hash}.flag` file. `mark_done` writes the flag and deletes the lock.
  - On startup the queue subtracts completed hashes and shuffles what remains.
  - Source: [olmOCR work_queue.py](https://raw.githubusercontent.com/allenai/olmocr/main/olmocr/work_queue.py)
- **olmOCR multi-node setup:** the workspace can be a local folder or an S3 prefix. The first node "sets up a simple work queue in your AWS bucket", and other nodes point at the same workspace to pull from that queue. — [olmOCR README](https://github.com/allenai/olmocr)
- **olmOCR batching and concurrency defaults:**
  - `pages_per_group` defaults to 500 (50 when using an external API key). Items per group are computed as `max(1, int(pages_per_group / avg_pages_per_pdf))`, with the average taken from up to 100 sampled PDFs.
  - `workers=20`, `max_concurrent_requests=1600`, `max_model_len=16384`, `MAX_TOKENS=8000`.
  - Each work item writes `results/output_{hash}.jsonl`. `mark_done` is called only after the output has been written; on an exception the item is not marked done, so it gets retried.
  - Source: [olmOCR pipeline.py](https://raw.githubusercontent.com/allenai/olmocr/main/olmocr/pipeline.py)
- **olmOCR retry policy:**
  - `max_page_retries=8`, with `TEMPERATURE_BY_ATTEMPT = [0.1, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 1.0]`. A code comment says higher temperature helps with repetition.
  - If the model reports the rotation as invalid, the page is retried with the cumulative `rotation_correction` applied.
  - When the vLLM queue is empty, the remaining attempts are fired in parallel. The first valid result wins and the others are cancelled.
  - Connection errors use exponential backoff `10 * 2**n`, up to 10 attempts, and then the process exits.
  - Source: [olmOCR pipeline.py](https://raw.githubusercontent.com/allenai/olmocr/main/olmocr/pipeline.py)
- **Marker sharding and resume:**
  - `--workers` sets the number of conversion workers.
  - `--num_chunks <nodes> --chunk_idx <i>` shards the file list across machines.
  - `--skip_existing` resumes by skipping inputs that already have output.
  - `--page_range` selects pages, and `VLLM_GPUS=0,1,2,3` spans one vLLM server across GPUs.
  - For out-of-memory problems the README suggests "splitting up long PDFs into multiple files" or lowering the worker count.
  - Source: [Marker README](https://github.com/datalab-to/marker)
- **Client-side page splitting in other systems:**
  - The Unstructured SDK's `split_pdf_page` (on by default) splits the PDF on the client into batches of 2 to 20 pages and sends them as concurrent requests. — [Unstructured docs](https://docs.unstructured.io/api-reference/partition/speed-up-large-files-batches)
  - NVIDIA NeMo Retriever's V2 API splits large PDFs into chunks of 1 to 128 pages and reports 1.3–1.5x faster processing for large documents. This figure comes from a search-result summary; the page itself redirected and was not read in full. — [NeMo Retriever v2 API guide](https://docs.nvidia.com/nemo/retriever/26.1.1/extraction/v2-api-guide)
- **MinerU tuning (third-party blog, unverified against official docs):**
  - The memory-control environment variables include `MINERU_PROCESSING_WINDOW_SIZE`, `MINERU_HYBRID_BATCH_RATIO` and `MINERU_API_MAX_CONCURRENT_REQUESTS`.
  - `MINERU_HYBRID_BATCH_RATIO` defaults to 1 on GPUs under 8 GB and 2 on GPUs with 8 GB or more, and it only controls batch size for the small models.
  - Hybrid mode keeps 4–5 models resident in VRAM, while `vlm-auto-engine` loads a single model of about 2B parameters.
  - Source: [nite07 blog](https://nite07.com/posts/mineru-low-vram/)
- **Celery reliability for long tasks:**
  - On a Redis broker, a task not acknowledged within `visibility_timeout` (default 1 hour) is redelivered to another worker. It is raised with `broker_transport_options={'visibility_timeout': ...}`.
  - `acks_late` makes crashed tasks redeliverable, but tasks must then be idempotent. A long timeout also delays redelivery of tasks lost to hard crashes.
  - Sources: [Celery Redis broker docs](https://docs.celeryq.dev/en/v5.0.0/getting-started/brokers/redis.html); [dev.to write-up](https://dev.to/akoladefaj/celery-loses-8-of-your-tasks-by-default-heres-the-reliability-layer-i-built-to-fix-that-40mc)
- **Ray Data with vLLM:** large batch sizes tend to saturate compute, while small batch sizes are more fault-tolerant. — [Ray vLLMEngineProcessorConfig](https://docs.ray.io/en/latest/data/api/doc/ray.data.llm.vLLMEngineProcessorConfig.html)
- **Azure async pattern:**
  - Analyze is a POST, and results come back through GET polling.
  - Defaults: 15 analyze TPS and 50 GET TPS. Microsoft recommends polling no more than once every 2 seconds, honouring the `retry-after` header and backing off exponentially on 429s (for example 2-5-13-34 s).
  - Source: [Azure Document Intelligence service limits](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/service-limits)

### Inferences
- **Recommended design for a self-hosted vLLM page model:**
  1. `POST /jobs` (multipart upload or a URI, plus an `Idempotency-Key` or the file's SHA-256) returns `202 {job_id}`.
  2. The server stores the PDF in object storage, counts the pages and writes a `pages` table: `(job_id, page_no, status ∈ {pending, rendering, ocr, done, failed, fallback}, attempts, output_key, quality_flags)`.
  3. It enqueues page groups of about 8–32 pages. Groups need to be large enough to keep vLLM's batch full; vLLM batches internally, so the group size mostly affects bookkeeping.
  4. Workers process each page and write `{job_id}/pages/{n}.md` plus a JSON sidecar. Then, in one transaction, they set the page row to done.
  5. `GET /jobs/{id}` returns `{status, pages_total, pages_done, pages_failed, progress}` and an optional webhook on completion.
  6. A finalizer task runs when `pages_done + pages_failed == pages_total` and assembles the document.
- Checkpointing then comes for free. Resuming means re-enqueuing pages that are not done, and idempotency holds because page outputs are keyed by `(doc_sha256, page_no, model_version, prompt_version)`.
- **Queue choice:**
  - A single GPU node with an async FastAPI front end fits Arq (asyncio, Redis) or even a pure-asyncio in-process queue with a Postgres or SQLite state table. This mirrors olmOCR's asyncio plus semaphore design.
  - Multi-node fits Celery (with `acks_late` and a visibility timeout longer than one page-group's runtime) or Ray Data.
  - olmOCR shows that an object-store-only queue (lock and done files) is enough for batch workloads.
- Keep page-level timeouts short and group-level visibility timeouts longer than the expected group runtime, so a stuck page never blocks a 300-page job.

### Gaps
- No primary-source comparison of RQ, Dramatiq or Arq for GPU OCR workloads was found in this pass.
- Docling-serve's async API and its page limits were not verified.
- MinerU's official documentation on processing windows and long PDFs was not retrieved.
- Azure webhook support was not covered by the page fetched (it describes polling only).

## 3. Production systems and cloud reference limits

### Takeaway
The cloud async OCR APIs cap pages per request: Azure at 2,000, AWS Textract at 3,000 and Google Document AI Enterprise OCR at 500. So even the clouds expect very large documents to be split by the client. Open-source pipelines (olmOCR, Marker) shard by file or page group and do not impose a hard page cap.

### Cited Findings
- **Azure Document Intelligence (S0):**
  - Limits: 500 MB maximum document size, 2,000 pages per analysis, 500 MB maximum OCR JSON response.
  - The F0 tier analyzes only the first 2 pages.
  - A `pages` parameter restricts analysis to a page range.
  - Source: [Azure Document Intelligence service limits](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/service-limits)
- **AWS Textract:**
  - Synchronous calls: 10 MB, and 1 page for PDF/TIFF.
  - Asynchronous calls: 500 MB and 3,000 pages for PDF/TIFF.
  - Images are limited to 10,000 px per side.
  - Source: [AWS Textract set quotas](https://docs.aws.amazon.com/textract/latest/dg/limits-document.html)
- **Google Document AI:**
  - Enterprise Document OCR allows 15 pages online and 500 pages in batch. Layout Parser also allows 15 and 500.
  - Imageless online mode raises the online limit to 30 pages, but only for contiguous pages starting at page 1.
  - File sizes: 40 MB online, 1 GB batch. A batch request takes up to 5,000 files.
  - Source: [Google Document AI limits](https://docs.cloud.google.com/document-ai/limits)
- **olmOCR:**
  - No hard per-PDF page cap.
  - `LONG_CONTEXT_THRESHOLD = 32768` is used only in statistics reporting.
  - Optional `--apply_filter` restricts input by language and drops forms and spam.
  - Source: [olmOCR pipeline.py](https://raw.githubusercontent.com/allenai/olmocr/main/olmocr/pipeline.py)
- **DeepSeek-OCR on vLLM:** the official recipe batches many images per `llm.generate()` call. It also serves the model through `vllm serve deepseek-ai/DeepSeek-OCR --logits_processors vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor --no-enable-prefix-caching --mm-processor-cache-gb 0`. — [vLLM DeepSeek-OCR recipe](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-OCR.html)

### Inferences
- A self-hosted API does not need a hard page cap, but it should set a configurable limit (for example 2,000–3,000 pages, matching the clouds) and a file-size limit, and return 413 or 422 above them.
- Page-level parallelism inside one document is the main latency lever. A 300-page job is just 300 independent vLLM requests, and the batch size vLLM can keep in flight sets the wall-clock time.

### Gaps
- AWS GetDocumentTextDetection pagination (`MaxResults` / `NextToken`) and SNS completion details were not returned by the fetched quota page.
- Result retention periods for Azure and AWS were not confirmed.

## 4. Reassembly: merging pages, cross-page tables and paragraphs, headers and footers, provenance

### Takeaway
Concatenate per-page Markdown in page order, but keep a provenance map from character spans to page numbers (olmOCR's `pdf_page_numbers`). Strip running headers and footers. Fix cross-page tables and paragraphs in a separate pass over adjacent page pairs. Cross-page table merging still has no standard method; the strongest signals are repeated header rows, "(continued)" cues, matching column counts and, at the top end, a learned page-pair classifier.

### Cited Findings
- **olmOCR document assembly:**
  - Page texts are joined with `"\n"`.
  - `pdf_page_numbers` holds `[start_char, end_char, page_num]` spans.
  - Per-page attributes include `primary_language`, `is_rotation_valid`, `rotation_correction`, `is_table` and `is_diagram`.
  - Metadata records `pdf-total-pages`, `total-fallback-pages`, and input and output token counts.
  - The document id is SHA-1 of the text.
  - Source: [olmOCR pipeline.py](https://raw.githubusercontent.com/allenai/olmocr/main/olmocr/pipeline.py)
- **Marker:**
  - `--paginate_output` inserts a page-number separator (a dashed line) between pages.
  - Running headers and footers are removed by default and kept with `--keep_pageheader_in_output` / `--keep_pagefooter_in_output`.
  - `--use_llm` can "merge tables across pages".
  - Source: [Marker README](https://github.com/datalab-to/marker)
- **OCRFlux:** pairs adjacent pages after page-to-Markdown conversion, detects elements split across the boundary and merges tables when both sides carry matching table tags. — [OCRFlux write-up](https://aiexpjourney.substack.com/p/the-logic-behind-ocrflux-ai-innovations)
- **Learned continuation detection:**
  - PubTables-v2 reports a ViT-B-16 classifier with F1 0.991 at detecting cross-page table continuation. — [PubTables-v2, arXiv 2512.10888](https://arxiv.org/pdf/2512.10888)
  - POTATR takes a pair of page images, predicts whether a table spans the boundary and concatenates the parts, improving results without retraining the page-level model. — [POTATR, arXiv 2606.09788](https://arxiv.org/pdf/2606.09788)
- **Heuristic merging:** check the geometry (the table touches the page edge) and matching columns, strip the repeated Markdown header row from the second fragment, then join the two tables. — [Heuristic merging blog](https://app-pack.telkomuniversity.ac.id/?p=447158)
- **Commercial approaches:**
  - Tensorlake's agent reads continuation cues such as "(continued)" labels and repeated headers, and checks that the column structures match. — [Tensorlake blog](https://tensorlake.ai/blog/agentic-table-merging)
  - LlamaParse offers a `merge_continued_tables` option. — [LlamaParse tables](https://developers.llamaindex.ai/llamaparse/parse/features/tables/)

### Inferences
- **Recommended reassembly pass, run on adjacent page pairs `(i, i+1)`:**
  1. **Header/footer removal:** take the first and last 1–3 lines of each page and normalize digits (`Page 12` becomes `Page #`). Lines that recur on at least about 50% of pages within a sliding window are running headers or footers and should be removed. This is an inference based on Marker's behaviour, not a cited algorithm.
  2. **Paragraph join:** if page i ends without terminal punctuation and page i+1 starts lowercase or mid-sentence, join with a space instead of a blank line. De-hyphenate `word-\n` followed by `continuation`.
  3. **Table join:** if page i ends with a Markdown or HTML table and page i+1 starts with one that has the same column count, drop the duplicated header row and concatenate.
  4. **Footnotes:** collect footnote blocks per page and either keep them per page as endnotes with page anchors or relocate them to the end of each section.
- Emit page anchors (`<!-- page: 17 -->` or `<a id="p17"></a>`) in the merged Markdown. Keep olmOCR-style `[start, end, page]` spans in a JSON sidecar so downstream RAG chunks can cite pages.

### Gaps
- No published accuracy numbers were found for heuristic header/footer removal or paragraph joining.
- Docling's and MinerU's cross-page merge logic was not inspected in this pass.

## 5. Quality control at scale: garbage pages, repetition loops, fallbacks, sampling

### Takeaway
The minimum QC set is:
- treat any generation not ending with `finish_reason == "stop"`, or exceeding the token budget, as a failure;
- suppress loops at decode time with an n-gram logits processor;
- retry with higher temperature;
- fall back to the text layer;
- fail or flag the document when the fallback-page rate passes a threshold (olmOCR uses 1/250).

### Cited Findings
- **olmOCR validity checks and document threshold:**
  - A result is invalid if `finish_reason != "stop"` or if `total_tokens > MODEL_MAX_CONTEXT`.
  - Retries escalate temperature from 0.1 to 1.0.
  - If nothing valid comes back, the page falls back to the pdftotext anchor text.
  - A document is discarded if its fallback ratio exceeds `max_page_error_rate` (default 0.004, about 1/250). Otherwise it is kept with a warning.
  - Source: [olmOCR pipeline.py](https://raw.githubusercontent.com/allenai/olmocr/main/olmocr/pipeline.py); [olmOCR README](https://github.com/allenai/olmocr)
- **DeepSeek-OCR repetition control:**
  - The `NGramPerReqLogitsProcessor` uses `ngram_size=30` and `window_size=90`.
  - The table tags (`whitelist_token_ids={128821, 128822}`) are whitelisted so legitimate repeated table markup is not penalized.
  - Source: [vLLM DeepSeek-OCR recipe](https://docs.vllm.ai/projects/recipes/en/stable/DeepSeek/DeepSeek-OCR.html)
- **Purpose of that processor:** it suppresses the repetition loops OCR models fall into on tables and dense layouts. — [localaimaster guide (secondary)](https://localaimaster.com/blog/deepseek-ocr-setup-guide); [uv-scripts commit](https://huggingface.co/datasets/uv-scripts/ocr/commit/952354b79567cf4e386a499ce20fb554df6ed93e)
- **olmOCR rotation detection:** the model emits `is_rotation_valid` and `rotation_correction` for each page, and an invalid rotation triggers a re-render and retry. — [olmOCR pipeline.py](https://raw.githubusercontent.com/allenai/olmocr/main/olmocr/pipeline.py)

### Inferences
- **Per-page QC flags to store in the page row:**
  - `empty` (output is empty or whitespace while the image has ink, i.e. its pixel variance is above a threshold);
  - `truncated` (finish_reason is `length`);
  - `repetition` (a compression ratio such as zlib(len)/len below a threshold, or a high top n-gram frequency);
  - `text_layer_disagreement` (low character-level similarity with the embedded text where one exists);
  - `low_density` (characters per page far below the document median).
- **Escalation ladder:**
  1. retry at higher temperature;
  2. retry at a different resolution or with a rotation;
  3. split the page into top and bottom halves and OCR each;
  4. fall back to the text layer;
  5. mark the page `failed` and keep the job partially successful.
- **Sampling:** surface the flagged pages plus a random 1–2% of clean pages in a review queue, showing the page image next to its Markdown.

### Gaps
- No published thresholds for compression-ratio or n-gram repetition detectors in OCR were found in this pass.
- Unlimited-OCR's own failure modes need empirical calibration.

## 6. Storage and delivery of large outputs

### Takeaway
Store artifacts per page in object storage and assemble the final document lazily. Deliver results paginated, or as a streamed or downloadable artifact, rather than as one huge JSON response. Even Azure caps a single OCR JSON response at 500 MB.

### Cited Findings
- Azure caps the OCR JSON response at 500 MB and the input document at 500 MB. — [Azure Document Intelligence service limits](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/service-limits)
- olmOCR writes one JSONL file per work item to `results/output_{hash}.jsonl` in the local or S3 workspace. With `--markdown`, it also writes Markdown files that mirror the input folder structure. — [olmOCR pipeline.py](https://raw.githubusercontent.com/allenai/olmocr/main/olmocr/pipeline.py); [olmOCR README](https://github.com/allenai/olmocr)
- Marker's `--output_format chunks` flattens output into a single list of blocks instead of a tree, which suits streaming and chunked delivery. — [Marker README](https://github.com/datalab-to/marker)

### Inferences
- **Suggested API surface:**
  - `GET /jobs/{id}/pages?offset=&limit=` returns page Markdown plus QC flags, with cursor pagination.
  - `GET /jobs/{id}/pages/{n}` returns a single page.
  - `GET /jobs/{id}/result.md` (and `.json` with provenance spans) returns the assembled output, streamed from object storage or served as a presigned URL.
  - `GET /jobs/{id}/events` sends Server-Sent Events with per-page progress.
  - Optional webhook on completion, carrying an HMAC signature.
- Use partial results as a feature: clients can consume pages while the job is still running, which matters for 300+ page documents.
- Set a TTL or lifecycle policy on page images, which are large, separately from text outputs, which are small. Keep the source-PDF hash for deduplication and cache hits.

### Gaps
- No primary source on how Google Document AI batch output sharding works (output files per N pages in GCS) was retrieved.
- AWS Textract's result pagination and retention period were not retrieved.
