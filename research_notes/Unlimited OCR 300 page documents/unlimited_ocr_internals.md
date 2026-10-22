# Unlimited-OCR internals and limits for 300+ page documents

Scope: local repo `/data/anything-ocr/Unlimited-OCR` (branch `feat/ocr-api`, HEAD `f34e409`), the bundled paper `Unlimited-OCR.pdf`, the bundled SGLang wheel, the vLLM implementation inside the Docker image `vllm/vllm-openai:unlimited-ocr` (vLLM `0.23.1rc1.dev541+g7aaf016a1`), the HF model snapshots in `/data/hf-cache`, and the upstream HF card, vLLM recipe and GitHub issues. Checked on 2026-10-22.

Citation conventions:
- Repo files are cited as `path:line`, relative to the repo root.
- `paper p.N` is the page of `Unlimited-OCR.pdf`, which is also https://arxiv.org/abs/2606.23050.
- `vllm-image:unlimited_ocr.py` is `/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/unlimited_ocr.py` inside the Docker image.
- `hf-snapshot:modeling_unlimitedocr.py` is `/data/hf-cache/hub/models--baidu--Unlimited-OCR/snapshots/8249d20b.../modeling_unlimitedocr.py`.

## 1. What is Unlimited-OCR (base model, architecture, "unlimited" claims, context, tokens per page, resolutions)?

### Takeaway
Unlimited-OCR is Baidu's continued training of DeepSeek-OCR. It keeps the same DeepEncoder and the same 3B-total / ~0.5B-active MoE decoder, but every decoder attention layer is replaced with Reference Sliding Window Attention (R-SWA), which keeps the decoder's KV cache constant during generation. The name "unlimited" refers to output length only. Input is still capped by the 32K context. The paper says outright that the model "cannot achieve truly unlimited parsing under a finite context length (e.g., 32K)." In multi-page mode each page costs about 273 prefill tokens, so roughly 120 pages fill the entire 32K window before any output is produced.

### Cited Findings
- **Base model and goal:** the README calls Unlimited-OCR an effort "aiming to push Deepseek-OCR one step further" — [README.md:44](README.md). Upstream links: GitHub https://github.com/baidu/Unlimited-OCR, HF https://huggingface.co/baidu/Unlimited-OCR, arXiv 2606.23050 — [README.md:13-23](README.md).
- **Architecture:**
  - The paper describes "DeepEncoder paired with a Mixture-of-Experts (MoE) architecture that enjoys 3B total and 500M activated parameters", with vanilla MHA replaced by R-SWA — [paper p.5 §3.2](Unlimited-OCR.pdf).
  - The figure labels the decoder "3B-MOE-A570M" — [paper p.5 Fig. 2](Unlimited-OCR.pdf).
  - The model config shows `hidden_size` 1280, 12 layers, 10 heads, 64 routed + 2 shared experts, 6 experts per token, `use_mla: false`, `max_position_embeddings: 32768` and `sliding_window_size: 128` — [/data/hf-cache/hub/models--baidu--Unlimited-OCR/snapshots/07dea832.../config.json](file:///data/hf-cache/hub/models--baidu--Unlimited-OCR/snapshots/07dea832e22aefee32ad281d4b80551282e1c168/config.json).
- **DeepEncoder:**
  - DeepEncoder is SAM-ViT cascaded into CLIP-ViT with 16x token compression at the bridge — [paper p.5-6 §3.3](Unlimited-OCR.pdf).
  - It "can compress a 1024×1024 PDF-image to just 256 tokens", and the visual tokens "are encoded once and remain static throughout the entire long-horizon parsing process" — [paper p.6 §3.3](Unlimited-OCR.pdf).
- **R-SWA:**
  - Each generated token attends to all prefix tokens (visual tokens plus prompt) and to the previous n output tokens, with n = 128 by default — [paper p.1 Fig. 1; p.3](Unlimited-OCR.pdf).
  - The KV cache is bounded by L_m + n, versus L_m + T for MHA — [paper p.7 eq. 5-6](Unlimited-OCR.pdf).
  - The KV cache is implemented "as a queue with a capacity of m+n" — [paper p.5 Fig. 2 caption](Unlimited-OCR.pdf).
- **Resolutions:**
  - DeepEncoder has five native modes. Unlimited-OCR keeps two of them: "Base" (1024×1024, used for multi-page input) and "Gundam" (dynamic resolution, used for single pages) — [paper p.6](Unlimited-OCR.pdf).
  - Gundam is `base_size=1024, image_size=640, crop_mode=True`. Base is `base_size=1024, image_size=1024, crop_mode=False`. "Multi page / PDF only uses base (image_size=1024)" — [README.md:80-100](README.md).
  - The SGLang processor defines the presets tiny (512), small (640), base (1024), large (1280) and gundam (1024/640/crop). Multi-image requests accept only tiny, small or base; gundam with more than one image raises `ValueError` — [wheel sglang/srt/multimodal/processors/unlimited_ocr.py:19-50](wheel/sglang-0.0.0.dev11416+g92e8bb79e-py3-none-any.whl).
- **Image tokens per page:**
  - Formula, in [vllm-image:unlimited_ocr.py:135-166](vllm/vllm-openai:unlimited-ocr):
    - global view = 16·(16+1) = 272
    - local tiles = (h_tiles·10)·(w_tiles·10+1)
    - total = global + local + 1
  - Base mode: 273 tokens per page.
  - Gundam mode: 273 when the image fits within 640×640. Larger images are tiled into up to 32 crops of 640×640 (versus 6 for DeepSeek-OCR) — [vllm-image:unlimited_ocr.py:52-53,103-110](vllm/vllm-openai:unlimited-ocr). The maximum is the 4×8 grid, at 273 + 80·41 = 3,553 tokens per page (computed from the formula above).
  - The Transformers multi-page path builds `([image_token_id]*num_queries + [image_token_id])*num_queries + [image_token_id]` per page, with `num_queries = ceil((1024/16)/4) = 16`, which also gives 273 — [hf-snapshot:modeling_unlimitedocr.py:1185-1206](file:///data/hf-cache/hub/models--baidu--Unlimited-OCR/snapshots/8249d20b2822a7d8a4c22c8d8c9069b5ac36bda3/modeling_unlimitedocr.py).
- **Context length:**
  - Training used "a maximum sequence length of 32K", with all data "packed into a sequence length of 32K tokens" — [paper p.8 §4.1-4.2](Unlimited-OCR.pdf).
  - The SGLang server uses `--context-length 32768` — [README.md:173](README.md), [infer.py:32](infer.py).
  - The vLLM compose file uses `--max-model-len=32768` — [api/docker-compose.yml:14](api/docker-compose.yml).
- **The paper's own "not unlimited" admission:**
  - "Our model cannot achieve truly unlimited parsing under a finite context length (e.g., 32K), as it is also constrained by the prefill length... the prefill still becomes very long as the number of pages accumulates" — [paper p.11 §7](Unlimited-OCR.pdf).
  - Planned fixes: 128K-context models, and a "prefill pool" in which the model fetches KV chunks on demand — [paper p.11 §7](Unlimited-OCR.pdf).
- **Marketing claims, which partly contradict the limitation:**
  - "transcribe dozens of pages ... in a single forward pass under a standard maximum length of 32K" — [paper p.1](Unlimited-OCR.pdf).
  - "enables one-shot parsing of an entire book" — [paper p.3](Unlimited-OCR.pdf).
  - "can prefill tens to hundreds of document pages in a single pass" — [paper p.10 §5.4](Unlimited-OCR.pdf).
- **Accuracy:** 93.23 overall on OmniDocBench v1.5 (DeepSeek-OCR: 87.01) and 93.92 on v1.6 — [paper p.9 Table 1](Unlimited-OCR.pdf).
- **Model-revision pin:** HF commit `8249d20` (2026-10-14) renamed `model_type` to `unlimited_ocr`, which the vLLM image does not recognise, so compose pins revision `07dea832` — [api/README.md:18-20](api/README.md), [api/docker-compose.yml:6-8](api/docker-compose.yml).

### Inferences
- At 273 tokens per page in base mode, the pure prefill ceiling in a 32,768-token context is about 120 pages (32768 / 273), leaving zero tokens for output. "Hundreds of pages in a single pass" is therefore not achievable with the released 32K model. That claim conflicts with the paper's own §7 limitation.
- In practice the useful range for one-shot parsing is the range the paper actually evaluated: up to "40+" pages (Table 3). A 300-page PDF must be split into chunks, whatever the inference engine.

### Gaps
- The paper does not state the tokens-per-page figure directly. The value 273 is derived from code (256 + 16 row-newline tokens + 1 separator).
- The paper does not say what output budget the "40+" page tests used, or how output length relates to the 32K limit. See section 2.

## 2. One page per request, or several? What do the paper and README say about long or multi-page documents?

### Takeaway
The model supports several pages per request: a single `<image>` prompt with N page images in base mode only, using the prompt "Multi page parsing." and an n-gram window of 1024. Pages are separated in the output by a `<page>` token learned in training. The paper evaluated up to "40+" pages, where quality degrades: edit distance 0.107 versus 0.036 at 2 pages. In vLLM, R-SWA is only a mask over a full KV cache, so the constant-memory benefit does not apply in the API's serving stack. Upstream users report that page separators are emitted unreliably, especially under vLLM.

### Cited Findings
- **Multi-page API surface:**
  - Transformers: `model.infer_multi(..., image_files=[...], image_size=1024, max_length=32768, no_repeat_ngram_size=35, ngram_window=1024)`. The README's PDF recipe rasterises with PyMuPDF at 300 DPI and passes all pages into one `infer_multi` call — [README.md:94-130](README.md).
  - SGLang: `generate("Multi page parsing.", pdf_to_images(..., dpi=300), image_mode="base", ngram_window=1024)` sends all pages in one streaming request with `timeout=1200` — [README.md:223-269](README.md).
- **How `infer_multi` builds the input:**
  - All page images are concatenated at one `<image>` position, with a single separator token between images. It "Does NOT support crop mode" — [hf-snapshot:modeling_unlimitedocr.py:1134-1146, 1196-1210](file:///data/hf-cache/hub/models--baidu--Unlimited-OCR/snapshots/8249d20b2822a7d8a4c22c8d8c9069b5ac36bda3/modeling_unlimitedocr.py).
  - `max_length=32768` is passed to HF `generate`, so prompt plus output share 32,768 tokens — [hf-snapshot:modeling_unlimitedocr.py:1244](file:///data/hf-cache/hub/models--baidu--Unlimited-OCR/snapshots/8249d20b2822a7d8a4c22c8d8c9069b5ac36bda3/modeling_unlimitedocr.py).
- **Training data:**
  - Single-page to multi-page ratio is 9:1. There are about 200k synthetic multi-page samples of 2 to 50 pages, concatenated with `<page>` as the separator — [paper p.8 §4.1](Unlimited-OCR.pdf).
  - Training ran 4,000 steps at 32K with the encoder frozen — [paper p.8 §4.2](Unlimited-OCR.pdf).
- **Long-horizon results:**

  | Pages | Edit distance | Distinct-35 |
  |---|---|---|
  | 2 | 0.0362 | 99.87% |
  | 5 | 0.0452 | 99.98% |
  | 10 | 0.0526 | 99.83% |
  | 15 | 0.0787 | 99.99% |
  | 20 | 0.0572 | 99.89% |
  | 40+ | 0.1069 | 96.90% |

  Source: [paper p.11 Table 3](Unlimited-OCR.pdf). The authors attribute the errors to small text being unreadable at Base 1024×1024 resolution, "rather than R-SWA losing direction" — [paper p.10 §5.4](Unlimited-OCR.pdf). The in-house set is books of 2, 5, 10, 20 and 40+ pages, at least 10 books per category — [paper p.8 §5.1](Unlimited-OCR.pdf).
- **Output volume per page:**
  - The paper assumes "a compression ratio of 1:10 between visual and text tokens ... 10K visual tokens (equivalent to roughly 20-30 pages at 1024×1024) demand an output length of 100k+ tokens" — [paper p.6 §3.4](Unlimited-OCR.pdf).
  - That implies roughly 2,700 output tokens per dense page.
- **Throughput:**
  - 5,580 TPS at 512 concurrency on OmniDocBench in Base mode, versus 4,951 for DeepSeek-OCR — [paper p.10](Unlimited-OCR.pdf).
  - Theoretical TPS stays at about 7,850 out to 6,144 output tokens — [paper p.11 Table 4](Unlimited-OCR.pdf).
- **How vLLM implements R-SWA:** "We reproduce this (Level 1: full KV cache + custom mask) by forcing the language model onto the FlexAttention backend and installing an R-SWA mask_mod" — [vllm-image:unlimited_ocr.py:21-32](vllm/vllm-openai:unlimited-ocr).
  - The KV cache therefore still grows with output, and every generated position consumes `max_model_len`.
  - Constant-memory KV management exists only in Transformers ("ring buffer in SlidingWindowLlamaAttention") and SGLang — [hf-snapshot:modeling_unlimitedocr.py:1228-1232](file:///data/hf-cache/hub/models--baidu--Unlimited-OCR/snapshots/8249d20b2822a7d8a4c22c8d8c9069b5ac36bda3/modeling_unlimitedocr.py), [paper p.8 §4.2](Unlimited-OCR.pdf).
- **vLLM multi-image behaviour:**
  - Multi-image requests automatically fall back to non-crop (base) mode — [vllm-image:unlimited_ocr.py:55-63, 208-216](vllm/vllm-openai:unlimited-ocr).
  - DeepSeek-OCR's processing info declares `get_supported_mm_limits -> {"image": None}`, meaning no per-prompt image cap — [vllm-image deepseek_ocr.py:212-213](vllm/vllm-openai:unlimited-ocr).
- **Official vLLM recipe:**
  - `max_tokens=8192`, `temperature=0`, `skip_special_tokens=False`, `vllm_xargs={"ngram_size":35,"window_size":128}`.
  - "use window_size=1024 for multi-page / PDF input".
  - The NGram logits processor is "required" because "without it long documents loop on <|det|> coordinate tokens".
  - Client timeout is 3600 s. No page-per-request maximum is stated.
  - Source: https://recipes.vllm.ai/baidu/Unlimited-OCR
- **HF model card:** it repeats the README (base only for multi-page, `max_length=32768`, `ngram_window=1024`, DPI 300) and states no page maximum — https://huggingface.co/baidu/Unlimited-OCR
- **Upstream issue #107 (2026-10-18):** with 2-3 pages per request, "page-separation markers are missing or incorrectly placed ... Text from multiple pages may be merged". The reporter sees it in both vLLM and Transformers, more often with vLLM — https://github.com/baidu/Unlimited-OCR/issues/107
- **Upstream issue #55:** looping on some pages, because `no_repeat_ngram_size=35` misses repeating units shorter than 35 tokens.
  - Base mode loops about 1.2% of the time versus 0.24% for Gundam (figures reported in the thread).
  - The looping page in the test only degenerated at `max_length >= 8192`.
  - Source: https://github.com/baidu/Unlimited-OCR/issues/55

### Inferences
- Under vLLM, a multi-page request with N pages needs 273·N + output ≤ 32,768.
- Taking the paper's roughly 2,700 output tokens per dense page, about 11 dense pages fit before the context fills (273·11 + 2,700·11 ≈ 32.7K).
  - Text-light pages allow more.
  - This is consistent with the paper's "dozens of pages" and 20-40 page tests. It is an estimate, not a measured figure.
- Even in Transformers or SGLang, where the KV cache is bounded, `max_length=32768` in HF `generate` still counts prompt plus generated tokens. Whether SGLang's context-length check counts R-SWA output positions was not verified.
- Splitting into multi-page chunks needs page boundaries. Because `<page>` emission is unreliable (#107), chunking at 1 page per request, or reconciling chunk outputs with page numbers, is safer for page-level mapping.

### Gaps
- No upstream source states the maximum number of pages per request, or a recommended chunk size for books of 100+ pages.
- How SGLang counts R-SWA output tokens against `--context-length` was not verified in code. The wheel's `pure_swa_radix_cache.py` and `managers/scheduler.py` reference Unlimited-OCR but were not read.
- No measured per-page latency on this deployment's GPU was available locally.

## 3. How do `infer.py` and the REST API (`api/`) handle PDFs today?

### Takeaway
Both code paths rasterise every page with PyMuPDF up front and default to one request per page.
- `infer.py` writes 300-DPI PNGs to a temp directory that is never removed, sends one streaming SGLang request per page from 8 threads, and writes one `.md` file per page. There is no merged output, no resume, and failures are silently empty.
- The FastAPI service holds the upload and every rendered 200-DPI PNG in memory. It runs one synchronous HTTP request per PDF and rejects PDFs over 200 pages or 50 MB. It fans pages out to vLLM under a process-wide semaphore of 8, or sends all pages in one request (`whole_document`). It joins the per-page markdown into one JSON response.

### Cited Findings: `infer.py` (SGLang batch script)
- **Constants:**
  - `PDF_DPI=300`, `CONTEXT_LENGTH=32768`, `NO_REPEAT_NGRAM_SIZE=35`, `NGRAM_WINDOW=128`, `REQUEST_TIMEOUT=1200`, `MAX_RETRIES=5`, `PROMPT="document parsing."` — [infer.py:25-37](infer.py).
  - The SGLang server is started with `--page-size 1 --mem-fraction-static 0.8 --context-length 32768` — [infer.py:94-117](infer.py).
- **Rasterisation:**
  - `pdf_to_images` opens the PDF with `fitz` and renders every page with `Matrix(dpi/72)` at 300 DPI.
  - It saves `page_NNNN.png` into `tempfile.mkdtemp(prefix="pdf_ocr_")` before any inference starts.
  - Source: [infer.py:50-62](infer.py), called in [infer.py:240-250](infer.py).
- **Per-page requests:**
  - Each page becomes one job, and the module docstring says "each converted page is sent as one request" — [infer.py:1-7](infer.py).
  - `build_content` sends a single image per request — [infer.py:73-74](infer.py).
  - Despite the README's multi-page recipe, `infer.py` never sends multiple pages per request.
- **Image mode:**
  - `--image_mode` defaults to `gundam` — [infer.py:314](infer.py). It is passed as `images_config` — [infer.py:195](infer.py).
  - The n-gram window is hard-coded at 128 with no CLI override — [infer.py:197-202](infer.py). Upstream issue #17 confirms this.
- **Concurrency:**
  - A `ThreadPoolExecutor(max_workers=args.concurrency)` with default 8 submits every page at once — [infer.py:277-283](infer.py), [infer.py:311](infer.py).
  - Requests use `requests.post(..., timeout=1200, stream=True)`.
  - Retries happen on 502 or on any exception, up to 5 attempts with linear back-off of 3·(attempt+1) s — [infer.py:205-227](infer.py).
- **Failure handling:**
  - After 5 failed attempts the page returns `{"tokens":0,"text":""}` — [infer.py:226-227](infer.py).
  - The output file is opened in `"w"` mode before streaming, so a failed page leaves an empty or partial `.md` — [infer.py:155](infer.py).
- **Output:**
  - One file per page, `{prefix}_page_{i:04d}.md`, written incrementally while streaming — [infer.py:247-248](infer.py), [infer.py:177-178](infer.py).
  - There is no merged document file. The run prints only aggregate statistics — [infer.py:285-300](infer.py).
  - No `max_tokens` is sent, so the server limit applies (context 32,768 minus prompt).
- **No resume:** existing outputs are not checked, so the whole PDF is re-rendered and every page re-requested — [infer.py:240-283](infer.py). Upstream issue #17 notes "no --resume" and the tmpdir leak: https://github.com/baidu/Unlimited-OCR/issues/17

### Cited Findings: REST API (`api/`, FastAPI + vLLM)
- **Configuration defaults** — [api/app/core/config.py:12-28](api/app/core/config.py), [api/.env.example](api/.env.example):

  | Setting | Default |
  |---|---|
  | `request_timeout_seconds` | 1200 |
  | `max_tokens` | 8192 |
  | `ngram_size` | 35 |
  | `single_image_window` | 128 |
  | `multi_image_window` | 1024 |
  | `max_upload_mb` | 50 |
  | `max_pdf_pages` | 200 |
  | `pdf_dpi` | 200 (allowed 72-600) |
  | `page_concurrency` | 8 |

- **vLLM server flags:** `--max-model-len=32768`, `--no-enable-prefix-caching`, `--mm-processor-cache-gb=0`, the NGram logits processor, and one GPU — [api/docker-compose.yml:4-26](api/docker-compose.yml).
- **Upload handling:**
  - The upload is read in 1 MB chunks and rejected with 413 once it exceeds 50 MB. Otherwise all bytes are joined in memory — [api/app/api/dependencies.py:16-27](api/app/api/dependencies.py).
  - The endpoint is a single synchronous `POST /v1/ocr/pdf` returning `OcrResponse`. There is no job ID, polling, streaming or webhook — [api/app/api/routes.py:44-62](api/app/api/routes.py).
- **Rasterisation:**
  - `PyMuPdfRenderer.render` checks the `%PDF-` signature, rejects password-protected and empty PDFs, and rejects `page_count > max_pages` with `DocumentTooLargeError` (HTTP 413).
  - It then returns a list of every page as PNG bytes (`get_pixmap(dpi=200).tobytes("png")`), all held in memory — [api/app/infrastructure/pdf_renderer.py:18-38](api/app/infrastructure/pdf_renderer.py).
  - Rendering runs off the event loop through `asyncio.to_thread`, but must finish for all pages before any OCR starts — [api/app/services/ocr_service.py:31](api/app/services/ocr_service.py).
- **`per_page` strategy (default):**
  - `asyncio.gather` over every page's `_recognize([image])` — [api/app/services/ocr_service.py:34-38](api/app/services/ocr_service.py).
  - Concurrency is bounded by a single `asyncio.Semaphore(page_concurrency)` created once per service, so all HTTP requests share it — [api/app/services/ocr_service.py:24, 43-46](api/app/services/ocr_service.py).
  - `gather` is called without `return_exceptions`. One failed page raises, and the whole request fails with 502 or 503 while the other page tasks keep running.
- **`whole_document` strategy:** all pages go into one request (`[await self._recognize(pages, page=None)]`), and the result has `page=None` — [api/app/services/ocr_service.py:32-33](api/app/services/ocr_service.py).
- **vLLM payload:**
  - The single-image prompt is `"<image>document parsing."` with window 128. The multi-image prompt is `"<image>Multi page parsing."` with window 1024.
  - Each payload sets `max_tokens=8192`, `temperature 0`, `skip_special_tokens False` and `vllm_xargs` n-gram parameters.
  - Images are inlined as base64 data URLs.
  - Source: [api/app/infrastructure/vllm_engine.py:10-12, 44-58, 79-81](api/app/infrastructure/vllm_engine.py).
  - `images_config` is not sent. vLLM uses gundam (crop) for single-image requests and base for multi-image requests automatically — [vllm-image:unlimited_ocr.py:208-216](vllm/vllm-openai:unlimited-ocr).
- **No streaming:**
  - The engine does a non-streaming `client.post` and reads `choices[0].message.content` — [api/app/infrastructure/vllm_engine.py:60-76](api/app/infrastructure/vllm_engine.py).
  - `finish_reason` is never checked, so truncation at `max_tokens` (`"length"`) passes silently.
- **HTTP client:**
  - `httpx.AsyncClient` uses `Timeout(1200, connect=10)` and `Limits(max_connections=page_concurrency*2)`, which is 16 — [api/app/main.py:42-47](api/app/main.py).
  - A timeout maps to `EngineUnavailableError` (503) — [api/app/infrastructure/vllm_engine.py:63-64](api/app/infrastructure/vllm_engine.py), [api/app/api/error_handlers.py:16-21](api/app/api/error_handlers.py).
- **Output:**
  - Each page's raw output is parsed into `<|det|>category [bbox]<|/det|>` blocks, with bbox on a 0-999 grid, then converted to markdown — [api/app/domain/parser.py:16-48](api/app/domain/parser.py).
  - The response holds `page_count`, top-level `markdown` (page markdown joined with `"\n\n"`) and `pages[]`, with optional `raw` per page — [api/app/api/schemas.py:38-51](api/app/api/schemas.py).
  - In `whole_document` mode the parser ignores `<page>` separators, so the output is one "page" with `page: null` — [api/app/domain/models.py:29](api/app/domain/models.py), [api/app/domain/parser.py:23-42](api/app/domain/parser.py).
- **The API's own documentation:**
  - `whole_document`: "total output dibatasi konteks 32k token" (total output is limited by the 32k-token context) — [api/README.md:34](api/README.md).
  - The route docstring says "bounded by the model's 32k context" — [api/app/api/routes.py:52-56](api/app/api/routes.py).
- **Server:** a single uvicorn process with no worker count, as `CMD ["uvicorn", "app.main:app", ...]` — [api/Dockerfile:17](api/Dockerfile).

### Inferences
- With `include_raw=true`, the response roughly doubles in size because every page's raw output is included.
- The `httpx` `Timeout(1200)` applies separately to each phase (connect, read, write, pool). It is not a total-request deadline. A per-page request only fails if no bytes arrive for 1,200 s, which for a non-streaming completion means generation itself takes longer than 1,200 s.

### Gaps
- No measured end-to-end time per page on this hardware was available, so total wall time for 300 pages cannot be stated from evidence.

## 4. Concrete bottlenecks and failure points for a 300+ page PDF

### Takeaway
With default settings the API rejects a 300-page PDF immediately (HTTP 413, `max_pdf_pages=200`), and a scanned 300-page PDF may also exceed the 50 MB upload limit. Raising those limits exposes deeper problems:
- `whole_document` cannot work: about 82K prefill tokens against a 32K context.
- `per_page` runs as one long, all-or-nothing synchronous HTTP request with every page image held in RAM.
- The run has no progress reporting, no partial results and no resume.
- Output truncation at 8,192 tokens and repetition loops pass silently.

`infer.py` handles 300 pages more gracefully, because output goes to per-page files. Its weaknesses are no resume, silent empty pages, a leaked temp directory, and per-page requests that never use the model's multi-page capability.

### Cited Findings and derived limits
1. **Hard page cap:** `max_pdf_pages=200`, so a PDF over 200 pages gets 413 "PDF has N pages; the limit is 200" — [api/app/core/config.py:26](api/app/core/config.py), [api/app/infrastructure/pdf_renderer.py:32-35](api/app/infrastructure/pdf_renderer.py).
2. **Upload cap:** 50 MB — [api/app/core/config.py:25](api/app/core/config.py), [api/app/api/dependencies.py:18-23](api/app/api/dependencies.py). Scanned 300-page PDFs commonly exceed this. This is an inference; the actual size depends on the scan.
3. **`whole_document` overflows the context:**
   - 300 × 273 = 81,900 prefill tokens, plus `max_tokens` 8,192, is far above `max_model_len` 32,768 (sources: token formula in §1; [api/docker-compose.yml:14](api/docker-compose.yml); [api/app/core/config.py:19](api/app/core/config.py)). vLLM rejects prompt + max_tokens > max_model_len with a 400 error, which the API surfaces as 502 `EngineResponseError` — [api/app/infrastructure/vllm_engine.py:67-68](api/app/infrastructure/vllm_engine.py).
   - The maximum is about (32,768 − 8,192) / 273 ≈ 90 pages per `whole_document` request. Even then the entire document's output is capped at 8,192 tokens, roughly 3 dense pages at the paper's ~2,700 tokens per page (paper p.6). Output is silently truncated because `finish_reason` is not checked — [api/app/infrastructure/vllm_engine.py:71-76](api/app/infrastructure/vllm_engine.py).
   - Under vLLM, R-SWA is "full KV cache + custom mask", so there is no KV-memory relief for long outputs — [vllm-image:unlimited_ocr.py:21-32](vllm/vllm-openai:unlimited-ocr).
4. **Whole PDF materialised in memory before OCR:**
   - The upload bytes, then a list of every page PNG at 200 DPI. An A4 page is about 1654×2339 px, which is about 11.6 MB as an uncompressed RGB pixmap during rendering. The PNG output is kept for all pages — [api/app/infrastructure/pdf_renderer.py:36-38](api/app/infrastructure/pdf_renderer.py). Pixel dimensions are computed; PNG sizes vary with content.
   - Each page is base64-encoded again (+33%) per request — [api/app/infrastructure/vllm_engine.py:79-81](api/app/infrastructure/vllm_engine.py).
   - In `whole_document`, a single JSON payload carries every page as base64.
   - Rendering 300 pages is sequential and CPU-bound in one thread, and nothing is sent to vLLM until it finishes — [api/app/services/ocr_service.py:31](api/app/services/ocr_service.py).
5. **One synchronous HTTP request for the whole document:**
   - There is no job queue, streaming or progress reporting — [api/app/api/routes.py:44-62](api/app/api/routes.py).
   - Wall time scales as about 300 pages / 8 concurrent × per-page latency. Any client, proxy or load-balancer timeout (often 60-600 s) will cut the connection. The API itself has no request deadline beyond the per-phase 1,200 s httpx timeout — [api/app/main.py:45](api/app/main.py). This is an inference.
   - If the client disconnects, the server keeps processing all pages anyway; nothing cancels them on disconnect.
6. **All-or-nothing failure:**
   - `asyncio.gather` without `return_exceptions` means one page's vLLM error or timeout fails the whole 300-page response, and every completed page is discarded — [api/app/services/ocr_service.py:34-38](api/app/services/ocr_service.py).
   - There are no retries in the API — [api/app/infrastructure/vllm_engine.py:60-69](api/app/infrastructure/vllm_engine.py). Compare `infer.py`, which retries 5 times — [infer.py:205-227](infer.py).
   - There is no persistence or resume.
7. **Global semaphore:**
   - `page_concurrency=8` is shared across all callers. One 300-page job holds all 8 slots for its whole duration, and other users' single-image requests wait behind it — [api/app/services/ocr_service.py:24, 44](api/app/services/ocr_service.py).
   - Eight concurrent requests is very low against vLLM's batching capacity: the paper's throughput figures are measured at 512 concurrency (paper p.10). Raising the value also requires raising `max_connections`, currently 2 × concurrency — [api/app/main.py:46](api/app/main.py).
8. **Silent truncation and looping per page:**
   - `max_tokens=8192` per page — [api/app/core/config.py:19](api/app/core/config.py). Without a `finish_reason` check, a looping page burns the full 8,192 tokens, which is the worst-case latency, and returns garbage without any flag.
   - Issue #55 reports looping in about 1.2% of base-mode pages and 0.24% of gundam pages, triggered at `max_length >= 8192`. Over 300 pages that is a nontrivial expected number of looped pages — https://github.com/baidu/Unlimited-OCR/issues/55
9. **Rasterisation DPI and mode mismatch between code paths:**

   | Path | DPI | Mode |
   |---|---|---|
   | API | 200 ([api/app/core/config.py:27](api/app/core/config.py)) | gundam single-page, chosen implicitly by vLLM |
   | `infer.py` | 300 ([infer.py:26](infer.py)) | gundam |
   | README / HF card | 300 | base (multi-page) |

   - Gundam per-page prefill can reach 3,553 tokens (32 tiles), against 273 for base. That means more prefill compute per page but better small-text accuracy. The paper blames multi-page errors on base-mode resolution — [paper p.10](Unlimited-OCR.pdf).
10. **`infer.py` specifics at 300 pages:**
    - Every page is rendered to a 300-DPI PNG before any request (about 2480×3508 px for A4), and the temp directory is never deleted — [infer.py:50-62](infer.py); issue #17.
    - All 300 futures are created up front — [infer.py:278-281](infer.py).
    - A page that fails 5 times silently becomes an empty `.md` — [infer.py:155, 226-227](infer.py).
    - No `--resume`: a crash means re-rendering and re-running every page.
    - No merged output document — [infer.py:247-248](infer.py).
    - The SGLang server-start wait is capped at 300 s — [infer.py:25, 125-136](infer.py).
11. **Page boundaries if chunking multi-page:** `<page>` separators are unreliable (issue #107), and the API parser does not split on `<page>` at all — [api/app/domain/parser.py:23-42](api/app/domain/parser.py).

### Inferences
- For 300+ pages under the current architecture, the realistic path is `per_page` (or small multi-page chunks of about 10-20 pages in base mode, if cross-page context matters). It would need:
  - an async job model with persisted per-page results and progress
  - per-page retries and isolated failures
  - streamed, lazy rendering instead of rendering all pages up front
  - `finish_reason` / loop detection
  - higher concurrency to saturate vLLM

  These are design implications, not upstream guidance.
- Chunked `whole_document` would need `max_tokens` budgeted per chunk as 32,768 − 273·pages, and page re-mapping from `<page>` markers, which are unreliable.

### Gaps
- vLLM's exact 400 error text for over-length prompts was not captured from a live run.
- Real per-page latency, PNG sizes, and peak RSS for this deployment were not measured.
- No official upstream guidance exists on processing documents of 300+ pages. The paper's future-work "prefill pool" and 128K model are unreleased as of 2026-10-22, and none was found on the HF card or the recipe.
