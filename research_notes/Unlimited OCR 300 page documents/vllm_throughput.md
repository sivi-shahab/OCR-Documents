# vLLM Throughput and Memory Stability for Page-Level VLM OCR (300+ page documents)

Scope: vLLM settings and serving patterns for high-throughput, memory-stable OCR with vision-language models. Researched October 2026. Note: published OCR benchmarks mostly use vLLM v0.10.2 to v0.11.x (late 2025). Current vLLM docs are a "latest" developer preview (the optimization page mentions features gated on v0.23.0+). Defaults and flag names have changed between versions, so check them against the vLLM version you deploy.

## Q1. Which vLLM settings matter most for VLM OCR throughput?

### Takeaway
Two things drive VLM OCR throughput. The first is the per-page visual token budget (image pixel limits, which set prefill cost and KV use). The second is how many pages run concurrently (max_num_seqs, max_num_batched_tokens, gpu_memory_utilization, KV headroom). For OCR, turn off prefix caching and the multimodal processor cache, as the official DeepSeek-OCR recipe does. On multi-GPU nodes, prefer data parallelism (one replica per GPU) for small (≤3B to 7B) OCR models. CPU-side image preprocessing is a frequent hidden bottleneck.

### Cited Findings
- **Chunked prefill** is on by default in V1 "whenever possible", and decode requests are batched before prefills. `max_num_batched_tokens` trades ITL against TTFT. Small values such as 2048 favor inter-token latency. For throughput, the docs recommend a value above 8192, "especially for smaller models on large GPUs." With chunked prefill disabled, `max_num_batched_tokens` must exceed `max_model_len` or startup may fail — [vLLM Optimization and Tuning](https://docs.vllm.ai/en/latest/configuration/optimization.html)
- **Encoder budget coupling:** the vLLM v1 encoder cache manager derives its compute budget from max(scheduler encoder-input limit, largest per-item multimodal token count). It raises a ValueError if chunked MM input is disabled and a single image's token count exceeds `max_num_batched_tokens`. In practice, `max_num_batched_tokens` must be at least the token count of your largest page image — [vLLM encoder_cache_manager v0.9.1 API](https://docs.vllm.ai/en/v0.9.1/api/vllm/v1/core/encoder_cache_manager.html)
- The encoder cache shares embeddings across requests by content hash. Unreferenced entries are evicted oldest-first — [vLLM encoder_cache_manager v0.11.2](https://docs.vllm.ai/en/v0.11.2/api/vllm/v1/core/encoder_cache_manager/)
- **Preemption:** when KV cache runs out, vLLM preempts requests and recomputes them later. The V1 default mode is RECOMPUTE, not SWAP. Mitigations: raise `gpu_memory_utilization`, lower `max_num_seqs` or `max_num_batched_tokens`, or add TP/PP. Monitor through Prometheus or `disable_log_stats=False` — [vLLM Optimization](https://docs.vllm.ai/en/latest/configuration/optimization.html)
- **gpu_memory_utilization** is the fraction of GPU memory vLLM pre-allocates. The default is 0.9 — [vLLM Optimization](https://docs.vllm.ai/en/latest/configuration/optimization.html); [Spheron DeepSeek-OCR guide](https://www.spheron.network/blog/deploy-deepseek-ocr-gpu-cloud/)
- **Parallelism:** TP is for models that don't fit on one GPU and frees KV memory, but adds sync overhead. DP (`data_parallel_size=N`) replicates the model to scale throughput — [vLLM Optimization](https://docs.vllm.ai/en/latest/configuration/optimization.html)
- **Batch-level DP for the vision encoder:** `mm_encoder_tp_mode="data"` shards the encoder over the input batch instead of the weights, while the LLM decoder keeps TP. Reported gains are about 10% throughput/TTFT at TP=8, and about 40% more for Conv3D-based encoders. Encoder weights are replicated per rank, so a tight fit can OOM. The model must set `supports_encoder_tp_data = True`; supported models include Qwen2-VL and later, InternVL, Llama4 and Step3 — [vLLM Optimization](https://docs.vllm.ai/en/latest/configuration/optimization.html)
- **Multimodal processor cache:** `mm_processor_cache_gb` defaults to 4 GiB. `mm_processor_cache_type="shm"` suits TP>1 or multi-worker setups. Setting `mm_processor_cache_gb=0` disables both IPC and processor caching — [vLLM Optimization](https://docs.vllm.ai/en/latest/configuration/optimization.html)
- **Official DeepSeek-OCR recipe:** `enable_prefix_caching=False` and `mm_processor_cache_gb=0`, because "OCR workloads don't benefit much from them, and they add hashing overhead." It also recommends tuning `max_num_batched_tokens` per hardware. Server: `vllm serve deepseek-ai/DeepSeek-OCR --logits_processors vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor --no-enable-prefix-caching --mm-processor-cache-gb 0` — [vLLM Recipes: DeepSeek-OCR](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-OCR.html)
- **Input processing / CPU:** for online serving, `--api-server-count` scales API processes and `-dp` adds engine cores, which helps when input processing is the bottleneck. Each API server loads media with 8 threads by default (`VLLM_MEDIA_LOADING_THREAD_COUNT`). Plan for at least `2 + N` physical cores for N GPUs — [vLLM Optimization](https://docs.vllm.ai/en/latest/configuration/optimization.html)
- **CPU contention example (weak source):** a third-party 0.6B MinerU2.5-Pro derivative reports going from 2.63 to 3.60 pages/s (+37%) on A100 with vLLM 0.11.x "by eliminating CPU thread contention during image preprocessing" — [Friendli model page](https://friendli.ai/models/aijiag/MinerU2.5-Pro-2605-0.6B)
- **Memory levers for multimodal:** `limit_mm_per_prompt={"image": N}` caps items per prompt. Size hints such as `{"image": {"count": 1, "width": W, "height": H}}` affect only memory profiling. `mm_processor_kwargs={"max_pixels": 768*768}` for Qwen2-VL-style models (default 1280*28*28), or `{"max_dynamic_patch": 4}` for InternVL (default 12). Other levers: lower `max_model_len`/`max_num_seqs`; `enforce_eager=True` or fewer `cudagraph_capture_sizes` (captured up to `max_num_seqs` by default); quantization — [vLLM Conserving Memory](https://docs.vllm.ai/en/latest/configuration/conserving_memory.html)
- MinerU2.5 says it tuned `max_num_batched_tokens`, `max_num_seqs` and `cuda_graph_sizes` (values not published). It reports going from 0.95 pages/s unoptimized to 2.12 pages/s on A100 with deployment optimizations — [MinerU2.5 paper](https://arxiv.org/html/2509.22186)
- PaddleOCR-VL tuned `max-num-batched-tokens` and `gpu-memory-utilization` for its benchmark (values not given) — [PaddleOCR-VL report](https://arxiv.org/html/2510.14528)
- olmOCR's pipeline exposes `--gpu-memory-utilization`, `--max_model_len` (examples use 16384), `--tensor-parallel-size` and `--data-parallel-size`, and defaults to an FP8 model, which "runs significantly faster" and needs fewer retries — [olmOCR GitHub](https://github.com/allenai/olmocr)

### Inferences
- For a ≤3B OCR model on one GPU, a reasonable starting point is: `gpu_memory_utilization` 0.85–0.92, `max_num_batched_tokens` ≥ 8192 and at least the largest page's visual tokens plus prompt, `max_model_len` = visual tokens + prompt + `max_tokens` cap (for example 8k–16k), `limit_mm_per_prompt={"image":1}`, prefix caching off, `mm_processor_cache_gb=0`. Then raise `max_num_seqs` (for example 64 → 256) until preemption counters start rising, and back off.
- On multi-GPU nodes, a small OCR model gets more throughput from DP (or N independent servers behind a balancer) than from TP, because TP's sync overhead is not worth it when the model already fits.
- Watch CPU utilization. Rendering PDFs and resizing images can starve the GPU, so rasterize pages in a separate process pool.

### Gaps
- No published ablation isolates each knob (for example `max_num_seqs` 64 vs 256) for an OCR VLM. MinerU2.5 and PaddleOCR-VL say they tuned but don't publish values.
- I found no source on whether prefix caching helps OCR when a long shared system prompt is used (the DeepSeek-OCR recipe uses a tiny prompt).

## Q2. Offline batch (LLM.generate) vs online server with concurrent async requests

### Takeaway
I found no authoritative head-to-head benchmark for OCR VLMs. Both paths use the same engine and scheduler. Offline `LLM.generate` with all pages submitted at once lets the scheduler keep the batch full with no HTTP or base64 overhead. Online serving reaches similar throughput only if the client keeps enough requests in flight to saturate `max_num_seqs`. Production OCR pipelines (olmOCR, PaddleOCR-VL, dots.ocr) use a server or engine with a large client-side concurrency (64–200 threads) and pipelined rendering.

### Cited Findings
- `llm.generate` adds prompts to the engine's waiting queue and runs the engine "with high throughput." The online server implements the OpenAI protocol — [vLLM quickstart](https://docs.vllm.ai/en/v0.8.2/_sources/getting_started/quickstart.md)
- The DeepSeek-OCR recipe positions offline mode for batch OCR jobs in Python and online mode for OpenAI-compatible serving. Its online example uses a 3600 s client timeout — [vLLM Recipes: DeepSeek-OCR](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-OCR.html)
- olmOCR switched from SGLang to a vLLM-based pipeline in v0.1.75 (June 2025). It batches documents into work items of about 500 pages queued to GPU workers, can coordinate workers through S3, and supports an external `--server` URL. `--max_concurrent_requests` caps in-flight requests (examples set 20 for hosted providers) — [olmOCR GitHub](https://github.com/allenai/olmocr); [olmOCR paper](https://arxiv.org/html/2502.18443v3)
- PaddleOCR-VL's benchmark runs PDF rendering, layout detection and VLM inference in separate threads connected by queues. VLM batches fire when the queue hits a threshold or a wait-time limit, and PDFs are processed "in batches of 512." dots.ocr was run with 200 threads — [PaddleOCR-VL report](https://arxiv.org/html/2510.14528)
- dots.ocr's parser uses `--num_thread 64` for PDFs, with "try bigger num_threads for pdf with a large number of pages" — [dots.ocr GitHub](https://github.com/rednote-hilab/dots.ocr)
- A vLLM forum thread says splitting one page into multiple prompts repeats image preprocessing and inference, so total compute rises — [vLLM forum](https://discuss.vllm.ai/t/speeding-up-vllm-inference-for-qwen2-5-vl/615) (caveat: replies are partly from an automated assistant)

### Inferences
- For a single 300+ page document on a dedicated GPU, offline `LLM.generate` over all pages, or AsyncLLM with all pages submitted, is the simplest way to reach peak throughput. The online server is better when multiple jobs share the GPU or the service must stay up between jobs.
- Online client concurrency should be at or slightly above the engine's effective `max_num_seqs` (for example 64–256 for a 1–3B model). Too few in-flight requests leave the batch underfilled. Far too many only add queueing and timeout risk. Measure with `vllm bench serve` and watch `num_requests_waiting` and preemption metrics.
- Pipelining page rendering with inference (producer/consumer queues, as PaddleOCR-VL and olmOCR do) matters more for a 300-page job than the choice between offline and online.

### Gaps
- I found no published measurement of the throughput gap between offline and online for OCR VLMs on the same hardware. The only side-by-side found was a vendor blog on text LLMs with incomplete numbers ([databasemart](https://databasemart.com/blog/vllm-gpu-benchmark-h100)), which is not usable.

## Q3. Published throughput numbers for OCR VLMs on vLLM

### Takeaway
On a single A100, current small OCR VLMs reach about 1–2 pages/s end to end (PaddleOCR-VL 0.9B about 1.35, MinerU2.5 1.2B about 1.06–2.12 depending on who measures). The 3B dots.ocr manages about 0.3–0.35 pages/s, and the 7B olmOCR about 1.25 (L40S) to 4.2 (H100, FP8-era SGLang). Numbers vary by up to 2x between papers because of different datasets, pipelines and tuning.

### Cited Findings
- **PaddleOCR-VL report (Table 13), single A100, OmniDocBench v1.0 (981 pages), batches of 512 PDFs:**
  - PaddleOCR-VL (vLLM 0.10.2): 1.3453 pages/s, 2067.6 tok/s, 728.7 s total
  - PaddleOCR-VL on SGLang 0.5.2: 1.1115 pages/s; on FastDeploy 2.3: 1.6184 pages/s
  - MinerU2.5 (vLLM 0.10.2): 1.0574 pages/s, 1647.9 tok/s
  - MonkeyOCR-pro-1.2B: 0.6730 pages/s
  - dots.ocr (vLLM 0.10.2): 0.3522 pages/s, 532.9 tok/s

  Source: [PaddleOCR-VL arXiv 2510.14528](https://arxiv.org/html/2510.14528)
- **PaddleOCR-VL-1.5:** A100 vLLM 1.3797 pages/s (981.4 s), FastDeploy 2.0160 pages/s, "without extensive parameter tuning" — [PaddleOCR-VL-1.5 arXiv 2601.21957](https://arxiv.org/pdf/2601.21957) (figures from a search snippet; not verified against the full text)
- **MinerU2.5 own paper (Table 3), vLLM, 1.2B, OmniDocBench (1,355 pages, more than 1,100 tokens per page):** RTX 4090 48G 1.70 pages/s (1875.82 tok/s); A100 80G 2.12 pages/s (2337.25 tok/s); H200 141G 4.47 pages/s (4938.31 tok/s). Unoptimized baseline: 0.95 pages/s. In the same table: dots.ocr 0.28 pages/s, MonkeyOCR-pro-3B 0.47, MonkeyOCR-pro-1.2B 0.53, Nanonets-OCR-s 0.55, and MinerU2-VLM (SGLang) 2.84. Hardware for the rivals is not stated per row — [MinerU2.5 arXiv 2509.22186](https://arxiv.org/html/2509.22186)
  - **Conflict:** MinerU2.5 reports 2.12 pages/s on A100, while PaddleOCR-VL's report measures MinerU2.5 at 1.0574 pages/s on A100. The likely causes are a different OmniDocBench version, page count and deployment tuning.
- **olmOCR (7B, Qwen2-VL based) paper, Table 6:** L40S 906 output tok/s (1,288 pages in 17m10s, about 1.25 pages/s), $176 per million pages. H100 3,050 tok/s (5m7s, about 4.2 pages/s), $178 per million pages. Comparisons: MinerU (pipeline) L40S 238 tok/s, $596/M; Marker H100 $1,484/M; GPT-4o batch $6,240/M. These runs used SGLang, and the pipeline later moved to vLLM — [olmOCR arXiv 2502.18443v3](https://arxiv.org/html/2502.18443v3)
- olmOCR README claims "less than $200 USD per million pages" — [olmOCR GitHub](https://github.com/allenai/olmocr). A secondary page claims olmOCR-2 FP8 sustains about 3,400 output tok/s on one H100 — [beri.net](https://www.beri.net/learning/olmocr-2-docs) (unverified, secondary)
- **DeepSeek-OCR:** the widely repeated "200K+ pages/day on one A100-40G" (about 2.3 pages/s) traces to the repo's vLLM PDF concurrency figure of about 2,500 tok/s. I found no reproducible vLLM benchmark — [Notebookcheck](https://www.notebookcheck.net/DeepSeek-OCR-AI-model-can-process-200-000-document-pages-a-day-on-a-single-Nvidia-A100-GPU.1144400.0.html); [Spheron](https://www.spheron.network/blog/deploy-deepseek-ocr-gpu-cloud/). Spheron also claims about 35 pages/min on an L40S (vendor claim, no method).
- Other cross-paper citations of the PaddleOCR numbers: Qianfan-OCR cites PaddleOCR-VL 1.224, MinerU2.5 1.057 and dots.ocr 0.352 pages/s on A100 vLLM 0.10.2 — [Qianfan-OCR arXiv 2603.13398](https://arxiv.org/pdf/2603.13398). A coarse-to-fine paper reports PaddleOCR-VL-L at 1.6192 pages/s — [arXiv 2603.24326](https://arxiv.org/pdf/2603.24326)

### Inferences
- For planning, a 300-page document on one A100-class GPU takes about 2.5–5 minutes with a ≤1.2B OCR VLM tuned on vLLM, about 15 minutes with a 3B model like dots.ocr, and about 1–1.5 minutes on H100/H200 with a fast small model. These are page-throughput extrapolations; latency for a single document also depends on the slowest (longest-output) pages.
- Output tokens per page (about 1,100–1,500) dominate. Throughput in tok/s divided by your corpus's average output tokens per page gives a better estimate than a headline pages/s figure.

### Gaps
- No vetted pages/s figures found for Nanonets-OCR2, Qwen3-VL-based OCR, or DeepSeek-OCR on vLLM under a controlled benchmark.
- MinerU2.5 does not publish its vLLM version, and Appendix C of PaddleOCR-VL (other hardware configurations) was not retrieved.

## Q4. How do image resolution and DPI affect token count, speed and accuracy?

### Takeaway
Visual tokens scale with pixel area. For Qwen2/2.5-VL-family encoders, one token covers a 28×28 pixel patch, so the default max of 1280×28×28 pixels is about 1,280 tokens. Above roughly 144–200 DPI, extra pixels mostly add tokens and cost with little accuracy gain, unless the page has small print or dense tables. Most OCR VLMs cap internally anyway (about 1024–1288 px longest side, or 2048×28×28 pixels). Render at a DPI that hits the model's native cap rather than rendering at 300 DPI and downscaling.

### Cited Findings
- Qwen2-VL-style models: `mm_processor_kwargs={"max_pixels": 768*768}`; "Default is 1280 * 28 * 28." Lowering it reduces memory — [vLLM Conserving Memory](https://docs.vllm.ai/en/latest/configuration/conserving_memory.html)
- **DeepSeek-OCR modes:** Tiny 512² (64 tokens), Small 640² (100), Base 1024² (256), Large 1280² (400). Gundam = n×640² tiles + 1024² global = n×100+256 tokens (n typically 2–9). Tiny and Small resize; Base and Large pad, which wastes tokens on padding — [DeepSeek-OCR arXiv 2510.18234](https://arxiv.org/pdf/2510.18234); [Voxel51 docs](https://docs.voxel51.com/plugins/plugins_ecosystem/deepseek_ocr.html)
- DeepSeek-OCR precision is about 97% at <10x text/vision-token compression and falls to about 60% at 20x — [DeepSeek-OCR paper](https://arxiv.org/pdf/2510.18234) (as summarized by [Towards AI](https://pub.towardsai.net/deepseek-ocr-the-quiet-revolution-thats-making-documents-10-cheaper-to-process-a9c00e2d171b))
- MinerU2.5 uses a two-stage design: layout on a fixed 1036×1036 thumbnail, then crops at native resolution capped at 2048×28×28 pixels. This avoids running the whole page at high resolution — [MinerU2.5](https://arxiv.org/html/2509.22186)
- PaddleOCR-VL (NaViT dynamic resolution) trained with a max of 1280×28×28 (stage 1) and 2048×28×28 (stage 2) — [PaddleOCR-VL report](https://arxiv.org/html/2510.14528)
- olmOCR fine-tuned with 1024 px on the longest edge, and its pipeline exposes `--target_longest_image_dim` for render size — [olmOCR paper](https://arxiv.org/html/2502.18443v3); [olmOCR GitHub](https://github.com/allenai/olmocr)
- A fixed-DPI sweep (Qwen3.5-4B and Sonnet 4.5) on long-document QA found accuracy peaked at 144 DPI. 180 and 216 DPI lowered Sonnet's accuracy by about 1.8–2.0 points while raising tokens — [arXiv 2609.29933](https://arxiv.org/pdf/2609.29933) (QA, not pure OCR)
- Rendered-text fidelity: 192 DPI gave near-perfect recognition but cost more tokens than the raw text, and 96 DPI was the best compression trade-off — [arXiv 2603.00155](https://arxiv.org/pdf/2603.00155) (synthetic typeset setup)
- Traditional OCR guidance of about 300 DPI comes from vendor blogs and does not carry over to VLMs — [imagetotable.ai](https://imagetotable.ai/es/blog/ocr-speed-vs-accuracy-tradeoff)

### Inferences
- A US-Letter page at 144 DPI is about 1224×1584 px, about 1.94 MP, roughly 2,470 Qwen-style tokens before the max_pixels cap. At 200 DPI it is about 1700×2200 (3.7 MP, about 4,770 tokens). At 300 DPI it is about 2550×3300 (8.4 MP, about 10,700 tokens). With the default 1280×28×28 cap (about 1.0 MP), anything above roughly 100 DPI gets downscaled anyway, so higher DPI only adds CPU rasterization and transfer cost. (These are my arithmetic from the 28×28 patch rule, not from a source.)
- Recommendation: render with a target longest side matching the model (for example 1288–1600 px for Qwen-based OCR, or DeepSeek-OCR Base/Gundam). Raise max_pixels to about 2048×28×28 only for dense or small-font pages, and accept roughly 1.6x more prefill tokens.

### Gaps
- No dedicated study found comparing 144 vs 200 vs 300 DPI on an OCR benchmark (OmniDocBench or olmOCR-Bench) with a fixed model on vLLM.

## Q5. Failure modes at scale and mitigations

### Takeaway
The main operational risks are: (1) runaway or repetition generation that holds a batch slot up to `max_tokens`, wrecking tail latency; (2) KV pressure causing preemption/recompute, or OOM when encoder replicas or CUDA graphs are mis-sized; (3) CPU preprocessing starvation; (4) client timeouts on long generations. Mitigations: per-request n-gram no-repeat logits processors (official for DeepSeek-OCR), layout-aware repetition and frequency penalties, hard `max_tokens` caps, greedy decoding, retries with a per-document page-failure budget, and fallback re-rendering.

### Cited Findings
- **DeepSeek-OCR official anti-loop config:** `NGramPerReqLogitsProcessor` with `ngram_size=30`, `window_size=90`, `whitelist_token_ids={128821,128822}` (`<td>`, `</td>` so tables can legitimately repeat). Settings: `temperature=0.0`, `skip_special_tokens=False`, `max_tokens=8192` offline and 2048 in the online example. Plain prompt `"<image>\nFree OCR."` — [vLLM Recipes: DeepSeek-OCR](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-OCR.html)
- vLLM has no native `no_repeat_ngram_size` sampling parameter. A custom logits processor is the suggested route, along with temperature 0 / top_p 1.0 for deterministic OCR — [vLLM forum](https://discuss.vllm.ai/t/speeding-up-vllm-inference-for-qwen2-5-vl/615)
- A higher `repetition_penalty` on Qwen2.5-VL breaks naturally repetitive content such as tables — [QwenLM/Qwen3-VL issue #1611](https://github.com/QwenLM/Qwen3-VL/issues/1611)
- MinerU2.5 adjusts `frequency_penalty` and `presence_penalty` per layout type: higher for text paragraphs, lower for tables — [MinerU2.5](https://arxiv.org/html/2509.22186)
- Tensorlake (Feb 2026): small-backbone OCR models often loop on sparse tables and forms. Early loop detection that stops generation, plus re-running on masked regions, cut worst-case latency from 60 s+ to about 10–20 s — [Tensorlake blog](https://tensorlake.ai/blog/vlm-pipeline-improvement) (search summary; page returned 404 on fetch)
- **olmOCR retries:** a 12% retry rate was measured. `--max_page_retries` and `--max_page_error_rate` (default 1/250 pages per document may fail) bound effort. Over-long prompts are regenerated with smaller character limits. olmOCR-Bench flags outputs ending in repeated n-grams longer than 30 chars. The FP8 model needs fewer retries — [olmOCR paper](https://arxiv.org/html/2502.18443v3); [olmOCR GitHub](https://github.com/allenai/olmocr)
- **Preemption:** the default is RECOMPUTE. Reduce it via `gpu_memory_utilization`, lower `max_num_seqs`/`max_num_batched_tokens`, or more parallelism, and monitor the preemption counters — [vLLM Optimization](https://docs.vllm.ai/en/latest/configuration/optimization.html)
- **OOM risks:** `mm_encoder_tp_mode="data"` replicates encoder weights per rank and can OOM on tight fits. CUDA graphs are captured up to `max_num_seqs` by default (reduce with `cudagraph_capture_sizes` or `enforce_eager`). Use `limit_mm_per_prompt` and size hints so profiling reserves the right encoder memory — [vLLM Optimization](https://docs.vllm.ai/en/latest/configuration/optimization.html); [vLLM Conserving Memory](https://docs.vllm.ai/en/latest/configuration/conserving_memory.html)
- **CPU exhaustion:** each API server uses 8 media-loading threads by default; tune `VLLM_MEDIA_LOADING_THREAD_COUNT` — [vLLM Optimization](https://docs.vllm.ai/en/latest/configuration/optimization.html)
- **Timeouts:** the DeepSeek-OCR online example sets a 3600 s client timeout — [vLLM Recipes: DeepSeek-OCR](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-OCR.html)

### Inferences
- Set `max_tokens` close to the realistic maximum page output (for example 4096; about 2–3x the 1,100–1,500 average tokens per page) rather than 8192+. Each looping page then wastes bounded decode time and KV. Detect `finish_reason == "length"` and retry that page with a no-repeat logits processor or penalty, a different resolution, or page tiling.
- For a 300-page job, use per-page retries with bounded attempts and a document-level failure budget (the olmOCR pattern), so one bad page doesn't fail the document or stall the batch.
- Use per-request timeouts sized to `max_tokens` divided by the expected per-sequence decode rate under load, not a global short timeout. Under heavy batching, per-sequence speed can drop to about 20–50 tok/s.
- Run long jobs with stats logging on and watch the KV cache usage %, waiting queue and preemption counters. Sustained preemption means `max_num_seqs` is too high for the visual token budget.

### Gaps
- I couldn't retrieve a specific vLLM GitHub issue documenting OOM under high multimodal concurrency (search returned only docs).
- No quantitative data found on how often repetition loops occur per 1,000 pages for specific models, apart from olmOCR's 12% overall retry rate.
