# Overlay validation and acceptance

## Current status

The BF16 overlay is opt-in and has no inherited BIZ acceptance. The 2026-09-24 observations below describe one TP=2 installation with the pinned NVIDIA checkpoint and donor; they do not qualify another installation or certify clinical use. The host-specific records remain private.

## Two-node observation on 2026-09-24

The loaded runtime used fork commit `9a142366e5b128636eb22965a4192c1a6c1a43f7` and the same local image ID `sha256:5c4f0b51a262ae519aa641634ccf63f775c9da47839735519555013eb56f86ab` on both ranks. NVIDIA revision `423acf37583782c51c142d145aef733d72943d93`, Dealign revision `745aac2ff0f10acf961f396df3f9418598aa7327`, and extracted-asset manifest SHA-256 `f6a9398f59a787ffee857ef89f86f71512362d1e9540fe560ad4fdd15342345a` were fixed. The later verifier-only commit `078c6df` does not change the serving image.

| Check | Observed result | Limit |
| --- | --- | --- |
| A: overlay off | The same fork started at TP=2 and passed health and a short synthetic API smoke. | Rank-local base readback was not completed after an incorrect verifier invocation. |
| B: overlay on | TP=2, native MTP depth 3 and 262,144-token context started. The loaded donor shards matched on both ranks for 29 main tensors plus MTP L45 (60/60); non-target L44 matched NVIDIA base on both ranks (2/2). | The loaded-value check belongs to the evaluation launch, not a separate resident launch. |
| A′: return to overlay off | Not run after the operator selected resident overlay operation. | Restart variability remains unmeasured in this sequence. |
| Synthetic clinic tasks | C1 57/60, C2 57/60, critical 60/60, context 6/6. | The archived ordinary-BIZ campaign had the same pass counts but different harness and workload hashes; these are descriptive comparisons, not a matched Contract 5 A/B. |
| Long input | An actual 249,658-token input produced 109 output tokens within the 262,144-token configuration. | This is one synthetic probe, not a guarantee for every long input. |
| Thinking modes | Authenticated direct API requests at low, high and max each produced a final answer. | One raw HTTP request with all thinking fields omitted returned HTTP 200 and empty final content; the BIZ client separately normalizes omission to low. |

The observed B run and 262K probe recorded no OOM, CUDA crash or swap growth. During evaluation shutdown, the rank-1 vLLM worker reached its SIGTERM-to-SIGKILL fallback; Docker reported `OOMKilled=false`, both GPU compute lists became empty and the Controller finish gates passed. The operator then selected the resident overlay profile. Controller revision 1717 reported open admission, no lease or warnings, all critical doctor checks passing and both ranks running. The ordinary BIZ profile and assets remain available for rollback.

The CPU suite after the verifier-only change ran 488 tests: 442 passed and 46 were skipped. Ruff check and the publication audit passed. Repository-wide Ruff format still reports five unchanged upstream files; both changed verifier files pass format check. The quality, performance and unrun A′ gates are reported separately; no measured speed equivalence or full routine-use qualification is claimed.

## Two-rank protocol

Before loading, record the running profile, ownership, admission state, jobs, container/image IDs, model revision, memory, disk, fabric, local modifications and a tested restoration path in private `records/`. Fix the test criteria and synthetic prompts before observing results. Treat both hosts as one exclusive GPU pair. Keep business traffic drained or on an already accepted alternate path during a controlled switch.

1. Build the image from one reviewed fork commit on both hosts. Confirm the source-pinned main and MTP loader patch hashes and identical image IDs.
2. Verify the NVIDIA snapshot, metadata-only MTP view when enabled, donor revision, manifest digest and every extracted Safetensors file on both hosts. Run `server preflight` for each rank. Do not disable the existing derived-checkpoint or foreign-GPU checks.
3. Run A (same fork, overlay off), B (same image, overlay on), then A′ (overlay off again), restarting both ranks through the existing switch for each arm. Keep TP=2, CPU and GPU settings, context, KV, MTP depth, tokenizer, generation parameters, warmup and cache condition fixed. Keep cold-load time separate from TTFT.
4. Instrument rank-local loaded parameters, accounting for the normal TP split. Verify the 29 main tensors and MTP layer 45 when enabled against donor values; verify layer 44 and other non-targets against the base. A disabled MTP run has no layer-45 application. Compare source files before and after without logging tensor contents.
5. Run synthetic task cases for Japanese answers, JSON and tool schemas, mock approval boundaries, tool results, SSE, stop reasons, multi-turn behavior and response anomalies. Check thinking off separately from high/max. Record successes and failures per case without changing criteria. Compare fixed-length timing separately from workflow completion.
6. Measure API and task success, TTFT, decode tokens/s, workflow p50/p95, token counts, MTP acceptance, both hosts' load/steady memory headroom and GPU/communication errors. Investigate any major metric regression of at least 10% with a limited repeat. Increase synthetic input length only after short cases pass, then 32K, 64K and the starting operational target within the supported limit.
7. Restore the exact starting service configuration and verify API health and a synthetic smoke request. Confirm that no test process remains.

Report overlay correctness, ordinary-profile regression, observed model behavior, task quality, speed and restoration separately. A response difference across restarts alone is not evidence of an overlay effect. This protocol does not certify operational, clinical or legal suitability.
