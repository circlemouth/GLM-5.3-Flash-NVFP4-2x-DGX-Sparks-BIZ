# Overlay validation and acceptance

## Current status

The optional overlay has no inherited production acceptance. CPU checks cover key selection, preservation of non-target tuples and metadata, off-by-default behavior, manifest and asset validation, missing/duplicate targets, unsupported shapes and derived-checkpoint rejection. A full two-rank load and A/B/A′ comparison must be recorded independently before calling the profile qualified.

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
