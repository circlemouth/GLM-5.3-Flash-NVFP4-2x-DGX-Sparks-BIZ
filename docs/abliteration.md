# Optional BF16 o_proj overlay

This unofficial BIZ fork can replace 29 main-model `self_attn.o_proj.weight` tensors (layers 15–43) and the native MTP tensor (layer 45) while loading NVIDIA's pinned NVFP4 checkpoint. Layer 44, layers 0–14, other attention weights, experts, vision, tokenizer, template and the API contract remain unchanged. The option is disabled unless `runtime.weight_overlay.enabled = true` is set. The fork does not distribute weights or a container image.

The donor is the immutable Dealign revision in the [asset manifest](../config/abliteration.lock.json). [Provenance and terms](abliteration-licensing.md) and [validation status](abliteration-validation.md) have separate owners. The original BIZ acceptance does not qualify this optional profile.

## Prepare the assets

Use the pinned revisions in `config/runtime.lock.json` and `config/abliteration.lock.json`. The extraction command reads the actual model indexes and Safetensors headers, downloads only the 30 named BF16 byte ranges, verifies HTTP partial responses and writes one Safetensors file per tensor under a lock. Its `manifest.json` and files stay outside Git.

```sh
python tools/prepare_abliteration.py --help
python tools/prepare_abliteration.py inspect --output /private/overlay-assets
python tools/prepare_abliteration.py extract --output /private/overlay-assets
sha256sum /private/overlay-assets/manifest.json
```

Run the same extraction or a verified, resumable transfer on both model hosts. Check `manifest.json` and all 30 file hashes on each host before launch. Do not copy the donor's `config.json`, tokenizer or chat template into the NVIDIA snapshot. `tools/prepare_mtp_view.py` still creates the existing read-only metadata view when MTP is enabled.

## Select a profile

The standard `examples/server.example.toml` has no overlay table and never opens donor files. For an overlay profile, keep the original settings and add this table beneath `[runtime]`, replacing the placeholders with the verified local path and exact digest. Give the overlay a distinct `api.served_model_name` and an image ID built from this fork's reviewed commit on both hosts.

```toml
[runtime.weight_overlay]
enabled = true
path = "/private/overlay-assets"
donor_revision = "745aac2ff0f10acf961f396df3f9418598aa7327"
manifest_sha256 = "<64 lowercase hexadecimal characters>"
```

The launcher mounts the directory read-only at `/weight-overlay`. The profile fingerprint includes the overlay setting, donor revision, manifest digest and pinned NVIDIA revision; the launch identity also records the inspected image ID. Preflight verifies the assets and rejects the derived AXL checkpoint. Switch the two ranks together through the existing [recoverable switch](launch-safety.md#all-rail-checks-and-two-rank-switch). A switch restarts both ranks, so KV and prefix-cache state is not reused across profiles. Native MTP loads layer 45 only when MTP is enabled; a disabled MTP profile applies 29 tensors, not 30.

The source-pinned image patch refuses an unknown vLLM model or MTP loader hash. At load, an absent, duplicated, changed, non-BF16 or wrong-shaped target fails the launch. It never edits the source checkpoint. Startup success alone does not prove the resulting rank-local parameters; use the [validation protocol](abliteration-validation.md) before accepting a deployment.
