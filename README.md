# Confidential Kimi K3

Kimi K3 (2.8T-parameter MoE, MXFP4, multimodal) served with vLLM on a single
8-GPU Blackwell Ultra node, following the
[upstream vLLM recipe](https://recipes.vllm.ai/moonshotai/Kimi-K3).

Deviations from the recipe for confidential computing:

- `--disable-custom-all-reduce`, `VLLM_ALLREDUCE_USE_SYMM_MEM=0`, and
  `fuse_allreduce_rms: false` — peer-mapped GPU memory is unavailable, so all
  cross-GPU communication goes through NCCL.
- `patches/0001` replaces vLLM's UVA zero-copy host buffers with device
  mirrors: GPU reads of host-mapped memory return corrupt values in this
  environment, which surfaced as NaN logits and garbage output.
- Single-node only: no `--all2all-backend` (the RDMA and one-sided NVLink
  backends are for multi-node deployments).
- Model weights load from a dm-verity-protected model pack
  (`--load-format runai_streamer`), not from the HF hub.
- `--max-model-len 262144` rather than the full 1M context.
- `--enable-prefix-caching --mamba-cache-mode align` — vLLM defaults prefix
  caching off for hybrid models (K3 has KDA linear-attention layers), so it
  must be opted into explicitly; K3 supports only the block-aligned KDA
  state-checkpoint mode.
- Expert parallelism must stay enabled under CC: with EP off, vLLM ≥ 0.27.1
  auto-enables the K3 latent-MoE tail-fusion kernels on SM100, which require
  NVLS multicast and fail engine init in CC mode (multicast is unavailable).
- Do not add CPU weight offloading (`--cpu-offload-gb`): its UVA path is not
  covered by `patches/0001` (`VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1` is the
  escape hatch if ever needed).

Releases are built and measured by the Tinfoil release workflows; the image
digest in `tinfoil-config.yml` is pinned at release time.
