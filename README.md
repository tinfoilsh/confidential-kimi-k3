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
- `--attention-backend TRITON_MLA` and default (bf16) KV cache: the FlashInfer
  MLA path and the fp8-KV prefill-quant path are affected by the same
  host-memory issue.
- `--enforce-eager`: CUDA graph capture aborts on this hardware; revisit once
  capture is stable.
- Single-node only: no `--all2all-backend` (the RDMA and one-sided NVLink
  backends are for multi-node deployments).
- Model weights load from a dm-verity-protected model pack
  (`--load-format runai_streamer`), not from the HF hub.
- `--max-model-len 262144` rather than the full 1M context.

Releases are built and measured by the Tinfoil release workflows; the image
digest in `tinfoil-config.yml` is pinned at release time.
