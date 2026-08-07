# syntax=docker/dockerfile:1.6
#
# Confidential Kimi K3 (single-node 8-GPU, MXFP4).
# Base is digest-pinned for attestation. Upstream K3 image requires CUDA 13
# and an r580+ guest driver.
ARG VLLM_BASE_IMAGE=vllm/vllm-openai:kimi-k3@sha256:fb16b180bd9727600067e16fcd6a6de43fb4db1baf4298ef20b4dbdf6bfa5a0e
FROM ${VLLM_BASE_IMAGE}

# Bake FlashInfer cubins at build time: the enclave has no egress for JIT
# downloads and the container rootfs is read-only, so the symlinks
# ensure_symlink() would create at runtime are pre-created here.
RUN set -eux; \
    if ! command -v flashinfer >/dev/null 2>&1; then \
        echo "flashinfer CLI not present; skipping cubin bake"; exit 0; \
    fi; \
    flashinfer download-cubin; \
    cubin_dir=$(python3 -c "import flashinfer_cubin, os; print(os.path.join(os.path.dirname(flashinfer_cubin.__file__), 'cubins'))"); \
    du -sh "$cubin_dir"; \
    mkdir -p "$cubin_dir/flashinfer/trtllm/batched_gemm" "$cubin_dir/flashinfer/trtllm/gemm"; \
    for d in "$cubin_dir"/*/; do \
        gemm_dir=$(find "$d" -maxdepth 3 -type d -name "trtllmGen_gemm_export" 2>/dev/null | head -1); \
        bmm_dir=$(find "$d" -maxdepth 3 -type d -name "trtllmGen_bmm_export" 2>/dev/null | head -1); \
        if [ -n "$gemm_dir" ]; then \
            ln -sf "$gemm_dir" "$cubin_dir/flashinfer/trtllm/gemm/trtllmGen_gemm_export"; \
        fi; \
        if [ -n "$bmm_dir" ]; then \
            ln -sf "$bmm_dir" "$cubin_dir/flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export"; \
        fi; \
    done; \
    python3 -c "import flashinfer; print('flashinfer', flashinfer.__version__, 'cubins baked')"
