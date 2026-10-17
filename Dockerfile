# syntax=docker/dockerfile:1.6
#
# Confidential Kimi K3 (single-node 8-GPU, MXFP4).
# Base is digest-pinned for attestation. K3 requires CUDA 13 and an r580+
# guest driver (base ships torch cu130).
ARG VLLM_BASE_IMAGE=vllm/vllm-openai:v0.31.0@sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b
ARG SIDECAR_IMAGE=ghcr.io/tinfoilsh/inference-sidecar@sha256:65ce23d6560c46a1e8614ede187fcbf9798b267aa33878905b4872404787f47d
FROM ${SIDECAR_IMAGE} AS sidecar

FROM ${VLLM_BASE_IMAGE}

ENV VLLM_DISABLE_UVA=1 \
    VLLM_TINFOIL_CC_OPTIMIZATIONS=0

# Patches are -p1 unified diffs rooted at /; they target
# usr/local/lib/python3.12/dist-packages/... to match the base image.
# 0001 selects explicit device copies for confidential computing;
# 0008 keeps generated FlashInfer headers in the writable cache.
# 0009 aligns prefill chunks with the recurrent-state cache blocks.
COPY patches/ /tmp/tinfoil-patches/
RUN set -eux; \
    test "$VLLM_BUILD_COMMIT" = "db9527a46873454610df6dbedf79a36d6bf1a7f6"; \
    python3 -c "from importlib.metadata import version; assert version('vllm') == '0.31.0'"; \
    cd /; \
    for p in /tmp/tinfoil-patches/*.patch; do \
        patch -p1 --batch --forward --no-backup-if-mismatch --fuzz=0 < "$p"; \
    done; \
    find /usr/local/lib/python3.12/dist-packages/vllm /usr/local/lib/python3.12/dist-packages/flashinfer -name '__pycache__' -type d -exec rm -rf {} +; \
    rm -rf /tmp/tinfoil-patches

# Reuse the audited SM103a artifact bake and verify both build inputs.
ADD --checksum=sha256:21ef6fe8a859fa204c66a5264c6bbf9045bfc5448326a4f4ef0cb66e29f71bdb https://raw.githubusercontent.com/tinfoilsh/confidential-glm5-3-flash/065a82d7364622962b0bd26b324dfee3c2c90452/scripts/bake-artifacts.py /tmp/tinfoil-artifacts/bake-artifacts.py
ADD --checksum=sha256:f8c32120255bbd57298e824e784b7ff121239235285c02851e792756e615026a https://raw.githubusercontent.com/tinfoilsh/confidential-glm5-3-flash/065a82d7364622962b0bd26b324dfee3c2c90452/scripts/flashinfer-artifacts.json /tmp/tinfoil-artifacts/flashinfer-artifacts.json
RUN FLASHINFER_CUDA_ARCH_LIST=10.3a python3 /tmp/tinfoil-artifacts/bake-artifacts.py && \
    rm -rf /tmp/tinfoil-artifacts

COPY --from=sidecar /inference-sidecar /opt/tinfoil/inference-sidecar
ENTRYPOINT ["/opt/tinfoil/inference-sidecar", "vllm", "serve"]
