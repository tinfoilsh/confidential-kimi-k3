# Confidential Kimi K3

Kimi K3 (2.8T-parameter MoE, MXFP4, multimodal) served with vLLM on a single
8-GPU Blackwell Ultra node, following the
[upstream vLLM recipe](https://recipes.vllm.ai/moonshotai/Kimi-K3).

The configuration in `tinfoil-config.yml` deviates from the upstream recipe
where required by this serving environment.

Releases are built and measured by the Tinfoil release workflows; the image
digest in `tinfoil-config.yml` is pinned at release time.
