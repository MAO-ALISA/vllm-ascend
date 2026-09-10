# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for the DeepSeek V4 local slot-cache path."""

from vllm import envs as vllm_envs
from vllm.config import VllmConfig
from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator

from vllm_ascend.utils import vllm_version_is

SLOT_SIZE = 128  # Original tokens, independent of the compression ratio.


def validate_slot_apc_config(config: VllmConfig) -> None:
    """Reject configurations whose copy/publication ordering is not supported."""
    reasons = []
    if not vllm_version_is("0.25.1") or not all(
        hasattr(KVCacheCoordinator, name) for name in ("on_step_scheduled", "on_request_completed", "on_step_processed")
    ):
        reasons.append("vLLM 0.25.1 with the local slot-APC lifecycle hooks")
    if config.model_config.hf_text_config.model_type != "deepseek_v4":
        reasons.append("a DeepSeek V4 model")
    if not config.cache_config.enable_prefix_caching:
        reasons.append("enable_prefix_caching=True")
    # After KV initialization this field is the minimum across all KV groups,
    # including compressor state, not the selected compressed-page size.
    # The per-group managers validate the actual KVCacheSpec block sizes.
    if config.cache_config.num_gpu_blocks is None and config.cache_config.block_size != SLOT_SIZE:
        reasons.append(f"block_size=128 (actual={config.cache_config.block_size!r})")
    parallel = config.parallel_config
    if parallel.decode_context_parallel_size != 1 or parallel.prefill_context_parallel_size != 1:
        reasons.append("DCP=PCP=1")
    if parallel.pipeline_parallel_size != 1:
        reasons.append("PP=1")
    if config.kv_transfer_config is not None:
        reasons.append("no KV connector")
    if (speculative := config.speculative_config) is not None:
        method = getattr(speculative, "method", None)
        if method not in ("mtp", "dspark"):
            reasons.append("speculative method mtp or dspark")
        if method == "dspark" and getattr(speculative, "draft_sample_method", "greedy") != "greedy":
            reasons.append("greedy DSpark draft sampling on the V1 model runner")
    if vllm_envs.VLLM_USE_V2_MODEL_RUNNER:
        reasons.append("the V1 model runner")
    if reasons:
        raise ValueError("VLLM_ASCEND_ENABLE_SLOT_APC requires " + "; ".join(reasons))
