# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for the synchronous DeepSeek V4 local slot-cache path."""

from vllm import envs as vllm_envs
from vllm.config import VllmConfig
from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator

from vllm_ascend.utils import vllm_version_is

SLOT_SIZE = 128  # Original tokens, independent of the compression ratio.


def validate_slot_apc_config(config: VllmConfig) -> None:
    """Reject configurations whose copy/publication ordering is not supported."""
    reasons = []
    if not vllm_version_is("0.25.1") or not hasattr(KVCacheCoordinator, "take_block_copies"):
        reasons.append("vLLM 0.25.1 with the local slot-APC lifecycle hooks")
    if config.model_config.hf_text_config.model_type != "deepseek_v4":
        reasons.append("a DeepSeek V4 model")
    if not config.cache_config.enable_prefix_caching:
        reasons.append("enable_prefix_caching=True")
    if config.cache_config.block_size != SLOT_SIZE:
        reasons.append("block_size=128")
    parallel = config.parallel_config
    if parallel.decode_context_parallel_size != 1 or parallel.prefill_context_parallel_size != 1:
        reasons.append("DCP=PCP=1")
    if parallel.pipeline_parallel_size != 1:
        reasons.append("PP=1 (one in-flight execution step)")
    if config.kv_transfer_config is not None:
        reasons.append("no KV connector")
    if config.scheduler_config.async_scheduling:
        reasons.append("async_scheduling=False")
    if config.speculative_config is not None:
        reasons.append("no speculative decoding/MTP")
    if vllm_envs.VLLM_USE_V2_MODEL_RUNNER:
        reasons.append("the V1 model runner")
    if reasons:
        raise ValueError("VLLM_ASCEND_ENABLE_SLOT_APC requires " + "; ".join(reasons))
