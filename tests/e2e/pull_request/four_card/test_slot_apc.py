# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device acceptance test for local DeepSeek V4 slot hits and divergent tails."""

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free


@pytest.mark.e2e_model("gdydems/DeepSeek-V4-Flash-w4a8-mtp")
@wait_until_npu_memory_free()
def test_slot_apc_matches_cold_prefill(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_SLOT_APC", "1")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    with VllmRunner(
        "gdydems/DeepSeek-V4-Flash-w4a8-mtp",
        tensor_parallel_size=4,
        enable_expert_parallel=True,
        quantization="ascend",
        tokenizer_mode="deepseek_v4",
        max_model_len=17408,
        max_num_batched_tokens=1024,
        max_num_seqs=4,
        block_size=128,
        enable_prefix_caching=True,
        async_scheduling=False,
        enforce_eager=True,
        gpu_memory_utilization=0.9,
    ) as runner:
        llm = runner.model
        sampling = SamplingParams(temperature=0, max_tokens=8, logprobs=5)
        for boundary in [128, 256, 384, 512, 640, 16256, 16384, 16512]:
            prefix = [10 + i % 97 for i in range(boundary)]
            target = {"prompt_token_ids": prefix + [201, 202, 203]}
            llm.reset_prefix_cache()
            cold = llm.generate([target], sampling, use_tqdm=False)[0]
            llm.reset_prefix_cache()
            llm.generate(
                [{"prompt_token_ids": prefix + [301, 302, 303]}],
                SamplingParams(temperature=0, max_tokens=1),
                use_tqdm=False,
            )
            warm = llm.generate([target], sampling, use_tqdm=False)[0]
            assert warm.num_cached_tokens == boundary
            assert cold.outputs[0].token_ids == warm.outputs[0].token_ids
            for token, cold_probs, warm_probs in zip(
                cold.outputs[0].token_ids, cold.outputs[0].logprobs, warm.outputs[0].logprobs
            ):
                assert warm_probs[token].logprob == pytest.approx(cold_probs[token].logprob, abs=0.05)
