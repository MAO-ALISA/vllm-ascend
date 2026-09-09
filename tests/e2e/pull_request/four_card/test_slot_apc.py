# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device acceptance test for local DeepSeek V4 slot hits and divergent tails."""

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free


@pytest.mark.e2e_model("gdydems/DeepSeek-V4-Flash-w4a8-mtp")
@pytest.mark.parametrize("async_scheduling", [False, True])
@wait_until_npu_memory_free()
def test_slot_apc_matches_cold_prefill(monkeypatch, async_scheduling):
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
        async_scheduling=async_scheduling,
        enforce_eager=True,
        gpu_memory_utilization=0.9,
    ) as runner:
        llm = runner.model
        # Different finish times exercise outstanding decode steps after an
        # earlier request has already freed its scheduler-side block table.
        sampling = [SamplingParams(temperature=0, max_tokens=n, logprobs=5) for n in (1, 8, 17, 33)]
        for boundary in [128, 256, 384, 512, 640, 16256, 16384, 16512]:
            prefix = [10 + i % 97 for i in range(boundary)]
            targets = [{"prompt_token_ids": prefix + [201 + i, 202, 203]} for i in range(4)]
            cold = []
            for target, params in zip(targets, sampling):
                # Build independent cold references: batching these requests
                # could allow later admissions to hit an earlier cold request.
                llm.reset_prefix_cache()
                result = llm.generate([target], params, use_tqdm=False)[0]
                assert result.num_cached_tokens == 0
                cold.append(result)
            llm.reset_prefix_cache()
            llm.generate(
                [{"prompt_token_ids": prefix + [301, 302, 303]}],
                SamplingParams(temperature=0, max_tokens=1),
                use_tqdm=False,
            )
            warm = llm.generate(targets, sampling, use_tqdm=False)
            for cold_req, warm_req in zip(cold, warm):
                assert warm_req.num_cached_tokens == boundary
                assert cold_req.outputs[0].token_ids == warm_req.outputs[0].token_ids
                for token, cold_probs, warm_probs in zip(
                    cold_req.outputs[0].token_ids, cold_req.outputs[0].logprobs, warm_req.outputs[0].logprobs
                ):
                    assert warm_probs[token].logprob == pytest.approx(cold_probs[token].logprob, abs=0.05)
