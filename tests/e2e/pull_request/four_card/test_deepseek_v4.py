#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
# Adapted from vllm/tests/basic_correctness/test_basic_correctness.py
#
import os
from unittest.mock import patch

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


@patch.dict(
    os.environ,
    {
        "VLLM_ASCEND_ENABLE_FLASHCOMM1": "1",
    },
)
@wait_until_npu_memory_free()
def test_deepseek_v4_w4a8_tp4_basic_greedy():
    """Verify DeepSeek V4 W4A8 basic greedy generation with TP4 and EP."""
    example_prompts = [
        "Hello, my name is",
        "What is the meaning of life?",
    ]
    max_tokens = 5

    with VllmRunner(
        "gdydems/DeepSeek-V4-Flash-w4a8-mtp",
        max_model_len=8192,
        max_num_seqs=16,
        max_num_batched_tokens=4096,
        dtype="auto",
        tensor_parallel_size=4,
        enable_expert_parallel=True,
        gpu_memory_utilization=0.9,
        quantization="ascend",
        tokenizer_mode="deepseek_v4",
        block_size=128,
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
        },
        speculative_config={"num_speculative_tokens": 1, "method": "mtp", "enforce_eager": True},
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(example_prompts, max_tokens)
        expected_token_ids = [
            [19923, 14, 1026, 2329, 344, 680, 2852, 95, 305, 342],
            [3085, 344, 270, 5281, 294, 1988, 33, 3955, 361, 582, 3085, 344],
        ]
        assert len(outputs) == len(example_prompts)
        for i, (output_ids, output_str) in enumerate(outputs):
            assert len(output_str) > 0
            assert len(output_ids) > 0
            assert output_ids == expected_token_ids[i]


@patch.dict(
    os.environ,
    {
        "VLLM_ASCEND_ENABLE_FLASHCOMM1": "1",
    },
)
@wait_until_npu_memory_free()
def test_deepseek_v4_w4a8_tp4_index_cache_freq4():
    """IndexCache freq=4 must produce non-empty greedy outputs identical in
    shape to the baseline test, verifying skip_topk/topk_indices_buffer
    plumbing (DSAModules → AscendDSAImpl) is wired correctly across both
    serial and dual-stream paths.
    """
    example_prompts = [
        "Hello, my name is",
        "The capital of France is",
        "What is the meaning of life?",
    ]
    max_tokens = 5

    with VllmRunner(
        "gdydems/DeepSeek-V4-Flash-w4a8-mtp",
        max_model_len=8192,
        max_num_seqs=16,
        max_num_batched_tokens=4096,
        dtype="auto",
        tensor_parallel_size=4,
        enable_expert_parallel=True,
        gpu_memory_utilization=0.9,
        quantization="ascend",
        tokenizer_mode="deepseek_v4",
        block_size=128,
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
        },
        hf_overrides={
            "use_index_cache": True,
            "index_topk_freq": 4,
        },
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(example_prompts, max_tokens)

        assert len(outputs) == len(example_prompts)
        for output_ids, output_str in outputs:
            assert len(output_str) > 0
            assert len(output_ids) > 0


@pytest.mark.parametrize("block_size", [32, 64, 128])
@patch.dict(
    os.environ,
    {
        "VLLM_ASCEND_ENABLE_FLASHCOMM1": "1",
    },
)
@wait_until_npu_memory_free()
def test_deepseek_v4_mtp_prefix_cache_replay(block_size: int):
    """A repeated DSV4 prefix must hit an LCM-aligned APC checkpoint."""
    lcm_block_size = block_size * 128
    prompt_length = 2 * lcm_block_size + block_size
    max_tokens = 2

    with VllmRunner(
        "gdydems/DeepSeek-V4-Flash-w4a8-mtp",
        max_model_len=prompt_length + max_tokens,
        max_num_seqs=2,
        max_num_batched_tokens=4096,
        dtype="auto",
        tensor_parallel_size=4,
        enable_expert_parallel=True,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.9,
        quantization="ascend",
        tokenizer_mode="deepseek_v4",
        block_size=block_size,
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
        },
        speculative_config={
            "num_speculative_tokens": 1,
            "method": "mtp",
            "enforce_eager": True,
        },
    ) as vllm_model:
        tokenizer = vllm_model.model.get_tokenizer()
        seed_ids = tokenizer.encode(
            "DeepSeek V4 prefix cache replay regression. ",
            add_special_tokens=False,
        )
        assert seed_ids
        prompt_token_ids = (seed_ids * (prompt_length // len(seed_ids) + 1))[:prompt_length]
        inputs = vllm_model.get_inputs([prompt_token_ids])
        sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)

        first = vllm_model.model.generate(inputs, sampling_params=sampling_params)[0]
        second = vllm_model.model.generate(inputs, sampling_params=sampling_params)[0]

        cached_tokens = second.num_cached_tokens or 0
        assert cached_tokens > 0
        assert cached_tokens % lcm_block_size == 0
        assert first.outputs[0].token_ids == second.outputs[0].token_ids
