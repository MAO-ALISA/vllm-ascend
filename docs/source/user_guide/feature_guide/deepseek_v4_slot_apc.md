# DeepSeek V4 local slot APC (experimental)

This opt-in path reuses local compressed KV cache prefixes at boundaries of
128 **original tokens**. With physical block size 128, C4 pages cover 512
original tokens and C128 pages cover 16384 original tokens. A slot is a hash
alias inside one of those pages; it does not change the physical block size.

## Supported configuration

Use both `feat/dsv4-local-slot-apc-v1` working-tree changes together:

- vLLM: based on the `v0.25.1` tag, with the scheduler copy/completion hooks.
- vLLM Ascend: based on `releases/v0.25.1rc` (`a675940fa`).
- DeepSeek V4, physical block size 128, prefix caching enabled.
- DCP=1, PCP=1, PP=1; TP and EP remain available.
- V1 model runner, synchronous scheduling, no speculative decoding/MTP.
- No KV transfer connector or external KV pool.

The flag is disabled by default. Configurations excluded above fail at startup.
The NPU acceptance test exercises eager execution; graph replay, alternate
schedulers and other configurations still need their own device validation.

```bash
export VLLM_ASCEND_ENABLE_SLOT_APC=1
export VLLM_USE_V2_MODEL_RUNNER=0

vllm serve /path/to/deepseek-v4 \
  --block-size 128 \
  --enable-prefix-caching \
  --no-async-scheduling \
  --enforce-eager
```

Add the model's usual quantization, TP, EP and tokenizer arguments. Omit
`--speculative-config` and `--kv-transfer-config`. For a source installation
whose generated version differs from the release tag, the existing
`VLLM_VERSION=0.25.1` Ascend override is appropriate only when the underlying
vLLM code actually has this version's API and the paired lifecycle changes.

## Implementation

`SlotCompressAttentionManager` looks up complete logical pages, then searches
the next page's slot boundaries from longest to shortest. It returns the exact
hit length separately from the page list. For example, 256 cached original
tokens require one C4 page and one C128 page, but both hits remain 256 tokens.

`AscendSlotKVCacheCoordinator` checks all compressed and sliding-window groups
at a common boundary, reducing the candidate until every group can reuse it.
SWA and compressor-state cache publication use the same 128-token alignment.
Slot hashes reuse the existing fine-grained request hash chain, which need
not itself have a block size of 128.

Partial hits reserve a private tail page before admission. All local source
pages are touched before allocation. The source's hit reference and an extra
destination reference protect the copy until the step completes. Cancellation
or preemption before dispatch removes the copy and releases those references.

The formal scheduler output carries `(group_id, src_block_id, dst_block_id)`
copy operations. The NPU worker copies after base state update/zeroing and
before forward. A copy plan built from the bounded raw allocations includes
indexer scale bytes and respects storage offsets, padding and shared views.

Cache publication is deferred until the scheduler processes the completed
model output. This prevents a request from copying pages that another request
in the same scheduled batch has not written yet. This implementation therefore
supports one in-flight step only.

The existing `BlockPool.cache_partial_block()` primitive clears aliases when
its primary hash advances or becomes a full-page hash. The manager rebuilds
all interior aliases on changed pages in descending boundary order. This has
CPU/hash-metadata overhead, particularly for C128's 127 interior boundaries.
The unchanged pool eviction and reset paths remove all aliases together.

## Validation

Run in an environment with both modified packages installed:

```bash
# From vllm/
.venv/bin/python -m pytest -q \
  tests/v1/core/prefix_cache/test_partial_prefix_cache_primitives.py \
  tests/v1/core/test_kv_cache_copy_lifecycle.py

# From vllm-ascend/
pytest -q tests/ut/core/test_slot_apc.py tests/ut/worker/test_slot_kv_cache_copy.py
```

The tests cover exact C4/C128 hits, hash promotion and eviction, independent
tail representatives, capacity failures, COW reference retention and cleanup,
deferred publication, hybrid state constraints, chunked prefill recycling,
disabled-mode compatibility, and bounded raw-page copies.

To measure CPU lookup and initial alias-publication costs independently of
NPU execution:

```bash
python benchmarks/benchmark_slot_apc.py --length 131072 --repeats 1000
```

This microbenchmark is not a model-throughput or COW-device-copy benchmark.

On four Ascend devices with the test model available:

```bash
pytest -q tests/e2e/pull_request/four_card/test_slot_apc.py
```

The device test compares cold prefill against warmed prefixes with divergent
suffixes. It checks exact cached-token counts at 128, 256, 384, 512, 640,
16256, 16384 and 16512 tokens, greedy output equality, and generated-token
log probabilities. This test must pass on the deployment hardware before
enabling the feature there. It has not been executed in the Windows workspace.
