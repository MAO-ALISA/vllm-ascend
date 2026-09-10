# DeepSeek V4 local slot APC (experimental)

This opt-in path reuses local compressed KV cache prefixes at boundaries of
128 **original tokens**. With physical block size 128, C4 pages cover 512
original tokens and C128 pages cover 16384 original tokens. A slot is a hash
alias inside one of those pages; it does not change the physical block size.

## Supported configuration

Use both `feat/dsv4-local-slot-apc-v1` working-tree changes together:

- vLLM: based on the `v0.25.1` tag, with paired per-step scheduler lifecycle hooks.
- vLLM Ascend: based on `releases/v0.25.1rc` (`a675940fa`).
- DeepSeek V4, physical block size 128, prefix caching enabled.
- DCP=1, PCP=1, PP=1; TP and EP remain available.
- V1 model runner, synchronous or asynchronous scheduling.
- Optional speculative method `mtp` or `dspark`, also with async scheduling.
  Draft KV must use uncompressed sliding-window attention. All MTP/DSpark
  layers participate in cache validation, including multi-layer DSpark models.
  DSpark on the V1 runner requires greedy draft sampling.
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
  --async-scheduling \
  --enforce-eager
```

Add the model's usual quantization, TP, EP and tokenizer arguments. Omit
`--kv-transfer-config`. For a source installation
whose generated version differs from the release tag, the existing
`VLLM_VERSION=0.25.1` Ascend override is appropriate only when the underlying
vLLM code actually has this version's API and the paired lifecycle changes.
Use `--no-async-scheduling` for the synchronous baseline. Both repositories
must include `on_step_scheduled`, `on_request_completed`, `on_step_processed`
and the `kv_cache_step_id` field. Merely removing the speculative/async
configuration checks from an older revision is not sufficient.

For a model with compatible MTP weights, add:

```bash
--speculative-config '{"method":"mtp","num_speculative_tokens":3,"enforce_eager":true}'
```

For a model with compatible DSpark weights, use instead:

```bash
--speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"greedy","enforce_eager":true}'
```

These are alternative modes, not two drafters enabled on the same engine. The
weights must match the selected method; changing the flag does not convert an
MTP checkpoint into a DSpark checkpoint. Keep `--enforce-eager` for initial
device acceptance of both the target and draft paths.

The `block_size=128` requirement applies before KV cache initialization. After
initialization, EngineCore records the smallest KV group block size in
`cache_config.block_size` (for example, 8 for compressor state). Handshake
revalidation preserves this runtime value and still checks the other slot-APC
constraints. The per-group cache managers check the actual physical page sizes.

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

Cache publication is deferred until the scheduler processes the corresponding
completed model output. The scheduler seals each publication/copy batch with
an opaque step ID before advancing optimistic request counters. Each snapshot
records that step's token end position and block table, pinning unpublished
pages even if a later schedule removes SWA/state blocks from the live table.
Published full pages do not need extra pins; ordinary decode steps that do not
advance a slot boundary skip block-table snapshots.

EngineCore consumes outputs in dispatch order. Without speculation, completion
publishes only that step's snapshot, capped by finalized input hashes, then
releases that step's snapshot/COW references. Async output callbacks do not publish against
the live, ahead-of-execution request counters. A sampled output token is not
cached until a later forward has actually computed its KV. Empty steps carry
no fence; out-of-order or duplicate nonempty fences fail closed.

Free/preemption invalidates the request's publication lifetime without dropping
dispatched references early. A resumed request receives a new lifetime even
when it reuses the same Request object or ID. Old completions cannot publish
into the new block table. Cache reset can return false while cancelled steps
still hold references; drain those outputs and retry. No device-wide sync or
separate COW stream is introduced: copies remain ordered with forward on the
compute stream.

### MTP / DSpark acceptance and rollback

With speculation, device completion starts processing a step but does not yet
publish it. After Scheduler applies its existing rejection count and appends
accepted token IDs, `on_request_completed` publishes that request's valid
snapshot before stop/free handling. `on_step_processed` releases the step's
references after the output loop, including references for aborted requests
that no longer receive a per-request callback.

Each lifetime records cumulative rejected tokens. Each snapshot records the
unrounded end position and the rejection count known when dispatched:

```text
executed_end = scheduled_end - (rejected_now - rejected_at_dispatch)
published_end = floor(min(executed_end, hashed_tokens) / 128) * 128
```

This corrects later async steps dispatched before an earlier rejection was
known, without subtracting the same rejection twice from a newly scheduled
step. For example, end 130 with one rejection still publishes through 128;
end 130 with three rejections does not. Even decode steps that cannot publish
a new slot keep a small record so their rejections correct subsequent steps.
Bonus tokens whose IDs are known but whose target KV has not executed remain
unpublished.

SWA/compressor-state recycling subtracts unresolved output placeholders from
the optimistic computed position. This keeps the state pages needed if the
next async query rewinds after rejection. The existing worker position,
slot-mapping and acceptance paths perform recomputation; this change does not
add a new acceptance algorithm or copy rejected state into the shared cache.
Compressed partial tails still use the existing private-page COW path.

Draft groups retain the upstream EAGLE/MTP next-block check. For a draft block
size of 128, a hit at H must also find a valid draft block through H+128; the
extra block is checked and dropped, not returned as part of the hit. Target
groups do not apply this drop. Equal target/draft specs are kept in separate
lookup groups, and every DSpark draft layer is checked.

**Hit granularity remains 128, but hit length can be shorter than without
speculation.** With a 256-token common prefix and different suffixes, the
conservative speculative hit is 128. A common prefix of only 205 tokens may
yield no joint hit, even though the target-only path can reuse 128. This is a
draft-state validity constraint, not a change to the slot size. Recovering
the last target-only slot would require a separate draft reconstruction path.

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
pytest -q tests/ut/core/test_slot_apc.py \
  tests/ut/core/test_slot_apc_config.py \
  tests/ut/worker/test_slot_kv_cache_copy.py
```

The tests cover exact C4/C128 hits, hash promotion and eviction, independent
tail representatives, capacity failures, COW reference retention and cleanup,
deferred publication, hybrid state constraints, chunked prefill recycling,
disabled-mode compatibility, and bounded raw-page copies. Async tests also
cover overlapping chunked-prefill snapshots, delayed decode token hashes,
per-step COW reference release, cancellation and resumed request IDs, empty
batches, duplicate/out-of-order fences, and reset after draining.
Configuration tests also cover KV initialization followed by handshake
revalidation, repeated validation, and preservation of the runtime block size.
Speculative tests cover one MTP and three DSpark draft groups, missing peek
blocks, accepted/rejected token boundaries at 128/512/16384, unexecuted bonus
tokens, cumulative rejection across overlapping steps, SWA rewind retention,
stop/free ordering, and cancellation followed by request reuse.

To measure CPU lookup and initial alias-publication costs independently of
NPU execution:

```bash
python benchmarks/benchmark_slot_apc.py --length 131072 --repeats 1000
```

This microbenchmark is not a model-throughput or COW-device-copy benchmark.

To compare CPU allocation/snapshot/publication overhead with one or more
in-flight steps (also checks that all retained references are released):

```bash
VLLM_ASCEND_ENABLE_SLOT_APC=1 python benchmarks/benchmark_slot_apc_async.py \
  --prefix-length 131072 --steps 1024 --inflight-depth 2
```

This synthetic benchmark uses known token IDs and no NPU execution. It does
not measure end-to-end async speedup or replace the device acceptance test.

On four Ascend devices with the test model available:

```bash
pytest -q tests/e2e/pull_request/four_card/test_slot_apc.py
```

The device test runs both sync and async scheduling and compares cold prefill
against four warmed requests with divergent suffixes and different output
lengths. It checks exact cached-token counts at 128, 256, 384, 512, 640,
16256, 16384 and 16512 tokens, greedy output equality, and generated-token
log probabilities. This test must pass on the deployment hardware before
enabling the feature there. Additional speculative cases run MTP and DSpark
with both scheduling modes at 128/384/512/16384-token hit boundaries. They use
an extra 128-token common block for draft validation and output lengths up to
257 to exercise new slot publication during decode. The tests also require
nonzero draft activity, so a silently disabled proposer cannot pass.
These device tests have not been executed in the Windows workspace; CPU
lifecycle tests do not validate NPU kernels, graph replay, or model accuracy.

## Next stages

- Stage 3 device acceptance: run the MTP/DSpark cold/warm tests above before
  deployment. The local implementation and CPU lifecycle coverage are in place.
- Stage 4: MooncakeHybridConnector only, including transferred slot metadata
  and coordination with local COW/publication. Other connectors are out of scope.
- Physical block sizes 32/64: keep the APC match granularity at 128 original
  tokens, but adapt per-group page coverage, COW boundaries and worker layouts.
  These are independent parameters, not a change to `SLOT_SIZE`. This change
  still accepts physical block size 128 only; smaller pages need separate tests.
