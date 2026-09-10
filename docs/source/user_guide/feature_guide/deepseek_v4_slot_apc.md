# DeepSeek V4 slot APC (experimental)

This opt-in path reuses local compressed KV cache prefixes at boundaries of
128 **original tokens**, with physical block sizes 32, 64 or 128. A partial
slot is a hash alias inside a compressed page; it does not change that page's
physical size. With B=32 a C4 page itself covers exactly one 128-token slot.

## Supported configuration

Use both `feat/dsv4-local-slot-apc-v1` working-tree changes together:

- vLLM: based on the `v0.25.1` tag, with paired per-step scheduler lifecycle hooks.
- vLLM Ascend: based on `releases/v0.25.1rc` (`a675940fa`).
- DeepSeek V4, physical block size 32/64/128, prefix caching enabled.
- DCP=1, PCP=1, PP=1; TP and EP remain available.
- V1 model runner, synchronous or asynchronous scheduling.
- Optional speculative method `mtp` or `dspark`, also with async scheduling.
  Draft KV must use uncompressed sliding-window attention. All MTP/DSpark
  layers participate in cache validation, including multi-layer DSpark models.
  DSpark on the V1 runner requires greedy draft sampling.
- Optional built-in `MooncakeHybridConnector` with dedicated P/D roles.
  Other connectors, MultiConnector, bidirectional transfer and external KV
  pools are not supported. See the P/D section below.

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
`--kv-transfer-config` for local-only execution. For a source installation
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

Choose `--block-size 32`, `64` or `128`; the example uses 128. Before KV
initialization this selects a layout from `DSV4_BLOCK_SIZES`, not a single
common physical size for every KV group:

| Selected B | C4 original tokens/page | C128 original tokens/page | SWA block | C4 state block | C128 state block (non-A5 / A5) |
| --- | --- | --- | --- | --- | --- |
| 32 | 128 | 4096 | 32 | 2 | 8 / 4 |
| 64 | 256 | 8192 | 64 | 4 | 16 / 8 |
| 128 | 512 | 16384 | 128 | 8 | 32 / 16 |

Page padding and indexer scales also follow the selected device layout. State
window sizes do not shrink with B; a window spans more physical pages at
smaller B. Do not change `SLOT_SIZE` or the window size to implement small pages.

After initialization, EngineCore records the smallest KV group block size in
`cache_config.block_size` (typically 2, 4 or 8 for these layouts). Handshake
revalidation preserves this runtime value and still checks the other slot-APC
constraints. The per-group managers validate the actual compressed page size.
Unsupported explicit sizes fail before the ordinary V4 fallback could silently
replace them with 32. The default without `--block-size` remains 128.

Leave `--hash-block-size` unset unless deliberately testing a compatible
override. The default hash granularity is the GCD of group sizes: typically
2/4/8 at B=32/64/128. A fixed hash size of 8 is incompatible with 2- or 4-token
state pages. Hashing can be finer than 128 without making APC hits finer.

## Implementation

`SlotCompressAttentionManager` looks up complete logical pages, then searches
the next page's slot boundaries from longest to shortest. It returns the exact
hit length separately from the page list. For example, at B=128, 256 cached
original tokens require one C4 page and one C128 page, but both hits remain
256 tokens. At B=32 the same hit needs two C4 pages and one C128 page.

`AscendSlotKVCacheCoordinator` checks all compressed and sliding-window groups
at a common boundary, reducing the candidate until every group can reuse it.
SWA and compressor-state cache publication use the same 128-token alignment.
Slot hashes reuse the existing fine-grained request hash chain, which need
not itself have a block size of 128.

Partial hits reserve a private tail page before admission. All local source
pages are touched before allocation. The source's hit reference and an extra
destination reference protect the copy until the step completes. Cancellation
or preemption before dispatch removes the copy and releases those references.
Tail ownership uses `H % (B * compress_ratio) != 0`. At H=128 and B=32,
the C4 page is complete and shared without COW; C128 still needs a private tail.
At B=64/128, both compressed groups need private tails for that hit. At each
group's full-page boundary, aliases promote to a full-page hash normally.

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

Draft groups retain the upstream EAGLE/MTP next-block check. For draft block
size B, a hit at H must also find a valid draft block through H+B; the
extra block is checked and dropped, not returned as part of the hit. Target
groups do not apply this drop. Equal target/draft specs are kept in separate
lookup groups, and every DSpark draft layer is checked.
Snapshot publication still occurs at accepted 128-token boundaries for all
page sizes. Consequently, a B=32/64 peek page may have executed but remain
unpublished until the next slot snapshot. This implementation keeps that
conservative behavior rather than publishing draft-only sub-slot snapshots.
Tests warm one additional 128-token common slot for all physical page sizes.

**Hit granularity remains 128, but hit length can be shorter than without
speculation.** With a 256-token common prefix and different suffixes, the
conservative speculative hit is 128. A common prefix of only 205 tokens may
yield no joint hit, even though the target-only path can reuse 128. This is a
draft-state validity constraint, not a change to the slot size. Recovering
the last target-only slot would require a separate draft reconstruction path.

The existing `BlockPool.cache_partial_block()` primitive clears aliases when
its primary hash advances or becomes a full-page hash. The manager rebuilds
all interior aliases on changed pages in descending boundary order. This has
CPU/hash-metadata overhead, particularly for C128's 127 interior boundaries
at B=128 (63 at B=64 and 31 at B=32).
The unchanged pool eviction and reset paths remove all aliases together.

### MooncakeHybridConnector P/D transfer

Stage 4 adds a versioned transfer manifest, explicit logical-to-physical page
mapping, raw-page registration and a remote-completion publication hook.
**Update both repositories on both P and D.** In addition to the phase 3
lifecycle hooks, core vLLM must include `on_remote_cache_ready`. P and D must
use the same model weights, tokenizer, target/draft layers, cache dtype and
page layout. Model weight identity is a deployment requirement; the protocol
checks layout and prefix token IDs, not weight-file contents.

Use `VLLM_ASCEND_ENABLE_SLOT_APC=1`, V1, physical block size 32/64/128 and prefix
caching on both services. Keep DCP=PCP=PP=1. Initially validate one DP replica
per side with eager execution; then repeat under the intended TP/DP/EP and
async settings. Existing Mooncake TP routing is retained (prefill TP must be
at least decode TP). Different raw page geometries are rejected, not reshaped.
P and D must select the same B: P32/D32, P64/D64 or P128/D128. A shared
128-token APC granularity does not make P32/D64 physical pages interchangeable.

For example, add these arguments to the otherwise identical P/D commands
above, with the TP sizes changed to match the actual deployment:

```bash
# P: add to vllm serve ... --tensor-parallel-size 4
--kv-transfer-config '{"kv_connector":"MooncakeHybridConnector","kv_role":"kv_producer","kv_port":20001,"kv_connector_extra_config":{"prefill":{"tp_size":4,"dp_size":1},"decode":{"tp_size":4,"dp_size":1}}}'

# D: add to vllm serve ... --tensor-parallel-size 4
--kv-transfer-config '{"kv_connector":"MooncakeHybridConnector","kv_role":"kv_consumer","kv_port":21001,"kv_connector_extra_config":{"prefill":{"tp_size":4,"dp_size":1},"decode":{"tp_size":4,"dp_size":1}}}'
```

Keep the existing deployment's network/Mooncake settings, model quantization
and device placement. Add `--enable-prompt-tokens-details` for acceptance-test
cache counters. For MTP or DSpark, configure the same draft layout on both
sides; a P service without the D service's draft groups is rejected. The
standard request/response proxy can forward `kv_transfer_params` unchanged.
It must preserve the original prompt: do not append `last_token_id` to it.

The transfer lifecycle is:

1. P performs its normal local slot lookup and computes the **full N-token
   prompt**, with `max_tokens=1`. Unlike the legacy compressed path, it does
   not truncate the Request after the scheduler's lookup. D imports E=N-1
   tokens and recomputes the final prompt token. P requests that decode longer
   are rejected because they could recycle the prompt's compressor state.
2. P exports the layout fingerprint, E, prefix digest and required logical
   page indices for each KV group. Compressed groups require
   `ceil(floor(E / compress_ratio) / physical_block_size)` pages. SWA/state
   groups export the surviving window needed at E, including all draft groups.
3. D combines its local H-token slot hit with the remote continuation. Shared
   full pages are never RDMA destinations. Partial compressed tails receive
   private pages; the remote full-page payload supplies their local prefix too,
   so no local COW is queued for those tails. Admission reserves these pages
   even when the total remote end is a full compressed-page boundary.
4. The worker registers bounded block-major raw views from the existing slot
   copy plan, including indexer scale bytes, padding and storage offsets.
   Handshake checks protocol, group membership, page byte sizes and address
   counts. Physical page IDs are bounds-checked before RDMA reads.
5. D remains `WAITING_FOR_REMOTE_KVS` until the connector reports completion.
   Only then does `on_remote_cache_ready` publish the received snapshot and
   let local execution resume. Allocation and ordinary async completion do
   not publish pending transfers. P retains its request pages until the
   existing Mooncake ACK/timeout lifecycle releases them. Even a full D-local
   hit sends an ACK-only task so P does not wait unnecessarily.

Only transferred state can participate in a joint APC hit. An unaligned E
does **not** guarantee a hit at `floor(E / 128) * 128`: the state window for
that earlier boundary may not have been transferred. Compressed aliases alone
are insufficient. Subsequent local computation publishes valid new boundaries
normally, and the MTP/DSpark next-block check still applies.

Malformed manifests and mixed P/D layouts are rejected. A worker handshake
or RDMA failure is sticky and raises an error; it is **not** reported as a
successful receive. This phase deliberately has no automatic hybrid
partial-transfer/recompute fallback. An operational failure may stop the engine
and require service restart. Keep the existing producer lease timeout longer
than the intended request/transfer time. Graph execution, failover, timeout
races and production-scale throughput still require device-level acceptance.

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
  tests/ut/core/test_slot_apc_page_sizes.py \
  tests/ut/core/test_slot_apc_transfer.py \
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
Transfer tests additionally cover real coordinator allocation and scheduler
resume hooks, byte-for-byte copies through the actual receiver with a CPU
transfer-engine substitute, aligned/unaligned 128/512/16384 boundaries,
MTP/DSpark groups, mixed local/remote hits, admission failure, raw registration,
handshake mismatches, fast completion, failed receives and ACK-only full hits.
These tests do not execute actual RDMA or NPU kernels.
Physical-page tests additionally cover the 32/64/128 layouts and both state
geometries, actual hash-size resolution, C4/C128 full-page transitions,
abort/readmission, small-page draft peeks, cumulative async rejection, state
rewind retention, transferred draft bytes and mismatched P/D page sizes.

To measure CPU lookup and initial alias-publication costs independently of
NPU execution:

```bash
python benchmarks/benchmark_slot_apc.py --length 131072 --repeats 1000 --block-size 32
```

This microbenchmark is not a model-throughput or COW-device-copy benchmark.

To compare CPU allocation/snapshot/publication overhead with one or more
in-flight steps (also checks that all retained references are released):

```bash
VLLM_ASCEND_ENABLE_SLOT_APC=1 python benchmarks/benchmark_slot_apc_async.py \
  --prefix-length 131072 --steps 1024 --inflight-depth 2 --block-size 32
```

This synthetic benchmark uses known token IDs and no NPU execution. It does
not measure end-to-end async speedup or replace the device acceptance test.
Repeat with B=64 and B=128. Smaller pages can reduce tail-copy payloads but
increase block-table and fine-grained hash metadata work; do not infer overall
throughput or memory-capacity improvements from page size alone.

On four Ascend devices with the test model available:

```bash
pytest -q tests/e2e/pull_request/four_card/test_slot_apc.py
```

The device test runs B=32/64/128 with both sync and async scheduling and compares cold prefill
against four warmed requests with divergent suffixes and different output
lengths. It checks exact cached-token counts at 128, 256, 384, 512, 640,
16256, 16384 and 16512 tokens, greedy output equality, and generated-token
log probabilities. This test must pass on the deployment hardware before
enabling the feature there. Additional speculative cases run MTP and DSpark
with both scheduling modes at 128/384/512/16384-token hit boundaries. They use
an extra 128-token common block for draft validation and output lengths up to
257 to exercise new slot publication during decode. The tests also require
nonzero draft activity, so a silently disabled proposer cannot pass. The C128
4096/8192/16384-token page transitions are included for the corresponding B.
These device tests have not been executed in the Windows workspace; CPU
lifecycle tests do not validate NPU kernels, graph replay, or model accuracy.

For two running P/D servers, run the manual system acceptance script:

```bash
python tests/e2e/manual/slot_apc_mooncake.py \
  --prefill-url http://P_HOST:8000 --decode-url http://D_HOST:8000 \
  --model dsv4 --output-length 257 --block-size 32
```

It uses exact token-ID prompts and unique cache salts (no service cache reset),
compares cold D-only generation with cold/warm P-to-D generation and repeated
requests, checks greedy token IDs and log probabilities, and requires P-side
128-token local hits in the warmed cases. The services need
`--enable-prompt-tokens-details`. Start with DP=1 so the seeded requests reach
the same pool. Run separately for sync/async, no speculation, MTP and DSpark
using compatible checkpoints. Independently inspect D-side local/remote cache
metrics and producer memory reclamation under the intended DP routing. Use
isolated test services for injected RDMA failure, abort and lease-timeout tests;
the script does not inject faults or prove concurrency/failure recovery.
This NPU/RDMA acceptance has not been run in the Windows workspace.
The script's `--block-size` selects boundary cases; it does not reconfigure
either server. Restart both servers with the same B before testing B=64/128.

## Next stages

- Stage 3 device acceptance: run the MTP/DSpark cold/warm tests above before
  deployment. The local implementation and CPU lifecycle coverage are in place.
- Stage 4 device acceptance: run the P/D tests above and deployment-specific
  concurrency/failure tests. The local implementation and CPU regression
  coverage are in place. Other connectors remain out of scope.
- Physical block sizes 32/64: implementation and CPU regression coverage are
  in place, with APC still aligned to 128 original tokens. Device correctness,
  model accuracy and performance acceptance remain required for each size.
