# Overnight run report — multichunk traced Kimi prefill

Branch `ppopovic/trace_experiments`. Autonomous overnight session started 2026-06-26 ~22:57.
Plan: `/data/ppopovic/.claude/plans/i-want-to-do-twinkly-volcano.md`.

Policy: commit at checkpoints; grind blockers; runner trace gated behind `PREFILL_USE_TRACE`.
Device reset: `tt-smi -glx_reset` (kill pytest first; `pkill -9 -f chunked` for hung mutex).

## Env
- **UPDATED per user (2026-06-26 ~23:20): `KIMI_K2_6_HF_MODEL=/data/nbabin/Kimi-K2_6-dequantized`**
  (dot-free; same TTNN cache → identical results to kimi-forge). All runs after this use it.
- Tests (earlier runs): `KIMI_K2_6_HF_MODEL=/mnt/models/kimi-forge`,
  `TT_KIMI_PREFILL_TTNN_CACHE=/mnt/models/Kimi-K2_6-Cache/Kimi-K2_6-Cache-prefill`,
  `TT_KIMI_PREFILL_HOST_REF_CACHE=/mnt/models/kimi-prefill-cache/golden`.
- Runner: `PREFILL_MODEL_VARIANT=kimi_k2_6`, `PREFILL_HF_MODEL=models/demos/deepseek_v3_d_p/reference/kimi_k2_6`,
  `PREFILL_TTNN_CACHE=...prefill`, `PREFILL_TRACE_DIR=/mnt/models/deepseek-prefill-cache/golden/kimi-26/kimi_longbook_56320`.

## Timeline

### Phase A — baseline sanity
- L1 multichunk trace (11 chunks, metadata): **PASS** (31.86s). Trace mechanism healthy.
- **L10 single-chunk scalar KV-PCC baseline (committed HEAD, pre-edit): FAILED, min PCC 0.658 < 0.96.**
  - Breakdown: `nope` slice fine (~0.97-0.98), `pe` (RoPE) slice BROKEN (~0.70 on layers 7-9).
  - Memory `kimi-prefill-trace` recorded this same test PASSING at 0.9939 on 2026-06-24.
  - ⇒ a recent commit (ring_mla task5 series, or `1002cdf Trace updates`) regressed the **RoPE
    portion of the KV cache** on the SCALAR path (single chunk, use_metadata=False). nope is healthy.
  - Full log: scratchpad/phaseA_L10_baseline.log.

### Phase B — 11-chunk metadata + KV PCC (debugging)
Edits done: `assert_layer_depth` param on `_record_kv_cache_pcc`; relaxed `n_chunks==1` gate for the
metadata path; KV-PCC call added to the metadata branch; `trace_kv_pcc` test → 11 chunks + metadata.

**Two distinct problems found (data):**
- **Issue 1 — pe (RoPE) ~0.70 uniform, chunk 0, BOTH scalar+metadata.** Single-chunk scalar L10:
  nope healthy 0.97-0.9999, pe 0.66-0.77 on every layer incl. L0 (identity rope) ⇒ a pe *layout*
  mismatch (interleave vs half-split), not a rotation-angle error. Hypothesis: device-internal pe
  layout convention changed and the test's HF→Meta conversion is now wrong (model may be
  self-consistent), OR a real rope-write regression. Need candidate-layout diagnostic to decide.
- **Issue 2 — nope collapses with DEPTH, 11-chunk metadata only.** L1 nope 0.98 → L8 nope 0.11.
  Single-chunk nope was fine, so this is a multi-chunk accumulation bug (deeper hidden states drift
  across chunks) — consistent with the user's SDPA/ring_mla suspicion. Need: is it metadata-path
  specific or also on scalar multichunk?

**DECISIVE reference (scalar, non-trace, 11-chunk, decoder-output PCC vs golden):**
`test_kimi_prefill_transformer_chunked[L10-chunks11]` — chunk0 L0-9 all 0.99+; **layer 9 holds
0.991+ for EVERY chunk 0→10.** ⇒ the scalar multichunk MODEL is fully correct.

Conclusions:
- **Issue 1 (pe ~0.70) = TEST-COMPARISON ARTIFACT.** Model correct (decoder 0.997); device KV-cache
  pe layout is internally consistent but no longer matches the test's HF→Meta conversion. Fix = update
  the comparison (KV_PE_DEBUG run will reveal the true device pe layout).
- **Issue 2 (nope depth-collapse) is TRACE+METADATA-path specific**, not a model bug (scalar perfect).
  Implicates ring_mla on-device derivation under trace replay (task5: writer/compute/all-gather
  recompute from metadata). Matches user's "most likely sdpa" prediction.
- Added KV_PE_DEBUG diagnostic (candidate pe layouts + per-chunk-range nope) to `_record_kv_cache_pcc`.

**KV_PE_DEBUG metadata L10 results (sharp localization):**
- L0: nope PERFECT (0.9999) across ALL 11 chunk ranges. pe candidates: raw=0.10, hf2meta=0.75,
  meta2hf=0.13 (none ~0.99 → pe is a comparison-basis subtlety; scalar decoder ref proved chunk-0
  attention — which consumes this pe — correct at 0.997, so NOT a model bug).
- L8: nope ≈0.11 UNIFORMLY across every chunk range incl chunk[0:5120]. Monotonic depth-collapse
  (L1 0.98 → L8 0.11), position-independent.

⇒ **Root cause = metadata-path ring_mla attention output is slightly wrong even at chunk 0**, compounding
with depth. NOT trace (scalar+trace baseline had deep nope 0.98), NOT later-chunk masks (uniform per
position), NOT the model (scalar path perfect). Matches user's SDPA prediction.

**Per-op `test_ring_mla_metadata_matches_scalar_rotation`: PASSES bit-exact (kv64/256/320)**, even when
I forced the global-capacity placeholder logical_n (temp edit, reverted). So ring_mla's metadata
derivation is correct in isolation at tiny dims.

**Single-chunk metadata L10 (chunks1): collapses IDENTICALLY to 11-chunk** (L0 nope 0.9999 → L4 0.45 →
L8 0.11). ⇒ **bug is the metadata PATH itself at chunk 0, NOT multichunk.** Clean isolation now:
same chunk 0, same trace mechanism — scalar=good, metadata=collapse-with-depth.

⇒ Bug is in how the MODEL drives the metadata path at **Kimi dims** (per-op tests use tiny dims & can't
trigger it). nope-L0-perfect rules out update_padded_kv_cache. Suspects: Q-rope(metadata) and
ring_mla(metadata) at Kimi dims.

## ROOT CAUSE FOUND + CONFIRMED (Issue 2) — ring_mla metadata path drops the per-layer cache slot

**Confirmations:**
- L1 (single layer) metadata single-chunk: **PASS, nope=0.9999 AND pe=0.9999** (slot 0 correct for 1 layer).
- L10 metadata: L0 fine, layers>0 collapse (monotonic depth).
- per-op `test_ring_mla_metadata_matches_scalar_rotation`: PASS — uses a single slot/layer, so can't catch it.

**Mechanism (code-verified):**
- KV cache batch dim is (user,layer)-major: `cache_batch_idx = cache_user_id*num_layers + cache_layer_idx`.
- SCALAR ring_mla (mla.py:684) passes `kv_cache_batch_idx = cache_user_id*layer_num + cache_layer_idx`;
  host computes `input_batch_base = input_batch_base_pages(kv_cache_batch_idx,...)` (all_gather factory :525).
- METADATA ring_mla (mla.py:681) passes only `metadata`; the all-gather reader kernel (line 187-188)
  recomputes `input_batch_base = slot_id*heads*Ht*Wt` where `slot_id = metadata[0] = cache_user_id`
  — **MISSING `*num_layers + cache_layer_idx`.** `num_layers`/`layer_idx` exist NOWHERE in ring_joint /
  all-gather code. update_padded_kv_cache DOES take layer_idx+num_layers (so nope writes the right slot →
  L0 nope perfect), but ring_mla READS slot 0 for every layer.
- ⇒ layer i>0 ring_mla reads layer-0's KV → wrong attention → corrupts residual → compounds with depth.
  pe=0.77 at L10-L0 is a downstream artifact (L1 proves the pe comparison transform is correct).

**FIX (planned):** thread `cache_layer_idx` + `num_layers` (structural, trace-safe) into ring_mla's
metadata path; in the all-gather reader AND SDPA reader compute `slot = metadata[0]*num_layers +
cache_layer_idx` (mirror update_padded_kv_cache). Defaults (stride=1, offset=0) keep existing callers
bit-identical. Then re-run L1/L10/L61 KV PCC. Kernel change ⇒ ttnncpp rebuild + .so refresh.

### FIX IMPLEMENTED + VERIFIED ✅
Added `kv_cache_num_layers` (default 1) + `kv_cache_layer_idx` (default 0) to ring_mla, threaded through
ring_joint_sdpa device op + program factory + the fused all-gather helper to BOTH readers
(SDPA `ring_joint_reader.cpp`, all-gather `ring_attention_all_gather_reader.cpp`); slot now
`meta[0]*num_layers + layer_idx`. Hashed (one program/layer — trace-safe, each layer captures own trace).
mla.py `_chunked_attn` metadata branch passes `kv_cache_num_layers=self.layer_num,
kv_cache_layer_idx=cache_layer_idx`. 12 files; ttnncpp+ttnn(nanobind) rebuilt + .so refreshed.
- Per-op `test_ring_mla_metadata_matches_scalar_{rotation,indexed}`: **5 PASS** (defaults 1/0 → bit-identical).
- **L10 single-chunk metadata: PASS, min PCC 0.993906** (all layers nope 0.99+, pe 0.9999) — matches the
  historical 0.9939. The pe=0.77 anomaly was downstream of the slot bug; gone now.
Next: L10 11-chunk (the deliverable) + L61.

### PHASE B RESULTS (post-fix) ✅
- **L10 11-chunk metadata KV PCC: PASS, min 0.994096** (all 10 layers nope 0.99+, pe 0.9999) — the full
  56320-token cache matches golden across 11 chunks via ONE captured metadata trace. Deliverable met.
- **L61 11-chunk metadata KV PCC: PASS** (600s). Asserted layers 0-10 min 0.993545; full 61-layer min
  0.966851 (deep layers L59/60 nope ~0.97 = bf8 depth accumulation, healthy); pe all 0.998+. The full
  61-layer KV cache matches golden across 11 chunks via ONE captured metadata trace.
Committed: 10cea053409 (the fix) + 3fb9d4573c7 (WIP test scaffold + root cause).

**PHASE B DONE ✅** — both L10 and L61 11-chunk metadata KV-PCC pass; ring_mla per-layer slot bug fixed.

### Phase C — runner traced loop (gated PREFILL_USE_TRACE) — IN PROGRESS
Foundation done (Python, no device): extended `SubDeviceTraceController` with a per-layer ack boundary
(`set_layer_ack_callback`/`has_layer_ack`/`layer_ack`; capture splits w/o injecting, replay injects
between segments); `mla.py` both ack sites (_chunked_attn, _forward_kv_only) route through the controller
when it carries an ack callback, else direct sync+call (test path: controller has no ack cb → unchanged);
wired `set_trace_controller` transformer→block→MLA. Remaining: pipeline metadata-trace path with
persistent inbound tensors + runner PREFILL_USE_TRACE gate, then standalone KV-PCC validation.
