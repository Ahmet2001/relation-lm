# Benchmarks

All headline numbers below were measured on the same TRUBA CUDA node with
inference parameters frozen, IEEE FP32 matmul mode, `torch.compile(fullgraph=True)`,
and fixed greedy generation.

## Paired quality validation

100 paired validation batches:

| context | sparse vs dense joint BPB |
|---:|---:|
| 128 | +0.127% |
| 256 | +0.187% |
| 512 | +0.119% |

## Stateful cached decode

Compiled incremental cache was 7–10x faster than eager full recomputation at
256–512 context. Before custom selection, sparse cached decode remained slower
than dense cached decode.

## Custom Triton selection

Nine-repeat median, context 512, 64 generated positions:

| batch | dense cached | generic sparse | Triton sparse |
|---:|---:|---:|---:|
| 1 | 2548 tok/s | 2153 tok/s | 2355 tok/s |
| 8 | 11274 tok/s | 9477 tok/s | 10107 tok/s |

Triton vs generic sparse:

- batch 1: 1.094x
- batch 8: 1.067x

## Packed relation_select

Nine-repeat median, context 512, 64 generated positions:

| batch | dense cached | generic sparse | two-kernel Triton | packed relation_select |
|---:|---:|---:|---:|---:|
| 1 | 2533 tok/s | 2143 tok/s | 2340 tok/s | 2333 tok/s |
| 8 | 11299 tok/s | 9504 tok/s | 10136 tok/s | 10385 tok/s |

Packed `relation_select` vs generic sparse:

- batch 1: 1.089x
- batch 8: 1.093x

At batch 8, packed fusion was 1.025x faster than the prior two-kernel Triton
path. Selection parity passed for contexts 128/256/512 at batches 1 and 8;
16-step fullgraph greedy parity passed with maximum logit difference below
`7e-6`.

A standalone synthetic selector benchmark, independent of private checkpoints,
measured the packed custom op at approximately 2.37x the speed of the two-op
selection path for both batch 1 and batch 8. The smaller end-to-end decode gain
shows that relation construction, the relation MLP, cache updates, and output
preparation now dominate more of the remaining runtime.

Small JSON reports are stored in `benchmarks/results/`. Raw datasets, model
checkpoints, and cluster logs are intentionally excluded.

## Cached factorized relation reduction

Nine-repeat median, context 512, 64 generated positions:

| batch | dense cached | packed select | best relation reduction | sparse/dense |
|---:|---:|---:|---:|---:|
| 1 | 2547 tok/s | 2347 tok/s | **2398 tok/s** | 0.942x |
| 8 | 11295 tok/s | 10357 tok/s | **10470 tok/s** | 0.927x |

The first Relation-MLP layer is factorized exactly as
`(W_a+W_d)a + (W_p-W_d)p + W_m(a*p) + b`. Anchor and partner
contributions are cached per memory token; only the multiplicative term is
computed for selected pairs. LayerNorm, active-K softmax, and weighted reduction
are fused into one Triton kernel.

The verified dispatcher uses fused current-token cache update for batch 1 and a
separate cuBLAS cache projection for batch 8. Context parity was below `1.1e-6`;
16-step fullgraph greedy parity passed with maximum logit difference below
`6.7e-6`. A fully custom Triton `2304 → 576` projection was slower than cuBLAS
and is intentionally not the recommended path.


## Packed sparse-cache projection

Three validation prefixes, nine interleaved repeats per prefix, context 512 and
64 generated positions:

| batch | dense cached | previous sparse | packed-cache sparse | packed vs previous | packed vs dense |
|---:|---:|---:|---:|---:|---:|
| 1 | 25.483 ms | 27.004 ms | 25.462 ms | 1.061x | 1.001x |
| 8 | 45.704 ms | 49.168 ms | 44.433 ms | 1.107x | 1.028x |

The projection concatenates anchor key, partner key, relation operand,
partner-anchor factor, anchor-router key, and partner-router key weights into a
single `576 → 512` matrix. Six cache tensors are recovered with view/split
operations before their state updates.

A 100-batch partner-budget confirmation found K=6 at +0.130% BPB versus dense,
but it was only 1.008x faster than K=8 at batch 1 and 1.002x at batch 8. K=8
therefore remains the default partner budget.


## Fused router cache update

The packed sparse-cache projection leaves two router-key views that must update
raw history, depthwise causal-convolution history, and current-block mean/std
statistics. A state-mutating Triton op now performs both router streams and all
eight cache/stat writes in one launch. Non-contiguous views from the packed
`576 → 512` projection are consumed directly through explicit row strides.

Three validation prefixes, nine interleaved repeats per prefix, context 512 and
64 generated positions:

| batch | dense cached | packed-cache sparse | fused-router sparse | fused vs previous | fused vs dense |
|---:|---:|---:|---:|---:|---:|
| 1 | 25.434 ms | 25.485 ms | **25.060 ms** | 1.017x | 1.015x |
| 8 | 45.833 ms | 44.519 ms | **43.749 ms** | 1.018x | 1.048x |

State parity covered block-end and block-start positions at batches 1 and 8;
the maximum cache difference was `9.54e-7`. Thirty-two-step fullgraph parity
was exact at the output-logit level for both verified batches. A standalone
1,000-repeat compiled microbenchmark measured the fused update at 1.041x for
batch 1 and 1.012x for batch 8.

The relation reduction kernel also pads its active-partner axis to the next
power of two, allowing verified non-power-of-two budgets such as K=5 without
materializing extra partner rows. K=8 remains the default quality/speed policy.
