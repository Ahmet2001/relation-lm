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
