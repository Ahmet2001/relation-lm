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

Small JSON reports are stored in `benchmarks/results/`. Raw datasets, model
checkpoints, and cluster logs are intentionally excluded.
