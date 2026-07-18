# Distance-aware multi-anchor RelationBlocks

This experiment adapts the trained 20M RelationLex model into three independent
RelationBlocks. Each block selects four anchors and four partners per anchor.
Every selected anchor is removed from every partner pool.

The layer roles are fixed by partner-distance quota:

| Block | Near | Middle | Far |
|---:|---:|---:|---:|
| 1 | 2 | 1 | 1 |
| 2 | 1 | 2 | 1 |
| 3 | 1 | 1 | 2 |

The model was adapted for 500 mixed-context steps with 2,048 positions per step.
It has 25,166,245 parameters versus
20,391,587 for the continued single-block
baseline, so this is a structural experiment rather than a parameter-matched
claim.

## Paired quality result

One hundred validation batches were evaluated on exactly the same examples.
Negative values favor the three-block model.

| Context | Global BPB difference | Paired relative 95% CI |
|---:|---:|---:|
| 128 | -0.0245% | [-0.0537%, -0.0195%] |
| 256 | -0.0786% | [-0.1121%, -0.0683%] |
| 512 | -0.1268% | [-0.1275%, -0.0751%] |

The gain is small but the paired intervals are below zero in this run.
Training throughput is 0.485x
and incremental activation memory is
2.422x the baseline.

## Block roles at context 512

| Block | Selected near/middle/far | Mixture mass near/middle/far | Mix scale |
|---:|---|---|---:|
| 1 | 0.499/0.251/0.250 | 0.821/0.109/0.070 | 0.2193 |
| 2 | 0.250/0.500/0.250 | 0.337/0.408/0.255 | 0.0479 |
| 3 | 0.250/0.251/0.499 | 0.348/0.180/0.472 | 0.0479 |

The quotas successfully create local, middle-range, and distant relation roles.
All anchor-as-partner violation counts are zero.

## Partner and block ablations

Removing a block increases absolute BPB by:

| Ablation | Mean | 95% CI |
|---|---:|---:|
| Block 1 | 0.02161 | [0.01953, 0.02369] |
| Block 2 | 0.01096 | [0.00929, 0.01263] |
| Block 3 | 0.01282 | [0.01107, 0.01456] |
| All | 0.22578 | [0.21649, 0.23508] |

All four partner ranks have non-zero leave-one-out effects. However, hard partner
selection remains highly redundant: at context 512 approximately
10.49,
10.02, and
10.18 of the 16 slots are
duplicates. Queries with a shared partner are about
99.2%. Consecutive Block 2 to
Block 3 partner overlap is
99.3%.

This means distance specialization works, but the soft overlap penalty is not
strong enough to produce distinct hard assignments. A future version should
use exclusive assignment, Sinkhorn matching, or a hard duplicate budget.
