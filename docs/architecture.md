# Architecture

## Dense reference

Relation LM maintains separate query and memory streams. The query stream uses
fixed-lag causal mixers; the memory stream uses token-wise residual MLP blocks.
For every query position, dense routing chooses one anchor and up to `k_max`
partners from the causal history.

The relation feature for anchor `a` and partner `p` is:

```text
[a, p, a * p, a - p]
```

A relation MLP maps each feature to model width. Softmax-weighted partner
relations are added to the query residual before token and boundary heads.

## Sparse routing

The distilled router summarizes causal blocks with per-head mean and standard
deviation. A query scores each block through content, uncertainty, and
relative-distance terms. Exact anchor/partner scoring is restricted to a local
window and a small number of remote blocks.

## Stateful decode

Memory MLP blocks are token independent, and query history uses fixed lags.
Therefore prior states can be cached exactly. The stateful implementation stores
memory vectors, anchor/partner keys, relation operands, query histories, and
router block statistics. Only the new position is updated during decode.

## Kernel direction

The current custom Triton operators fuse block routing with exact anchor or
partner selection. The next operator, `relation_select`, will fuse the boundary
between the two stages: anchor selection, anchor gather, partner projections,
and partner top-k.
