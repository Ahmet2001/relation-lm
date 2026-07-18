# Partner structure and multi-relation blocks

The current 20M checkpoint selects **one anchor and up to eight partners per
query position**. It does not select only one partner.

## Diagnostics

The report in `benchmarks/results/partner_structure_report_20m.json` evaluates
four validation batches at contexts 128, 256, and 512.

- A **global anchor-role collision** means a selected partner position is used
  as an anchor somewhere in the same sequence. This is diagnostic only because
  knowing future anchor choices would not be causal.
- A **causal seen-anchor collision** means a partner has already served as an
  anchor at or before the current query.
- A **shared partner** is a partner position used by at least two distinct
  anchor positions.
- Near/middle/far are thirds of the available backward distance, not fixed token
  counts.

At context 512, 24.6% of partner selections also have an anchor role somewhere
in the sequence, 16.9% collide with an anchor already known causally, and 98.0%
of selections use a partner shared by multiple distinct anchors. Excluding
previously seen anchors from partner selection changed joint BPB by only
+0.022%, suggesting that a causal role registry is plausible.

Partner influence is strongly rank-concentrated. At context 512, the first two
partners receive about 37.9% and 18.1% mean weight. Removing rank 1 worsens BPB
by about 0.237%; removing rank 2 worsens it by about 0.090%. Lower ranks have
small or occasionally negative measured effect, motivating learned/dynamic K.

The relative-distance distribution is highly local at context 512: about 94.0%
of selections and 93.2% of weight mass fall in the nearest third, 4.0%/4.4% in
the middle third, and 2.0%/2.4% in the far third.

## Multi-relation depth

`MultiAnchorRelationBlock` adds several distinct anchor slots to one relation
layer. Each active anchor selects multiple partners, and every selected anchor
position is masked out of every partner set in that layer. Early positions use
only `min(anchor_slots, t + 1)` active slots.

`RelationBlockStack` composes these layers sequentially, analogous to stacking
attention layers. The module is currently a tested architecture prototype, not
a trained replacement for the released 20M checkpoint.
