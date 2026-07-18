# Roadmap

## Near term

- [x] Single packed `relation_select` Triton operator.
- [x] Packed sparse cache projection for token-local memory features.
- [x] Fuse relation operand gather and weighted relation reduction.
- [ ] Persistent ring-buffer state and contexts beyond 512.
- [ ] Kernel autotuning and batch-adaptive relation-cache dispatch.
- [ ] Optimize the remaining 2304-to-576 relation projection and output head.

## Reproducibility

- [ ] Public small-data training recipe.
- [ ] Open checkpoints with model cards.
- [ ] Three-seed matched Transformer comparison.
- [ ] Dataset-independent tokenizer training script.

## Research

- [ ] Learned variable partner budget.
- [ ] Router calibration and uncertainty ablations.
- [ ] Relation feature compression.
- [ ] Hybrid dense/sparse policy by context and batch size.
