# Roadmap

## Near term

- [x] Single packed `relation_select` Triton operator.
- [ ] Fuse relation operand gather and weighted relation reduction.
- [ ] Persistent ring-buffer state and contexts beyond 512.
- [ ] Kernel autotuning by batch, context, and head dimensions.
- [ ] Nsight breakdown for router, selection, relation MLP, and output head.

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
