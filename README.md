# Relation LM

Relation LM is an experimental language-model architecture that replaces dense
all-pairs attention with explicit **anchor–partner relations**. The repository
contains the RelationLex dual-channel tokenizer, dense and routed model
components, stateful decode experiments, and Relation-LM-specific Triton
selection kernels.

> Research status: sparse routing preserves dense Relation LM quality with a
> small BPB delta, while end-to-end sparse decode is still being optimized.

## Core idea

For each query position, Relation LM:

1. selects one high-value **anchor** from the causal history;
2. selects a small set of **partners** conditioned on the query and anchor;
3. constructs explicit relation features
   `[anchor, partner, anchor*partner, anchor-partner]`;
4. aggregates relation vectors into the next-token state.

The routed implementation first scores historical blocks, then performs exact
selection only inside a local window and a few remote blocks.

## RelationLex

RelationLex represents a document with aligned channels:

- lexical/punctuation subword IDs;
- exact whitespace-boundary IDs;
- optional raw-byte contribution counts.

Whitespace remains losslessly reconstructable without consuming ordinary
sequence positions. The included `relationlex-16k-v1` tokenizer has a 16,000
item lexical vocabulary. Boundary IDs are train-derived and kept separate from
lexical IDs.

```python
from pathlib import Path
from tokenizers import Tokenizer
from relation_lm.tokenization import factorize_document, reconstruct_document

root = Path("assets/relationlex-16k-v1")
tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
boundary_vocab = __import__("json").loads(
    (root / "boundary_vocab.json").read_text()
)["id_to_boundary"]
boundary_to_id = {value: index for index, value in enumerate(boundary_vocab)}

encoded = factorize_document(tokenizer, "hello\n  world", boundary_to_id)
text = reconstruct_document(
    tokenizer, encoded.token_ids, encoded.boundary_ids, boundary_vocab
)
assert text == "hello\n  world"
```

## Triton selection kernels

`relation_lm.kernels.triton_select` implements Relation-LM-specific operators:

- fused anchor block routing + exact anchor selection;
- fused partner block routing + small-K exact partner selection;
- deterministic causal masking and anchor exclusion;
- compatibility and strict-valid remote-block semantics.

The packed `relation_select` operator fuses anchor routing, exact anchor
selection, factorized partner projections, partner routing, and small-K partner
selection into one custom op. On the 512-context stateful decode benchmark it
was **8.9% faster at batch 1** and **9.3% faster at batch 8** than the generic
Inductor/`torch.topk` sparse path. At batch 8 it also beat the earlier two-kernel
Triton path by **2.5%**.

The packed sparse-cache projection then concatenates six token-local memory
projections into one `576 → 512` linear operation. In a three-sample,
interleaved context-512 benchmark this made sparse decode **6.1% faster at
batch 1** and **10.7% faster at batch 8** than the previous sparse path. The
result matched dense decode at batch 1 and was **2.8% faster at batch 8**.

## Installation

```bash
python -m pip install -e .
```

For development:

```bash
python -m pip install -e '.[dev]'
pytest -q
```

GPU kernels require a CUDA-enabled PyTorch build and Triton.

## Repository layout

```text
assets/                 RelationLex tokenizer and boundary vocabulary
benchmarks/             Reproducible benchmark entry points and small reports
docs/                    Architecture, results, and roadmap
examples/                Minimal usage examples
scripts/                 Dataset/tokenization utilities
src/relation_lm/
  tokenization/          Lossless RelationLex factorization
  models/                Dense Relation LM reference model
  routing/               Distilled block router
  kernels/               Triton selection kernels
  inference/             Decode interfaces and experimental state
 tests/                  CPU tests and optional CUDA parity tests
```

## Current verified findings

- 100-batch paired validation sparse BPB deltas:
  - context 128: +0.127%
  - context 256: +0.187%
  - context 512: +0.119%
- compiled/eager parity: approximately `2e-5` or better;
- stateful compiled cache: up to 10x faster than eager full recomputation;
- packed fused `relation_select`: 8.9–9.3% faster than generic sparse selection;
- cached factorized relation hidden + fused norm/reduction reduces relation-tail
  overhead while retaining approximately `7e-6` compiled parity;
- one packed `576 → 512` sparse-cache projection replaces six small linear
  projections and improves the previous sparse path by 6.1–10.7%;
- context-512 sparse decode is now approximately tied with dense at batch 1 and
  2.8% faster at batch 8 in the verified three-sample benchmark;
- reducing the partner budget from 8 to 6 passed quality but improved the robust
  K=8 path by less than 1%, so the recommended default remains K=8.

See [docs/benchmarks.md](docs/benchmarks.md) for protocol details.

## Roadmap

1. Fuse packed router-key projection, causal convolution, and block-statistics
   updates into one state mutation operator.
2. Add persistent/ring-buffer state for contexts beyond 512.
3. Fuse last-token output preparation where it improves end-to-end latency.
4. Publish reproducible training recipes, checkpoints, and multi-seed baselines.

## Contributing

Issues and pull requests are welcome. Please read
[CONTRIBUTING.md](CONTRIBUTING.md) before submitting changes.

## License

MIT. See [LICENSE](LICENSE).
