# Contributing

Thank you for helping improve Relation LM.

## Development setup

```bash
python -m pip install -e '.[dev]'
pytest -q
ruff check .
```

## Pull requests

- Keep public APIs documented and typed.
- Add a correctness test for every kernel or routing change.
- Report benchmark protocol, GPU model, software versions, warmup, and repeats.
- Do not commit datasets, checkpoints, credentials, absolute user paths, or job logs.
- Preserve deterministic tie-breaking in selection kernels.

## Research changes

Changes that alter routing semantics should include paired quality evaluation,
not only speed measurements. Prefer the compatibility routing semantics unless
a new policy has been trained and validated for the changed candidate set.
