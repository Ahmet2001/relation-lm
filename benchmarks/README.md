# Benchmarks

Small machine-readable reports are committed here. Large raw outputs are excluded.


## Router cache update

```bash
python benchmarks/benchmark_router_cache_update.py
```

This compares the compiled tensor-operation update with the state-mutating
Triton operator using non-contiguous router-key views from a packed projection.
