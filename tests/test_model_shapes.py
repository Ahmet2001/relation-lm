import torch

from relation_lm.models import RelationLexLM, RelationLMConfig


def test_model_shapes_and_causality() -> None:
    torch.manual_seed(7)
    config = RelationLMConfig(
        vocab_size=128,
        boundary_vocab_size=8,
        d_model=32,
        ff_mult=2,
        memory_blocks=2,
        query_blocks=1,
        query_stem_blocks=1,
        query_lags=(1, 2, 4),
        gate_dim=16,
        relation_dim=8,
        k_max=4,
        max_context=32,
    )
    model = RelationLexLM(config).eval()
    tokens = torch.randint(0, config.vocab_size, (2, 12))
    boundaries = torch.randint(0, config.boundary_vocab_size, (2, 12))
    token_logits, boundary_logits = model(tokens, boundaries)
    assert token_logits.shape == (2, 12, config.vocab_size)
    assert boundary_logits.shape == (2, 12, config.boundary_vocab_size)

    changed_tokens = tokens.clone()
    changed_boundaries = boundaries.clone()
    changed_tokens[:, 8:] = torch.randint(0, config.vocab_size, changed_tokens[:, 8:].shape)
    changed_boundaries[:, 8:] = torch.randint(
        0, config.boundary_vocab_size, changed_boundaries[:, 8:].shape
    )
    changed = model(changed_tokens, changed_boundaries)
    assert torch.allclose(token_logits[:, :8], changed[0][:, :8], atol=1e-5, rtol=1e-5)
    assert torch.allclose(boundary_logits[:, :8], changed[1][:, :8], atol=1e-5, rtol=1e-5)
