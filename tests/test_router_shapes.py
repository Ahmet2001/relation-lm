import torch

from relation_lm.routing import BlockRouterConfig, DistilledBlockRouter


def test_router_shapes() -> None:
    config = BlockRouterConfig(d_model=32, block_size=4, heads=2, head_dim=8)
    router = DistilledBlockRouter(config)
    query = torch.randn(2, 16, 32)
    memory = torch.randn(2, 16, 32)
    anchor = torch.randn(2, 16, 32)
    anchor_scores = router.anchor_scores(query, memory)
    partner_scores = router.partner_scores(query, anchor, memory)
    assert anchor_scores.shape == (2, 16, 4)
    assert partner_scores.shape == (2, 16, 4)
    assert torch.isfinite(anchor_scores[:, -1]).any()
