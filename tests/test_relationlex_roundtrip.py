from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer

from relation_lm.tokenization import factorize_document, reconstruct_document

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "relationlex-16k-v1"


def test_relationlex_roundtrip() -> None:
    tokenizer = Tokenizer.from_file(str(ASSETS / "tokenizer.json"))
    boundary_vocab = json.loads((ASSETS / "boundary_vocab.json").read_text())["id_to_boundary"]
    boundary_to_id = {value: index for index, value in enumerate(boundary_vocab)}
    samples = [
        "hello world",
        "Unicode: İstanbul, 世界",
        "no-boundary",
    ]
    samples.extend(f"left{boundary}right" for boundary in boundary_vocab[1:12])
    for text in samples:
        encoded = factorize_document(tokenizer, text, boundary_to_id)
        decoded = reconstruct_document(
            tokenizer, encoded.token_ids, encoded.boundary_ids, boundary_vocab
        )
        assert decoded == text
        assert encoded.raw_bytes == len(text.encode("utf-8"))
