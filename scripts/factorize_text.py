#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer

from relation_lm.tokenization import factorize_document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("--assets", type=Path, default=Path("assets/relationlex-16k-v1"))
    args = parser.parse_args()
    tokenizer = Tokenizer.from_file(str(args.assets / "tokenizer.json"))
    boundary_payload = json.loads((args.assets / "boundary_vocab.json").read_text())
    boundary_to_id = {
        value: index for index, value in enumerate(boundary_payload["id_to_boundary"])
    }
    encoded = factorize_document(tokenizer, args.text, boundary_to_id)
    print(json.dumps({
        "token_ids": encoded.token_ids.tolist(),
        "boundary_ids": encoded.boundary_ids.tolist(),
        "byte_lengths": encoded.byte_lengths.tolist(),
        "raw_bytes": encoded.raw_bytes,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
