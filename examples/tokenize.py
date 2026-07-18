import json
from pathlib import Path

from tokenizers import Tokenizer

from relation_lm.tokenization import factorize_document, reconstruct_document

assets = Path(__file__).resolve().parents[1] / "assets" / "relationlex-16k-v1"
tokenizer = Tokenizer.from_file(str(assets / "tokenizer.json"))
boundary_vocab = json.loads((assets / "boundary_vocab.json").read_text())["id_to_boundary"]
boundary_to_id = {value: index for index, value in enumerate(boundary_vocab)}

text = "Relation LM\n  keeps whitespace lossless."
encoded = factorize_document(tokenizer, text, boundary_to_id)
print(encoded)
print(reconstruct_document(tokenizer, encoded.token_ids, encoded.boundary_ids, boundary_vocab))
