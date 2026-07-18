"""Lossless dual-channel RelationLex factorization."""
from __future__ import annotations

import collections
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from tokenizers import Tokenizer

_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class RelationLexEncoding:
    token_ids: np.ndarray
    boundary_ids: np.ndarray
    byte_lengths: np.ndarray

    @property
    def raw_bytes(self) -> int:
        return int(self.byte_lengths.sum(dtype=np.uint64))


def collect_boundary_vocabulary(texts: Iterable[str]) -> tuple[list[str], collections.Counter[str]]:
    """Build a deterministic train-only whitespace-boundary vocabulary."""
    counts: collections.Counter[str] = collections.Counter({"": 0})
    for text in texts:
        counts.update(match.group(0) for match in _WHITESPACE.finditer(text))
    ordered = [""] + [
        boundary
        for boundary, _ in sorted(
            ((key, value) for key, value in counts.items() if key),
            key=lambda item: (-item[1], item[0].encode("utf-8")),
        )
    ]
    return ordered, counts


def _utf8_prefix(text: str) -> list[int] | None:
    if text.isascii():
        return None
    prefix = [0]
    total = 0
    for character in text:
        total += len(character.encode("utf-8"))
        prefix.append(total)
    return prefix


def _char_to_byte(prefix: list[int] | None, position: int) -> int:
    return position if prefix is None else prefix[position]


def factorize_document(
    tokenizer: Tokenizer,
    text: str,
    boundary_to_id: dict[str, int],
    *,
    strict_boundaries: bool = True,
) -> RelationLexEncoding:
    """Encode text into aligned lexical, boundary, and byte-count channels.

    The tokenizer must expose ``<s>`` and ``</s>`` token IDs. Pure-whitespace
    tokenizer pieces are removed; their exact strings are stored in the
    boundary channel preceding the next lexical piece. Trailing whitespace is
    aligned with EOS.
    """
    encoding = tokenizer.encode(text, add_special_tokens=False)
    prefix = _utf8_prefix(text)
    total_bytes = len(text.encode("utf-8"))
    previous_char_end = 0
    previous_byte_end = 0
    lexical_ids: list[int] = []
    lexical_boundaries: list[int] = []
    lexical_bytes: list[int] = []

    for token_id_raw, (start_raw, end_raw) in zip(encoding.ids, encoding.offsets, strict=False):
        token_id = int(token_id_raw)
        start, end = int(start_raw), int(end_raw)
        if not 0 <= start <= end <= len(text):
            raise ValueError(f"invalid tokenizer offset {(start, end)} for {len(text)} chars")
        surface = text[start:end]
        if surface and surface.isspace():
            continue

        boundary = text[previous_char_end:start] if start >= previous_char_end else ""
        if boundary and not boundary.isspace():
            raise ValueError(f"non-whitespace tokenizer gap: {boundary!r}")
        if boundary not in boundary_to_id:
            if strict_boundaries:
                raise KeyError(f"unseen boundary: {boundary!r}")
            boundary_id = 0
        else:
            boundary_id = boundary_to_id[boundary]

        end_byte = _char_to_byte(prefix, end)
        lexical_ids.append(token_id)
        lexical_boundaries.append(boundary_id)
        lexical_bytes.append(max(0, end_byte - previous_byte_end))
        previous_char_end = max(previous_char_end, end)
        previous_byte_end = max(previous_byte_end, end_byte)

    trailing = text[previous_char_end:]
    if trailing and not trailing.isspace():
        raise ValueError(f"non-whitespace trailing tokenizer gap: {trailing!r}")
    if trailing not in boundary_to_id:
        if strict_boundaries:
            raise KeyError(f"unseen trailing boundary: {trailing!r}")
        trailing_id = 0
    else:
        trailing_id = boundary_to_id[trailing]

    bos_id = tokenizer.token_to_id("<s>")
    eos_id = tokenizer.token_to_id("</s>")
    if bos_id is None or eos_id is None:
        raise ValueError("tokenizer must define <s> and </s>")

    result = RelationLexEncoding(
        token_ids=np.asarray([bos_id, *lexical_ids, eos_id], dtype=np.uint16),
        boundary_ids=np.asarray([0, *lexical_boundaries, trailing_id], dtype=np.uint16),
        byte_lengths=np.asarray(
            [0, *lexical_bytes, total_bytes - previous_byte_end], dtype=np.uint32
        ),
    )
    if not (
        len(result.token_ids) == len(result.boundary_ids) == len(result.byte_lengths)
    ):
        raise AssertionError("RelationLex channel lengths differ")
    if result.raw_bytes != total_bytes:
        raise AssertionError(f"accounted {result.raw_bytes} bytes, expected {total_bytes}")
    return result


def reconstruct_document(
    tokenizer: Tokenizer,
    token_ids: Sequence[int] | np.ndarray,
    boundary_ids: Sequence[int] | np.ndarray,
    boundary_vocab: Sequence[str],
) -> str:
    """Exactly reconstruct one document from lexical and boundary channels."""
    if len(token_ids) != len(boundary_ids):
        raise ValueError("token and boundary channel lengths differ")
    chunks: list[str] = []
    pending_ids: list[int] = []
    for token_id, boundary_id in zip(token_ids[1:-1], boundary_ids[1:-1], strict=False):
        boundary = boundary_vocab[int(boundary_id)]
        if boundary:
            if pending_ids:
                chunks.append(tokenizer.decode(pending_ids, skip_special_tokens=False))
                pending_ids.clear()
            chunks.append(boundary)
        pending_ids.append(int(token_id))
    if pending_ids:
        chunks.append(tokenizer.decode(pending_ids, skip_special_tokens=False))
    chunks.append(boundary_vocab[int(boundary_ids[-1])])
    return "".join(chunks)
