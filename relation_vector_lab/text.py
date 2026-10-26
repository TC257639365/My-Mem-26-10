"""Deterministic multilingual text helpers used by the lexical and smoke paths."""

from __future__ import annotations

import re
import unicodedata


_LATIN_TOKEN = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def multilingual_tokens(text: str) -> list[str]:
    """Return word tokens plus CJK unigrams/bigrams; never assumes English-only input."""

    value = normalize_text(text)
    tokens = _LATIN_TOKEN.findall(value)
    cjk = [char for char in value if "\u3400" <= char <= "\u9fff"]
    tokens.extend(f"c1:{char}" for char in cjk)
    tokens.extend(f"c2:{a}{b}" for a, b in zip(cjk, cjk[1:]))
    return tokens


def serialize_view(kind: str, text: str) -> str:
    """Keep view boundaries visible to the encoder without adding relation labels."""

    label = kind.upper().replace("_", " ")
    return f"[{label}] {text.strip()}"

