"""Commit-message sentiment in [-1, 1]: a HuggingFace model when available, else a lexicon."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import Any, Literal, Protocol

log = logging.getLogger(__name__)

HF_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
Backend = Literal["auto", "lexicon", "hf"]

# Words developers use when frustrated or relieved. Neutral technical words are absent on
# purpose: "fix", "bug" or "error" describe the work, not the author's mood.
NEGATIVE = frozenset(
    "ugh argh damn dammit crap shit wtf stupid dumb idiot idiotic hate hated annoying ugly "
    "horrible terrible awful broken mess messy hack hacky kludge garbage silly oops sorry "
    "frustrating sigh sucks painful nasty insane ridiculous desperate useless weird".split()
)
POSITIVE = frozenset(
    "great nice good better cleaner clean improve improved improves elegant awesome "
    "finally happy love thanks thank cool neat simplify simplified simpler faster".split()
)
_WORD_RE = re.compile(r"[a-z']+")


class SentimentScorer(Protocol):
    name: str

    def score(self, texts: Sequence[str]) -> list[float]: ...


class LexiconSentiment:
    """(positive - negative) word hits, squashed to [-1, 1]. Fast and dependency-free."""

    name = "lexicon"

    def score_one(self, text: str) -> float:
        words = _WORD_RE.findall(text.lower())
        pos = sum(w in POSITIVE for w in words)
        neg = sum(w in NEGATIVE for w in words)
        return (pos - neg) / (pos + neg + 1) if pos or neg else 0.0

    def score(self, texts: Sequence[str]) -> list[float]:
        return [self.score_one(t) for t in texts]


class HFSentiment:
    """SST-2 DistilBERT: P(positive) mapped to [-1, 1]."""

    name = "hf"

    def __init__(self, model: str = HF_MODEL, batch_size: int = 64) -> None:
        from transformers import pipeline  # optional dependency

        self._pipe: Any = pipeline("sentiment-analysis", model=model, truncation=True)
        self._batch_size = batch_size

    def score(self, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        out = self._pipe(list(texts), batch_size=self._batch_size)
        return [r["score"] * (1 if r["label"] == "POSITIVE" else -1) for r in out]


def get_scorer(backend: Backend = "auto") -> SentimentScorer:
    """``hf`` requires the model; ``auto`` tries it and falls back to the lexicon."""
    if backend == "lexicon":
        return LexiconSentiment()
    try:
        return HFSentiment()
    except Exception as exc:
        if backend == "hf":
            raise
        log.warning("HuggingFace sentiment unavailable (%s); using lexicon.", type(exc).__name__)
        return LexiconSentiment()
