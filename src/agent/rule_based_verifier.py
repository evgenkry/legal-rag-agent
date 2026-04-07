"""Rule-based верификация перед LLM-верификацией"""

import re
from dataclasses import dataclass


SOURCE_PATTERN = re.compile(r"(ст\.|статья|тк\s*рф)", re.IGNORECASE)
WORD_PATTERN = re.compile(r"\b[а-яa-z0-9]{4,}\b", re.IGNORECASE)


@dataclass
class RuleBasedResult:
    passed: bool
    reason: str | None = None


class RuleBasedVerifier:
    """Быстрая дешевая верификация без траты токенов LLM-модели."""

    @staticmethod
    def check_has_sources(answer: str) -> bool:
        return bool(SOURCE_PATTERN.search(answer or ""))

    @staticmethod
    def check_grounding(answer: str, docs: list) -> bool:
        answer_tokens = {
            t.lower()
            for t in WORD_PATTERN.findall(answer or "")
            if t and t.lower() not in {"когда", "можно", "нужно", "если", "ваше", "ваш"}
        }
        if not answer_tokens:
            return False
        corpus = " ".join((d.node.get_content() if getattr(d, "node", None) else "") for d in docs)
        corpus_lower = corpus.lower()
        hits = sum(1 for token in answer_tokens if token in corpus_lower)
        return hits >= 2

    def verify(self, answer: str, docs: list) -> RuleBasedResult:
        if not self.check_has_sources(answer):
            return RuleBasedResult(False, "rule_no_sources")
        if not self.check_grounding(answer, docs):
            return RuleBasedResult(False, "rule_not_grounded")
        return RuleBasedResult(True, None)
