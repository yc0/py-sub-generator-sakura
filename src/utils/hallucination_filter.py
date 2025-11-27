"""Hallucination filtering utilities for translation outputs."""

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

from ..models.subtitle_data import TranslationResult

logger = logging.getLogger(__name__)


@dataclass
class HallucinationFilter:
    """Filters out hallucinated or low-quality translations."""

    enabled: bool = True
    confidence_threshold: float = 0.3
    max_length_ratio: float = 4.0
    min_length_ratio: float = 0.2
    blacklist: List[str] = None
    whitelist_patterns: List[str] = None

    def __post_init__(self):
        if self.blacklist is None:
            self.blacklist = [r"^\.{2,}$", r"^…+$", r"^\[.*\]$"]
        if self.whitelist_patterns is None:
            self.whitelist_patterns = [r"[\u4e00-\u9fff]" , r"[a-zA-Z]+"]
        self._compiled_blacklist = [re.compile(p, re.IGNORECASE) for p in self.blacklist]
        self._compiled_whitelist = [re.compile(p, re.IGNORECASE) for p in self.whitelist_patterns]

    def filter(
        self, results: List[TranslationResult], language: Optional[str] = None
    ) -> List[TranslationResult]:
        if not self.enabled or not results:
            return results

        filtered: List[TranslationResult] = []
        removed = 0

        for result in results:
            text = result.translated_text.strip()
            if not text:
                removed += 1
                continue

            if self._is_blacklisted(text):
                removed += 1
                continue

            if not self._has_whitelisted_content(text):
                removed += 1
                continue

            if result.confidence is not None and result.confidence < self.confidence_threshold:
                removed += 1
                continue

            length_ratio = self._calculate_length_ratio(text, result.original_text)
            if length_ratio is not None and (
                length_ratio > self.max_length_ratio or length_ratio < self.min_length_ratio
            ):
                removed += 1
                continue

            filtered.append(result)

        if removed:
            logger.info(
                "Hallucination filter removed %d %s translations",
                removed,
                f"({language})" if language else "",
            )

        return filtered

    def _is_blacklisted(self, text: str) -> bool:
        return any(pattern.match(text) for pattern in self._compiled_blacklist)

    def _has_whitelisted_content(self, text: str) -> bool:
        return any(pattern.search(text) for pattern in self._compiled_whitelist)

    def _calculate_length_ratio(self, translated: str, original: str) -> Optional[float]:
        if not original:
            return None
        if not translated:
            return 0.0
        try:
            return len(translated) / len(original)
        except ZeroDivisionError:
            return None
