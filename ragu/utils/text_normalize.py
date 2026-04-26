import re
from abc import ABC, abstractmethod
from typing import List, Set
from typing_extensions import override

import pymorphy3


class BaseNormalizer(ABC):
    @abstractmethod
    def normalize(self, text: str) -> str:
        ...

    def normalize_batch(self, texts: List[str]) -> List[str]:
        return [self.normalize(t) for t in texts]


class PymorphyNormalizer(BaseNormalizer):
    """
    Class that provide text normalization via pymorphy3 (for Russian language).
    """

    _word_re = re.compile(r"\w+", flags=re.UNICODE)
    _DEFAULT_STOP_POS = {"PREP", "CONJ", "PRCL", "INTJ"}

    def __init__(
        self,
        lowercase: bool = True,
        min_token_length: int = 2,
        remove_numbers: bool = True,
        stopwords: Set[str] | None = None,
        stop_pos: Set[str] | None = None,
    ) -> None:
        self.lowercase = lowercase
        self.min_token_length = min_token_length
        self.remove_numbers = remove_numbers
        self.stopwords = set(stopwords) if stopwords is not None else set()
        self.stop_pos = set(stop_pos) if stop_pos is not None else self._DEFAULT_STOP_POS

        self._morph = pymorphy3.MorphAnalyzer()

    def _tokenize(self, text: str) -> List[str]:
        if self.lowercase:
            text = text.lower()
        return self._word_re.findall(text)

    def _is_valid(self, token: str) -> bool:
        if len(token) < self.min_token_length:
            return False
        if self.remove_numbers and token.isdigit():
            return False
        return True

    @override
    def normalize(self, text: str) -> str:
        """
        Normalize text.
        """
        result: List[str] = []

        for token in self._tokenize(text):
            if not self._is_valid(token):
                continue

            maybe_parsed = self._morph.parse(token)

            # If we can't lemmatize word, return it as is.
            if not maybe_parsed:
                result.append(token)
                continue

            parsed = maybe_parsed[0]
            if parsed.tag.POS in self.stop_pos or parsed.normal_form in self.stopwords:
                continue

            result.append(parsed.normal_form)

        return " ".join(result)