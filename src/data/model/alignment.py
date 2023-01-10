from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class SimiliarityMetric(str, Enum):
    COSINE = "cosine"
    DOT = "dot"


@dataclass(frozen=True)
class Alignment:
    a_uri: str
    b_uri: str

    similarity: float
    metric: Optional[SimiliarityMetric] = field(default=SimiliarityMetric.COSINE)

    def __post_init__(self) -> None:
        if not (isinstance(self.a_uri, str) and isinstance(self.b_uri, str)):
            raise ValueError("Unsupported type for URI!")
        # https://stackoverflow.com/questions/53756788/how-to-set-the-value-of-dataclass-field-in-post-init-when-frozen-true

        if self.metric == SimiliarityMetric.COSINE:
            assert -1. <= np.round(self.similarity, 5) <= 1.
        else:
            raise ValueError(f"{self.similarity} is out of range for metric '{self.metric}'!")

    def __str__(self) -> str:
        return f"Alginment({str(self.a_uri)} <=> {str(self.b_uri)} --> {self.similarity})"

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def _assert_is_valid_comparision(o: object) -> None:
        if not (isinstance(o, Alignment) or isinstance(o, float) or isinstance(o, int)):
            raise NotImplementedError

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Alignment):
            return False
        return (self.a_uri == o.a_uri and
                self.b_uri == o.b_uri and
                self.metric == o.metric and
                self.similarity == o.similarity)

    def __lt__(self, o: object) -> bool:
        self._assert_is_valid_comparision(o)

        if isinstance(o, Alignment):
            return self.similarity < o.similarity
        else:
            return self.similarity < o  # type: ignore

    def __le__(self, o: object) -> bool:
        self._assert_is_valid_comparision(o)

        if isinstance(o, Alignment):
            return self.similarity <= o.similarity
        else:
            return self.similarity <= o  # type: ignore

    def __gt__(self, o: object) -> bool:
        self._assert_is_valid_comparision(o)

        if isinstance(o, Alignment):
            return self.similarity > o.similarity
        else:
            return self.similarity > o  # type: ignore

    def __ge__(self, o: object) -> bool:
        self._assert_is_valid_comparision(o)

        if isinstance(o, Alignment):
            return self.similarity >= o.similarity
        else:
            return self.similarity >= o  # type: ignore

    def flip(self) -> "Alignment":
        return Alignment(self.b_uri, self.a_uri, self.similarity, self.metric)
