from enum import Enum
from typing import List, Union

import numpy as np
from sentence_transformers.util import cos_sim, dot_score
from torch import Tensor


class SimilarityMetric(str, Enum):
    COSINE = "cosine"
    DOT = "dot"  # dot product is not a "real" metric but we treat is as such


def compute_similarities(metric: SimilarityMetric,
                         reps_a: Union[List[Tensor], Tensor, np.ndarray],
                         reps_b: Union[List[Tensor], Tensor, np.ndarray]) -> Tensor:
    if metric == SimilarityMetric.COSINE:
        return cos_sim(reps_a, reps_b)  # type: ignore
    elif metric == SimilarityMetric.DOT:
        return dot_score(reps_a, reps_b)  # type: ignore
    else:
        raise NotImplementedError("Only 'cosine' and 'dot' similarities are implemented!")
