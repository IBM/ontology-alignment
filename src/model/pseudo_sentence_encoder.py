from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np
import pandas as pd
from torch import Tensor, nn
from sentence_transformers import SentenceTransformer


class PseudoSentenceEncoder(ABC):

    @property
    @abstractmethod
    def model(self) -> nn.Module: pass

    @abstractmethod
    def encode(self, pseudo_sents: Union[List[str], pd.Series], *args, **kwargs) \
        -> Union[List[Tensor], Tensor, np.ndarray]: pass

    def to_train(self):
        self.model.train()

    def to_eval(self):
        self.model.eval()

    def to_device(self, dev: str):
        assert "cuda" in dev or "cpu" in dev
        self.model.to(dev)


class SBertPSE(PseudoSentenceEncoder):

    def __init__(self, pretrained_model: str) -> None:
        super().__init__()
        self._sbert_model = SentenceTransformer(pretrained_model)

    @property
    def model(self) -> nn.Module:
        return self._sbert_model

    def encode(self,
               pseudo_sents: Union[List[str], pd.Series],
               batch_size: int = 32,
               device: str = 'cuda:0') -> Union[List[Tensor], Tensor, np.ndarray]:
        if isinstance(pseudo_sents, pd.Series):
            pseudo_sents = pseudo_sents.values  # type: ignore
        return self._sbert_model.encode(pseudo_sents, batch_size=batch_size, device=device)  # type: ignore
