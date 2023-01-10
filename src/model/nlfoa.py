import logging
from typing import Union
from pathlib import Path

from typing import List, Optional

from torch import nn, Tensor
from sentence_transformers import SentenceTransformer, models
import pandas as pd
import numpy as np

from model.pseudo_sentence_encoder import PseudoSentenceEncoder


class NLFOA(nn.Module, PseudoSentenceEncoder):
    """
    Natural Language Focussed Ontology Alignment model
    """

    def __init__(self,
                 model_path: Optional[Union[Path, str]] = None,
                 special_characters: Optional[List[str]] = None,
                 word_embedding_model: Optional[str] = None,
                 max_seq_length: Optional[int] = None,
                 sts_model: Optional[str] = None,
                 pooling_mode: Optional[str] = None,
                 device: str = 'cuda:0'):
        super().__init__()

        if word_embedding_model is None:
            if model_path is not None:
                logging.info(f"Instantiating pretrained NLFOA from {model_path}!")
                self._sbert_model = SentenceTransformer(model_name_or_path=str(model_path), device=device)
                self.word_embedding_model = self._sbert_model._first_module()
                self.special_tokens = self.word_embedding_model.tokenizer.get_added_vocab()
                self.pooling_layer = self._sbert_model._last_module()
            elif sts_model is not None:
                logging.info(f"Instantiating NLFOA from pretrained {sts_model} SentenceTransformer!")
                self._sbert_model = SentenceTransformer(model_name_or_path=str(sts_model), device=device)
                self.word_embedding_model = self._sbert_model._first_module()
                self.pooling_layer = self._sbert_model._last_module()
                self._add_special_characters(special_characters)
        else:
            logging.info(f"Instantiating NLFOA from pretrained {word_embedding_model}!")
            # setup word embedding model and add special tokens
            self.word_embedding_model = models.Transformer(word_embedding_model, max_seq_length=max_seq_length)
            self._add_special_characters(special_characters)
            # pooling layer
            self.pooling_layer = models.Pooling(self.word_embedding_model.get_word_embedding_dimension(),
                                                pooling_mode_mean_tokens=pooling_mode == 'mean',
                                                pooling_mode_cls_token=pooling_mode == 'cls',
                                                pooling_mode_max_tokens=pooling_mode == 'max')

            self._sbert_model = SentenceTransformer(modules=[self.word_embedding_model, self.pooling_layer], device=device)

    def _add_special_characters(self, special_characters: Optional[List[str]]):
        if special_characters is not None and len(special_characters) > 0:
            self.word_embedding_model.tokenizer.add_tokens(special_characters, special_tokens=True)
            self.word_embedding_model.auto_model.resize_token_embeddings(len(self.word_embedding_model.tokenizer))
            self.special_tokens = self.word_embedding_model.tokenizer.get_added_vocab()
            logging.info(f"Added {len(self.special_tokens)} special tokens!")

    def save(self, model_save_path: Union[Path, str]) -> Path:
        model_save_path = Path(model_save_path)

        if not model_save_path.exists():
            model_save_path.mkdir(parents=True)

        self._sbert_model.save(path=str(model_save_path), model_name='nlfoa', create_model_card=True)
        logging.info(f"Persisted NLFOA model at {model_save_path}")
        return model_save_path

    def to_train(self):
        self._sbert_model.train()

    def to_eval(self):
        self._sbert_model.eval()

    def to_device(self, dev: str):
        assert "cuda" in dev or "cpu" in dev
        self._sbert_model.to(dev)

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
