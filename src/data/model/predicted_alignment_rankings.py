import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union, Tuple

import pandas as pd
import numpy as np
from torch import Tensor
from data.model.ontology import Ontology
from tqdm import tqdm

PREDICTED_ALIGNMENT_RANKINGS_FILE_EXTENSION = "_predicted_alignment_rankings.parquet.df"


@dataclass(frozen=True)
class PredictedAlignmentRankings:
    onto_a: Ontology
    onto_b: Ontology

    a_uris: List[str]  # ordering must prevail!
    b_uris: List[str]  # ordering must prevail!
    similarity_scores: np.ndarray  # len(a_uris) x len(b_uris)

    _desc_sorted_sim_scores_a_b: np.ndarray = field(init=False)  # len(a_uris) x len(b_uris)
    _desc_sorted_uri_indices_a_b: np.ndarray = field(init=False)  # len(a_uris) x len(b_uris)
    _desc_sorted_sim_scores_b_a: np.ndarray = field(init=False)  # len(b_uris) x len(a_uris)
    _desc_sorted_uri_indices_b_a: np.ndarray = field(init=False)  # len(b_uris) x len(a_uris)

    def __post_init__(self) -> None:
        for uri in self.a_uris:
            assert uri in self.onto_a, f"{uri} is not part of {self.onto_a.name}!"

        for uri in self.b_uris:
            assert uri in self.onto_b, f"{uri} is not part of {self.onto_b.name}!"

        assert (self.similarity_scores is not None and
                self.similarity_scores.shape == (len(self.a_uris), len(self.b_uris))), \
            "Similarity Tensor shape does not match the number of URI combinations!"

        if isinstance(self.similarity_scores, Tensor):
            object.__setattr__(self, 'similarity_scores', self.similarity_scores.numpy())

        _desc_sorted_sim_scores_a_b = np.sort(self.similarity_scores, axis=1)[:, ::-1]  # len(a_uris) x len(b_uris)
        _desc_sorted_uri_indices_a_b = np.argsort(self.similarity_scores, axis=1)[:, ::-1]  # len(a_uris) x len(b_uris)
        _desc_sorted_sim_scores_b_a = np.sort(self.similarity_scores.T, axis=1)[:, ::-1]  # len(b_uris) x len(a_uris)
        _desc_sorted_uri_indices_b_a = np.argsort(self.similarity_scores.T, axis=1)[:, ::-1]  # len(b_uris) x len(a_uris)

        object.__setattr__(self, '_desc_sorted_sim_scores_a_b', _desc_sorted_sim_scores_a_b)
        object.__setattr__(self, '_desc_sorted_uri_indices_a_b', _desc_sorted_uri_indices_a_b)
        object.__setattr__(self, '_desc_sorted_sim_scores_b_a', _desc_sorted_sim_scores_b_a)
        object.__setattr__(self, '_desc_sorted_uri_indices_b_a', _desc_sorted_uri_indices_b_a)

    def __str__(self) -> str:
        return f"{(len(self.a_uris), len(self.b_uris))} PredictedAlignmentRankings between {self.onto_a.name} and {self.onto_b.name}"

    def __repr__(self) -> str:
        return self.__str__()

    def get_sorted_rankings_for_uri(self, uri: str) -> List[Tuple[str, float]]:
        if not isinstance(uri, str):
            raise ValueError("Unsupported type for URI! Only str is allowed!")
        elif uri not in self.a_uris and uri not in self.b_uris:
            msg = f"{uri} is not contained in this Predicted Alignment Rankings!"
            logging.error(msg)
            raise KeyError(msg)

        if uri in self.onto_a:
            uris = self.b_uris
            target_uri_idx = self.a_uris.index(uri)
            sorted_scores = self._desc_sorted_sim_scores_a_b[target_uri_idx]  # len(b_uris)
            sorted_indices = self._desc_sorted_uri_indices_a_b[target_uri_idx]  # len(b_uris)
        elif uri in self.onto_b:
            uris = self.a_uris
            target_uri_idx = self.b_uris.index(uri)
            sorted_scores = self._desc_sorted_sim_scores_b_a[target_uri_idx]  # len(a_uris)
            sorted_indices = self._desc_sorted_uri_indices_b_a[target_uri_idx]  # len(a_uris)
        else:
            msg = f"{uri} is neither part of Ontology {self.onto_a.name} nor Ontology {self.onto_b.name}!"
            logging.error(msg)
            raise KeyError(msg)

        return [(uris[uri_idx], score) for score, uri_idx in zip(sorted_scores, sorted_indices)]

    def to_dataframe(self, output_dir: Optional[Union[str, Path]] = None) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Path]]:
        # rows: ranked onto_b URI
        # cols: (onto_a_uri, score)
        data = {
            onto_a_uri: [(b_uri, score) for b_uri, score in self.get_sorted_rankings_for_uri(onto_a_uri)]
            for onto_a_uri in tqdm(self.a_uris, desc="Generating row data")
        }
        logging.debug(f"Building DataFrame with MultiIndex for {len(data)} rows...")
        df = pd.DataFrame.from_dict(data)
        # Split (onto_a_uri, score) tuples into hierarchical cols
        # https://stackoverflow.com/questions/44675679/convert-pandas-column-of-tuples-to-multiindex
        df = pd.concat([pd.DataFrame(x, columns=['URI', 'Similarity Score'])
                        for x in df.values.T.tolist()], axis=1, keys=df.columns)
        df.index.name = "Rank"

        if output_dir is not None:
            output_dir = Path(output_dir)
            if not output_dir.exists():
                raise FileNotFoundError(f"Cannot read {output_dir}")
            fn = f"{self.onto_a.name}-{self.onto_b.name}{PREDICTED_ALIGNMENT_RANKINGS_FILE_EXTENSION}"
            fn = output_dir.joinpath(fn)
            df.to_parquet(fn)
            logging.info(f"Persisted {self} at: {fn}")
            return df, fn

        return df
