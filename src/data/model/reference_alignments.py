import logging
from dataclasses import dataclass, field
from typing import Dict, List, Union, Optional, Tuple
from pathlib import Path
import pandas as pd

from data.model.alignment import Alignment
from data.model.ontology import Ontology

REFERENCE_ALIGNMENTS_FILE_EXTENSION = "_reference_alignments.parquet.df"

@dataclass(frozen=True)
class ReferenceAlignments:
    onto_a: Ontology
    onto_b: Ontology

    alignments: List[Alignment]

    _a2b_alignments: Dict[str, str] = field(default_factory=dict)
    _b2a_alignments: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        a2b = dict()
        b2a = dict()
        for ali in self.alignments:
            assert ali.a_uri in self.onto_a, f"{ali.a_uri} is not part of {self.onto_a.name}!"
            assert ali.b_uri in self.onto_b, f"{ali.b_uri} is not part of {self.onto_b.name}!"
            a2b[ali.a_uri] = ali.b_uri
            b2a[ali.b_uri] = ali.a_uri

        # https://stackoverflow.com/questions/53756788/how-to-set-the-value-of-dataclass-field-in-post-init-when-frozen-true
        object.__setattr__(self, '_a2b_alignments', a2b)
        object.__setattr__(self, '_b2a_alignments', b2a)

    def to_dataframe(self, output_dir: Optional[Union[str, Path]] = None) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Path]]:
        df = pd.DataFrame.from_records(list(map(lambda a: (a.a_uri, a.b_uri, a.similarity), self.alignments)),
                                       columns=["a_uri", "b_uri", "similarity"])
        if output_dir is not None:
            output_dir = Path(output_dir)
            if not output_dir.exists():
                raise FileNotFoundError(f"Cannot read {output_dir}")
            fn = f"{self.onto_a.name}-{self.onto_a.name}{REFERENCE_ALIGNMENTS_FILE_EXTENSION}"
            fn = output_dir.joinpath(fn)
            df.to_parquet(fn)
            logging.info(f"Persisted {self} at: {fn}")
            return df, fn

        return df

    def __len__(self) -> int:
        return len(self.alignments)

    def __str__(self) -> str:
        return f"{len(self)} Reference Alignments between {self.onto_a.name} and {self.onto_b.name}"

    def __repr__(self) -> str:
        return self.__str__()

    def __getitem__(self, uri: str) -> str:
        if uri in self._a2b_alignments:
            return self._a2b_alignments[uri]
        elif uri in self._b2a_alignments:
            return self._b2a_alignments[uri]
        else:
            msg = f"{uri} is not in Reference Alignments"
            logging.error(msg)
            raise KeyError(msg)

    def __contains__(self, uri: str) -> bool:
        if uri in self._a2b_alignments or uri in self._b2a_alignments:
            return True
        return False
