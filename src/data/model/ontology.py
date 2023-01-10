
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from data.model.concept import Concept
from data.model.property import Property

ONTOLOGY_FILE_EXT = ".pkl"


@dataclass
class Ontology:
    input_file: Union[str, Path]
    concepts: Dict[str, Concept] = field(default_factory=dict)
    properties: Dict[str, Property] = field(default_factory=dict)
    # TODO add instances?
    instances: Dict[str, Property] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return Path(self.input_file).stem

    def get_concept_by_name(self, name: str) -> Optional[Concept]:
        for c in self.concepts.values():
            if c.name == name:
                return c

    def get_property_by_name(self, name: str) -> Optional[Property]:
        for p in self.properties.values():
            if p.name == name:
                return p

    def get_concept_by_name_prefix(self, name_prefix: str) -> List[Concept]:
        return list(c for c in self.concepts.values() if c.name.startswith(name_prefix))

    def get_property_by_name_prefix(self, name_prefix: str) -> List[Property]:
        return list(p for p in self.properties.values() if p.name.startswith(name_prefix))

    def get_concept_by_uri(self, uri: str) -> Optional[Concept]:
        return self.concepts[uri]

    def get_property_by_uri(self, uri: str) -> Optional[Property]:
        return self.properties[uri]

    def persist(self, output_dir: Union[str, Path]) -> Path:
        output_dir = Path(output_dir)
        if not (output_dir.exists() and output_dir.is_dir()):
            raise FileNotFoundError(f"Cannot read directory {output_dir}")
        fn = output_dir.joinpath(f"{self.name}{ONTOLOGY_FILE_EXT}")
        with open(fn, "wb") as f:
            pickle.dump(self, f)
        logging.info(f"Persisted Ontology {self.name} at {fn}")
        return fn

    @classmethod
    def load_from_file(cls, fn: Union[str, Path]) -> "Ontology":
        fn = Path(fn)
        if not (fn.exists() and fn.is_file()):
            raise FileNotFoundError(f"Cannot read file {fn}")
        with open(fn, "rb") as f:
            onto = pickle.load(f)
        logging.info(f"Loaded Ontology {onto.name} from file {fn}")
        return onto

    @classmethod
    def load_from_directory(cls, dir: Union[str, Path], onto_name: str) -> "Ontology":
        dir = Path(dir)
        if not (dir.exists() and dir.is_dir()):
            raise FileNotFoundError(f"Cannot read directory {dir}")
        # remove file ext if user added accidentaly
        fn = dir.joinpath(f"{onto_name.replace(ONTOLOGY_FILE_EXT, '')}{ONTOLOGY_FILE_EXT}")
        return cls.load_from_file(fn)

    def __len__(self) -> int:
        return len(self.concepts) + len(self.properties) + len(self.instances)

    def __str__(self) -> str:
        return (
            f"Ontology('input_file': {str(self.input_file)}, 'concepts': {len(self.concepts)}, "
            f"'properties': {len(self.properties)}, 'instances': {len(self.instances)}) "
        )

    def __repr__(self) -> str:
        return self.__str__()

    def __contains__(self, uri: str) -> bool:
        return uri in self.concepts or uri in self.properties

    def __getitem__(self, uri: str) -> Union[Concept, Property]:
        if uri in self.concepts:
            return self.concepts[uri]
        elif uri in self.properties:
            return self.properties[uri]
        else:
            msg = f"URI {str(uri)} is not part of {self.name}"
            logging.error(msg)
            raise KeyError(msg)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Ontology):
            return False
        return (self.concepts == o.concepts and
                self.properties == o.properties)
