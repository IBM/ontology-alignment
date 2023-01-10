from torch.utils.data import Dataset
from typing import List, Union
from data.model.alignment import Alignment
from sentence_transformers.readers import InputExample
from data.model.ontology import Ontology
from data.model.concept import Concept
from data.model.property import Property
from omegaconf import DictConfig
from data.preprocessing.pseudo_sentence_generator import PseudoSentenceGenerator
import random


class DynamicPseudoSentenceAlignmentDataset(Dataset):
    def __init__(self,
                 ontologies: List[Ontology],
                 alignments: List[Alignment],
                 psg_config: DictConfig,
                 shuffle_ps: bool) -> None:
        super().__init__()

        self.ontologies = ontologies
        self.alignments = alignments
        self.psg = PseudoSentenceGenerator(psg_config)
        self.shuffle_ps = shuffle_ps

    def _get_concept_or_property(self, uri: str) -> Union[Concept, Property]:
        for onto in self.ontologies:
            if uri in onto:
                return onto[uri]
        raise KeyError(f"Cannot find {uri} in any of the Ontologies: {[onto.name for onto in self.ontologies]}")

    def __len__(self) -> int:
        return len(self.alignments)

    def __getitem__(self, idx: int) -> InputExample:
        ali = self.alignments[idx]

        entity_a = self._get_concept_or_property(ali.a_uri)
        entity_b = self._get_concept_or_property(ali.b_uri)

        if self.shuffle_ps:
            random.shuffle(self.psg.config.concepts.ordering)
            random.shuffle(self.psg.config.properties.ordering)

        if isinstance(entity_a, Concept):
            entity_a_ps = self.psg.generate_for_concept(entity_a)
        elif isinstance(entity_a, Property):
            entity_a_ps = self.psg.generate_for_property(entity_a)
        else:
            raise NotImplementedError("This never should have happened!")
        
        if isinstance(entity_b, Concept):
            entity_b_ps = self.psg.generate_for_concept(entity_b)
        elif isinstance(entity_b, Property):
            entity_b_ps = self.psg.generate_for_property(entity_b)
        else:
            raise NotImplementedError("This never should have happened!")

        return InputExample(str(idx), [entity_a_ps, entity_b_ps], ali.similarity)
