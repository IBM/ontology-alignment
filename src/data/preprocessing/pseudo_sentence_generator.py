import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
from data.model.concept import Concept
from data.model.ontology import Ontology
from data.model.property import Property
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

PSEUDO_SENTENCE_CACHE_FILE_EXTENSION = "_pseudo_sentences.parquet.df"

default_config = OmegaConf.create({
    "concepts": {
        "name": {
            "camel_case_split": True,
            "snake_case_split": True,
            "to_lower": False
        },
        "comments": {
            "include": True
        },
        "labels": {
            "include": True
        },
        "parents": {
            "include_name": True,
            "include_props": False,
            "max_level": 1
        },
        "children": {
            "include_name": True,
            "include_props": False,
            "max_level": 1
        },
        "props": {
            "incoming": {
                "include_name": True,
                "include_domain": True,
                "include_range": False
            },
            "outgoing": {
                "include_name": True,
                "include_domain": False,
                "include_range": True
            }
        },
        "ordering": [
            "name",
            "labels",
            "comments",
            "parents",
            "children",
            {
                "props": [
                    'incoming',
                    'outgoing'
                ]
            }
        ],
    },
    "properties": {
        "name": {
            "camel_case_split": True,
            "snake_case_split": True,
            "to_lower": False
        },
        "comments": {
            "include": True
        },
        "labels": {
            "include": True
        },
        "domain": {
            "include_name": True,
            "include_parents": False,
            "include_children": False
        },
        "range": {
            "include_name": True,
            "include_parents": False,
            "include_children": False
        },
        "children": {
            "include_name": True,
            "include_domain": False,
            "include_range": False,
            "max_level": 1
        },
        "parents": {
            "include_name": True,
            "include_domain": False,
            "include_range": False,
            "max_level": 1
        },
        "inverse": {
            "include_name": True,
            "include_domain": False,
            "include_range": False
        },
        "ordering": [
            "name",
            "labels",
            "comments",
            "domain",
            "range",
            "inverse",
            "parents",
            "children"
        ],
    },
    "special_characters": {
        # concept and property related
        "list_separator": "[|]",
        "name_start": "[NME]",
        "name_end": "[/NME]",
        "labels_start": "[LBL]",
        "labels_end": "[/LBL]",
        "comments_start": "[CMT]",
        "comments_end": "[/CMT]",
        "no_parents": "[NOP]",
        "no_children": "[NOC]",
        "properties_domain_start": "[PD]",
        "properties_domain_end": "[/PD]",
        "properties_range_start": "[PR]",
        "properties_range_end": "[/PR]",
        # only concept related
        "target_concept_start": "[TC]",
        "target_concept_end": "[/TC]",
        "parent_concepts_start": "[PC]",
        "parent_concepts_end": "[/PC]",
        "child_concepts_start": "[CC]",
        "child_concepts_end": "[/CC]",
        "incoming_properties_start": "[IP]",
        "incoming_properties_end": "[/IP]",
        "outgoing_properties_start": "[OP]",
        "outgoing_properties_end": "[/OP]",
        # only property realated
        "target_property_start": "[TPR]",
        "target_property_end": "[/TPR]",
        "parent_properties_start": "[PPR]",
        "parent_properties_end": "[/PPR]",
        "child_properties_start": "[CPR]",
        "child_properties_end": "[/CPR]",
        "inverse_properties_start": "[IPR]",
        "inverse_properties_end": "[/IPR]",
        "no_inverse": "[NOI]"
    }
})


def snake_case_split(input: str) -> List[str]:
    return input.split("_")


def camel_case_split(input: str) -> List[str]:
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', input)
    return [m.group(0) for m in matches]


class PseudoSentenceGenerator:
    def __init__(self, config: Optional[DictConfig] = None) -> None:
        if not config:
            self.config = default_config
        else:
            self.config = config

        # to check if two PSGs are equal (e.g., to check if a PS Cache was generated with the same config as another PS Cache)
        self._cache_headers = ["uri", "pseudo_sentence", "ontology_file", "psg_config_hash"]
        self.config_hash = self._generate_config_hash()

        # to enable ordering of string components from the config
        self._generate_concept_string: Dict[str, Callable[[Union[Concept, Property]], str]] = {
            "name": self._generate_name_string,
            "labels": self._generate_labels_string,
            "comments": self._generate_comments_string,
            "parents": self._generate_concept_parents_string,
            "children": self._generate_concept_children_string,
            "props": self._generate_concept_properties_string,
            "incoming": self._generate_concept_incoming_props_string,
            "outgoing": self._generate_concept_outgoing_props_string
        }

        # to enable ordering of string components from the config
        self._generate_property_string: Dict[str, Callable[[Union[Property, Concept]], str]] = {
            "name": self._generate_name_string,
            "labels": self._generate_labels_string,
            "comments": self._generate_comments_string,
            "domain": self._generate_property_domain_string,
            "range": self._generate_property_range_string,
            "inverse": self._generate_property_inverse_string,
            "parents": self._generate_property_parents_string,
            "children": self._generate_property_children_string
        }

    def _generate_psc_filename(self, onto: Ontology) -> str:
        return f"{onto.name}_{self.config_hash}{PSEUDO_SENTENCE_CACHE_FILE_EXTENSION}"

    def generate_for_ontology(self, onto: Ontology) -> Dict[str, str]:
        logging.info((f"Generating Pseudo Sentences for {onto.name} with "
                      f"{len(onto.concepts)} concepts and {len(onto.properties)} properties... "))
        ps_concepts = self.generate_for_all_concepts(onto)
        ps_props = self.generate_for_all_properties(onto)
        return {**ps_concepts, **ps_props}

    def build_psg_cache(self,
                        outdir: Union[str, Path],
                        onto: Ontology,
                        persist: bool = True) -> Union[pd.DataFrame, Tuple[Path, pd.DataFrame]]:

        pseudo_sents = self.generate_for_ontology(onto)

        data = {cache_header: [] for cache_header in self._cache_headers}
        for uri, ps in pseudo_sents.items():
            data["uri"].append(str(uri))
            data["pseudo_sentence"].append(ps)
            data["ontology_file"].append(str(onto.input_file))
            data["psg_config_hash"].append(self.config_hash)

        cache = pd.DataFrame.from_dict(data)
        self._validate_psg_cache(cache)

        if persist:
            outdir = Path(outdir)
            if not outdir.exists():
                outdir.mkdir(parents=True)

            psc_filename = self._generate_psc_filename(onto)
            out_path = outdir.joinpath(psc_filename)
            cache.reset_index(drop=True).to_feather(str(out_path))

            logging.info(f"Persisted Pseudo Sentence Cache at {out_path.resolve()}")

            return out_path, cache
        else:
            return cache

    def read_psg_cache_of_onto(self, psc_dir: Union[str, Path], onto: Ontology) -> Optional[pd.DataFrame]:
        psc_dir = Path(psc_dir)
        psc_filename = self._generate_psc_filename(onto)
        return self.read_psg_cache(psc_dir.joinpath(psc_filename))

    def read_psg_cache(self, cache_file: Union[str, Path]) -> Optional[pd.DataFrame]:
        if Path(cache_file).exists():
            cache = pd.read_feather(str(cache_file))
            self._validate_psg_cache(cache)
            return cache
        return None

    def _generate_config_hash(self) -> str:
        config_str = OmegaConf.to_yaml(self.config, sort_keys=True)
        return hashlib.sha256(bytes(config_str, encoding="utf-8")).hexdigest()

    def _validate_psg_cache(self, cache: pd.DataFrame) -> None:
        err = "This Pseudo Sentence Cache instance is invalid or is not compatible with this PSG instance!"
        assert all([cache_header in cache.columns for cache_header in self._cache_headers]), err
        assert len(cache["psg_config_hash"].unique()) == 1, err
        ch = cache["psg_config_hash"].iloc[0]
        assert ch == self.config_hash, err

    def _generate_name_string(self, c_or_p: Union[Concept, Property]) -> str:
        if self.config.concepts.name.include:
            return (
                f"{self.config.special_characters.name_start}"
                f" {self._apply_name_transformations(c_or_p.name)} "
                f"{self.config.special_characters.name_end} "
            )
        else:
            return ""

    def _apply_name_transformations(self, name: str) -> str:
        splitted = ""
        if self.config.concepts.name.camel_case_split and self.config.concepts.name.snake_case_split:
            for snake in snake_case_split(name):
                for camel in camel_case_split(snake):
                    splitted += f"{camel} "
        elif self.config.concepts.name.camel_case_split:
            for camel in camel_case_split(name):
                splitted += f"{camel} "
        elif self.config.concepts.name.snake_case_split:
            for snake in snake_case_split(name):
                splitted += f"{snake} "

        if self.config.concepts.name.to_lower:
            return splitted.lower().strip()

        return splitted.strip()

    def _generate_comments_string(self, c_or_p: Union[Concept, Property]) -> str:
        return self._generate_comment_or_labels_string(c_or_p, comment=True)

    def _generate_labels_string(self, c_or_p: Union[Concept, Property]) -> str:
        return self._generate_comment_or_labels_string(c_or_p, comment=False)

    def _generate_comment_or_labels_string(self, c_or_p: Union[Concept, Property], comment: bool = True) -> str:
        if comment:
            ls_or_cs = c_or_p.comments
            if isinstance(c_or_p, Concept):
                include = self.config.concepts.comments.include
            elif isinstance(c_or_p, Property):
                include = self.config.properties.comments.include
        else:
            ls_or_cs = c_or_p.labels
            if isinstance(c_or_p, Concept):
                include = self.config.concepts.labels.include
            elif isinstance(c_or_p, Property):
                include = self.config.properties.labels.include

        if not include or len(ls_or_cs) == 0:
            return ""

        ls = f"{self.config.special_characters.comments_start if comment else self.config.special_characters.labels_start}"
        for i, l_or_c in enumerate(ls_or_cs):
            ls += f" {str(l_or_c).strip()}"
            if i < len(ls_or_cs) - 1:
                ls += self.config.special_characters.list_separator
        ls += f"{self.config.special_characters.comments_end if comment else self.config.special_characters.labels_end} "

        return ls

    """
    Concept Related String Generation Methods
    """

    def generate_for_all_concepts(self, onto: Ontology) -> Dict[str, str]:
        return {uri: self.generate_for_concept(c) for uri, c in tqdm(onto.concepts.items(), total=len(onto.concepts))}

    def generate_for_concept(self, concept: Concept) -> str:
        if concept is None:
            raise ValueError("Concept must not be None!")
        ps = str(self.config.special_characters.target_concept_start)
        for part in self.config.concepts.ordering:
            # subordering i.e. nested list is wrapped in dict
            if type(part) != str:
                subpart = str(next(iter(part.keys())))
                ps += self._generate_concept_string[subpart](concept)
            else:
                ps += self._generate_concept_string[part](concept)

        ps = ps.strip()
        ps += str(self.config.special_characters.target_concept_end)
        return ps

    def _generate_concept_parents_string(self, concept: Concept, current_level: int = 0) -> str:
        if current_level >= self.config.concepts.parents.max_level:
            return ""

        icn = self.config.concepts.parents.include_name
        icp = self.config.concepts.parents.include_props
        if not any([icn, icp]):
            return ""

        ps = f"{self.config.special_characters.parent_concepts_start}"
        if len(concept.parents) == 0:
            ps += f" {self.config.special_characters.no_parents} "
        else:
            for i, parent in enumerate(concept.parents.values()):
                if icn:
                    ps += f" {self._apply_name_transformations(parent.name)} "
                if icp:
                    ps += self._generate_concept_properties_string(parent)

                ps += self._generate_concept_parents_string(parent, current_level + 1)

                if i < len(concept.parents) - 1:
                    ps += self.config.special_characters.list_separator

        ps += f"{self.config.special_characters.parent_concepts_end} "

        return ps

    def _generate_concept_children_string(self, concept: Concept, current_level: int = 0) -> str:
        if current_level >= self.config.concepts.children.max_level:
            return ""

        icn = self.config.concepts.children.include_name
        icp = self.config.concepts.children.include_props
        if not any([icn, icp]):
            return ""

        cs = f"{self.config.special_characters.child_concepts_start}"
        if len(concept.children) == 0:
            cs += f" {self.config.special_characters.no_children} "
        else:
            for i, child in enumerate(concept.children.values()):
                if icn:
                    cs += f" {self._apply_name_transformations(child.name)} "
                if icp:
                    cs += self._generate_concept_properties_string(child)

                cs += self._generate_concept_children_string(child, current_level + 1)

                if i < len(concept.children) - 1:
                    cs += self.config.special_characters.list_separator

        cs += f"{self.config.special_characters.child_concepts_end} "

        return cs

    def _generate_concept_properties_string(self, concept: Concept) -> str:
        ps = ""
        # since config.concepts.ordering is a ListConfig we cannot access the prop ordering directly
        # but have to linearly search it
        for part in self.config.concepts.ordering:
            # nested list is wrapped in DictConfig
            if not type(part) == str:
                if str(next(iter(part.keys()))) == "props":
                    for prop in part.props:
                        ps += self._generate_concept_string[prop](concept)

        return ps

    def _generate_domain_string(self, domain: List[Concept]) -> str:
        ds = ""
        ds += f"{self.config.special_characters.properties_domain_start}"
        for j, d in enumerate(domain):
            ds += f" {self._apply_name_transformations(d.name)} "
            if j < len(domain) - 1:
                ds += self.config.special_characters.list_separator
        ds += f"{self.config.special_characters.properties_domain_end}"
        return ds

    def _generate_range_string(self, range_: List[Union[Concept, Any]]) -> str:
        rs = ""
        rs += f"{self.config.special_characters.properties_range_start}"
        for j, r in enumerate(range_):
            if isinstance(r, Concept):
                rs += f" {self._apply_name_transformations(r.name)} "
            elif isinstance(r, type):
                rs += f" {self._apply_name_transformations(r.__name__)} "
            else:
                rs += f" {self._apply_name_transformations(str(r))} "
            if j < len(range_) - 1:
                rs += self.config.special_characters.list_separator
        rs += f"{self.config.special_characters.properties_range_end}"
        return rs

    def _generate_properties_string(self,
                                    props: List[Property],
                                    include_name: bool,
                                    include_range: bool,
                                    include_domain: bool) -> str:
        ps = ""
        for i, p in enumerate(props):
            if include_name:
                ps += f" {self._apply_name_transformations(p.name)} "
            if include_domain:
                ps += self._generate_domain_string(p.domain)
            if include_range:
                ps += self._generate_range_string(p.range)

            if i < len(props) - 1:
                ps += self.config.special_characters.list_separator

        return ps

    def _generate_concept_incoming_props_string(self, concept: Concept) -> str:
        icn = self.config.concepts.props.incoming.include_name
        icr = self.config.concepts.props.incoming.include_range
        icd = self.config.concepts.props.incoming.include_domain

        if not any([icn, icr, icd]):
            return ""

        inps = f"{self.config.special_characters.incoming_properties_start}"
        inps += self._generate_properties_string(list(concept.incoming_properties.values()), icn, icr, icd)
        inps += f"{self.config.special_characters.incoming_properties_end} "
        return inps

    def _generate_concept_outgoing_props_string(self, concept: Concept) -> str:
        icn = self.config.concepts.props.outgoing.include_name
        icr = self.config.concepts.props.outgoing.include_range
        icd = self.config.concepts.props.outgoing.include_domain

        if not any([icn, icr, icd]):
            return ""

        ops = f"{self.config.special_characters.outgoing_properties_start} "
        ops += self._generate_properties_string(list(concept.outgoing_properties.values()), icn, icr, icd)
        ops += f"{self.config.special_characters.outgoing_properties_end} "
        return ops

    """
    Property Related String Generation Methods
    """

    def generate_for_all_properties(self, onto: Ontology) -> Dict[str, str]:
        return {uri: self.generate_for_property(p) for uri, p in tqdm(onto.properties.items(), total=len(onto.properties))}

    def generate_for_property(self, prop: Property) -> str:
        if prop is None:
            raise ValueError("Property must not be None!")
        ps = ""
        for part in self.config.properties.ordering:
            # subordering i.e. nested list is wrapped in dict
            if type(part) != str:
                subpart = str(next(iter(part.keys())))
                ps += self._generate_property_string[subpart](prop)
            else:
                ps += self._generate_property_string[part](prop)

        return ps.strip()

    def _generate_property_domain_string(self, prop: Property) -> str:
        return self._generate_domain_string(prop.domain)

    def _generate_property_range_string(self, prop: Property) -> str:
        return self._generate_range_string(prop.range)

    def _generate_property_inverse_string(self, prop: Property) -> str:
        icn = self.config.properties.inverse.include_name
        icd = self.config.properties.inverse.include_domain
        icr = self.config.properties.inverse.include_range
        if not any([icn, icr, icd]):
            return ""

        ivs = f"{self.config.special_characters.inverse_properties_start}"
        if prop.inverse:
            ivs += self._generate_properties_string([prop.inverse], icn, icr, icd)
        else:
            ivs += self.config.special_characters.no_inverse
        ivs += f"{self.config.special_characters.inverse_properties_end} "
        return ivs

    def _generate_property_parents_string(self, prop: Property, current_level: int = 0) -> str:
        if current_level >= self.config.properties.parents.max_level:
            return ""

        icn = self.config.properties.parents.include_name
        icd = self.config.properties.parents.include_domain
        icr = self.config.properties.parents.include_range
        if not any([icn, icr, icd]):
            return ""

        ps = f"{self.config.special_characters.parent_properties_start}"
        if len(prop.parents) == 0:
            ps += f" {self.config.special_characters.no_parents} "
        else:
            for i, parent in enumerate(prop.parents.values()):
                ps += self._generate_properties_string([parent], icn, icr, icd)

                ps += self._generate_property_parents_string(parent, current_level + 1)

                if i < len(prop.parents) - 1:
                    ps += self.config.special_characters.list_separator

        ps += f"{self.config.special_characters.parent_properties_end} "

        return ps

    def _generate_property_children_string(self, prop: Property, current_level: int = 0) -> str:
        if current_level >= self.config.properties.children.max_level:
            return ""

        icn = self.config.properties.children.include_name
        icd = self.config.properties.children.include_domain
        icr = self.config.properties.children.include_range
        if not any([icn, icr, icd]):
            return ""

        cs = f"{self.config.special_characters.child_properties_start}"
        if len(prop.children) == 0:
            cs += f" {self.config.special_characters.no_children} "
        else:
            for i, child in enumerate(prop.children.values()):
                cs += self._generate_properties_string([child], icn, icr, icd)

                cs += self._generate_property_parents_string(child, current_level + 1)

                if i < len(prop.children) - 1:
                    cs += self.config.special_characters.list_separator

        cs += f"{self.config.special_characters.child_properties_end} "

        return cs
