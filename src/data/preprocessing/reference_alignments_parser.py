import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Union

from rdflib import Graph
from rdflib.term import Literal, URIRef

from data.model.alignment import Alignment
from data.model.reference_alignments import ReferenceAlignments
from data.model.ontology import Ontology


class IReferenceAlignmentsParser(ABC):

    @abstractmethod
    def parse_from_file(self,
                        input_file: Union[str, Path],
                        onto_a: Ontology,
                        onto_b: Ontology) -> ReferenceAlignments:
        pass


class RDFLibAlignmentFormatReferenceAlignmentsParser(IReferenceAlignmentsParser):
    _g: Graph

    def __init__(self) -> None:
        super().__init__()
        # https://moex.gitlabpages.inria.fr/alignapi/format.html
        self._alignment_format_ns = "http://knowledgeweb.semanticweb.org/heterogeneity/alignment#"
        self._alignment_format_relations: Dict[str, URIRef] = self._build_alignment_format_relations()

    def _build_alignment_format_relations(self) -> Dict[str, URIRef]:
        return {
            "alignment_map": URIRef(f"{self._alignment_format_ns}map"),
            "alignment_entity1": URIRef(f"{self._alignment_format_ns}entity1"),
            "alignment_entity2": URIRef(f"{self._alignment_format_ns}entity2"),
            "alignment_measure": URIRef(f"{self._alignment_format_ns}measure"),
            "alignment_relation": URIRef(f"{self._alignment_format_ns}relation")
        }

    def parse_from_file(self,
                        input_file: Union[str, Path],
                        onto_a: Ontology,
                        onto_b: Ontology) -> ReferenceAlignments:
        input_file = Path(input_file)
        assert input_file.exists() and input_file.is_file(), f"Cannot read Reference Alignments from: {input_file}"

        logging.info(f"Parsing Reference Alignments from {input_file} ...")
        start_parsing = time.time_ns()
        try:
            start = time.time_ns()
            self._g = Graph().parse(str(input_file))
            logging.debug(f"RDFLib Parsing took: {(time.time_ns() - start) * 1e-9} seconds")
            assert self._is_alginment_format_file(), ("This parser can only parse Reference Alignments "
                                                      "in Alignment Format (https://moex.gitlabpages.inria.fr/alignapi/format.html)")

            alis = self._get_alignments(onto_a, onto_b)

            ali = ReferenceAlignments(onto_a=onto_a, onto_b=onto_b, alignments=alis)
            logging.debug(f"Parsing Ontology took: {(time.time_ns() - start_parsing) * 1e-9} seconds")

        except Exception as e:
            msg = f"Cannot parse Reference Alignments {input_file} because:\n{e}"
            logging.error(msg)
            raise SystemExit(msg)

        return ali

    def _remove_hashtag_from_namespace(self) -> None:
        self._alignment_format_ns = self._alignment_format_ns.replace('#', '')
        self._alignment_format_relations = self._build_alignment_format_relations()

    def _is_alginment_format_file(self) -> bool:
        for (_, ns) in self._g.namespaces():
            normalized_ns = str(ns).lower()
            # some files have a trainling '#' at the namespace some do not ...
            if normalized_ns.startswith(self._alignment_format_ns[:-1]):
                if normalized_ns[-1] != '#' and normalized_ns[-1] == 't':
                    self._remove_hashtag_from_namespace()
                return True
        return False

    def _get_alignments(self, onto_a: Ontology, onto_b: Ontology) -> List[Alignment]:
        alis = []

        for alialignmet_map in self._g.objects(None, self._alignment_format_relations["alignment_map"]):
            onto_a_uri = str(list(self._g.objects(alialignmet_map, self._alignment_format_relations["alignment_entity1"]))[0])  # type: ignore
            onto_b_uri = str(list(self._g.objects(alialignmet_map, self._alignment_format_relations["alignment_entity2"]))[0])  # type: ignore
            m = float(list(self._g.objects(alialignmet_map, self._alignment_format_relations["alignment_measure"]))[0])  # type: ignore
            r: Literal = list(self._g.objects(alialignmet_map, self._alignment_format_relations["alignment_relation"]))[0]  # type: ignore
            
            ali = Alignment(onto_a_uri, onto_b_uri, m)

            if not r.toPython() == "=":
                logging.warning(f"Only equality relations '=' are allowed! {ali} will be ignored!")
            elif onto_a_uri not in onto_a:
                logging.warning(f"{onto_a_uri} is not part of {onto_a.name}! {ali} will be ignored!")
            elif onto_b_uri not in onto_b:
                logging.warning(f"{onto_b_uri} is not part of {onto_b.name}! {ali} will be ignored!")
            else:
                alis.append(ali)

        if not len(alis) > 0:
            msg = "This ReferenceAlignment File contains no alignments!"
            logging.error(msg)
            raise SystemError(msg)

        return alis
