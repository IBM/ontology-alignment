import logging
from typing import List, Dict

from data.preprocessing.reference_alignments_parser import RDFLibAlignmentFormatReferenceAlignmentsParser
from data.preprocessing.owlready2_ontology_parser import OwlReady2OntologyParser
from pathlib import Path
from glob import glob
from tqdm import tqdm
import pprint

import pytest

logging.getLogger().setLevel(logging.INFO)

"""
# '=' relations found in the files with rg (ripgrep)
 rg "relation>=" -c
ekaw-sigkdd.rdf:11
conference-ekaw.rdf:25
confOf-sigkdd.rdf:7
conference-iasted.rdf:14
conference-confOf.rdf:15
cmt-ekaw.rdf:11
edas-ekaw.rdf:23
cmt-conference.rdf:15
conference-sigkdd.rdf:15
cmt-confOf.rdf:16
iasted-sigkdd.rdf:15
confOf-edas.rdf:19
confOf-iasted.rdf:9
cmt-sigkdd.rdf:12
cmt-iasted.rdf:4
edas-iasted.rdf:19
cmt-edas.rdf:13
conference-edas.rdf:17
edas-sigkdd.rdf:15
confOf-ekaw.rdf:20
ekaw-iasted.rdf:10
"""


@pytest.fixture
def conference_num_reference_alignment() -> Dict[str, int]:
    return {
        "ekaw-sigkdd": 11,
        "conference-ekaw": 25,
        "confOf-sigkdd": 7,
        "conference-iasted": 14,
        "conference-confOf": 15,
        "cmt-ekaw": 11,
        "edas-ekaw": 23,
        "cmt-conference": 15,
        "conference-sigkdd": 15,
        "cmt-confOf": 16,
        "iasted-sigkdd": 15,
        "confOf-edas": 19,
        "confOf-iasted": 9,
        "cmt-sigkdd": 12,
        "cmt-iasted": 4,
        "edas-iasted": 19,
        "cmt-edas": 13,
        "conference-edas": 17,
        "edas-sigkdd": 15,
        "confOf-ekaw": 20,
        "ekaw-iasted": 10
    }


@pytest.fixture
def conference_ontologies() -> Dict[str, Path]:
    dataset_dir = Path("datasets/oaei/2021/conference/ontologies")
    assert dataset_dir.exists(), f"Cannot find datasets at: {dataset_dir}"
    return {str(Path(onto_file).stem): Path(onto_file)
            for onto_file in glob(str(dataset_dir.joinpath("*.owl")))}


@pytest.fixture
def conference_reference_alignments() -> List[Path]:
    dataset_dir = Path("datasets/oaei/2021/conference/alignments")
    assert dataset_dir.exists(), f"Cannot find datasets at: {dataset_dir}"
    return [Path(ref_ali_file) for ref_ali_file in glob(str(dataset_dir.joinpath("*.rdf")))]


def test_highlevel_functionality_on_all_conference_reference_alignments(conference_reference_alignments: List[Path],
                                                                        conference_ontologies: Dict[str, Path],
                                                                        conference_num_reference_alignment: Dict[str, int]):
    logging.info(
        f"Testing RDFLibAlignmentFormatAlignmentsParser on all {len(conference_reference_alignments)} Conference Ontologies!")
    onto_parser = OwlReady2OntologyParser()
    ali_parser = RDFLibAlignmentFormatReferenceAlignmentsParser()

    onto_cache = dict()

    for ref_ali_path in tqdm(conference_reference_alignments):
        ontos_in_alignment = str(ref_ali_path.stem).split("-")
        oa = ontos_in_alignment[0]
        ob = ontos_in_alignment[1]

        if oa not in onto_cache:
            logging.info(f"Parsing ontology in reference alignments: {conference_ontologies[oa]}")
            onto_a = onto_parser.parse_from_file(conference_ontologies[oa])
            onto_cache[oa] = onto_a
        else:
            onto_a = onto_cache[oa]

        if ob not in onto_cache:
            logging.info(f"Parsing ontology in reference alignments: {conference_ontologies[ob]}")
            onto_b = onto_parser.parse_from_file(conference_ontologies[ob])
            onto_cache[ob] = onto_b
        else:
            onto_b = onto_cache[ob]

        logging.info(f"Parsing reference alignments: {ref_ali_path}")
        ref_alis = ali_parser.parse_from_file(ref_ali_path, onto_a, onto_b)

        logging.info(pprint.pformat(ref_alis))

        assert len(ref_alis) == conference_num_reference_alignment[str(ref_ali_path.stem)]
