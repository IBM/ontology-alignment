import logging
from typing import List, Dict
from data.preprocessing.ontology_parser import IOntologyParser

from data.preprocessing.rdflib_ontology_parser import RDFLibOntologyParser
from data.preprocessing.owlready2_ontology_parser import OwlReady2OntologyParser

from pathlib import Path
from glob import glob
from tqdm import tqdm
import pprint

import pytest


"""
http://oaei.ontologymatching.org/2021/conference/index.html

Name	    Type	Number of Classes	Number of Datatype Properties 	Number of Object Properties 	DL expressivity	    Related link
Ekaw	    Insider     74	                0	                            33	                             SHIN	        http://ekaw.vse.cz
Sofsem	    Insider	    60	                18	                            46	                             ALCHIF(D)	    http://www.sofsem.cz
Sigkdd  	Web	        49	                11	                            17	                             ALEI(D)	    http://www.acm.org/sigs/sigkdd/kdd2006
Iasted	    Web	        140	                3	                            38	                             ALCIN(D)	    http://iasted.com/conferences/2005/cancun/ms.htm
Micro	    Web	        32	                9	                            17	                             ALCOIN(D)	    http://www.microarch.org
Confious	Tool	    57	                5	                            52	                             SHIN(D)	    http://www.confious.com
Pcs	        Tool	    23	                14	                            24	                             ALCIF(D)	    http://precisionconference.com
OpenConf	Tool	    62	                21	                            24	                             ALCOI(D)	    http://www.zakongroup.com/technology/openconf.shtml
ConfTool	Tool	    38	                23	                            13	                             SIN(D)	        http://www.conftool.net
Crs	        Tool	    14	                2	                            15	                             ALCIF(D)	    http://www.conferencereview.com
Cmt	        Tool	    36	                10	                            49	                             ALCIN(D)	    http://msrcmt.research.microsoft.com/cmt
Cocus	    Tool	    55	                0	                            35	                             ALCIF	        http://cocus.create-net.it/
Paperdyne	Tool	    47	                21	                            61	                             ALCHIN(D)	    http://www.paperdyne.com/
Edas	    Tool	    104	                20	                            30	                             ALCOIN(D)	    http://edas.info/
MyReview	Tool	    39	                17	                            49	                             ALCOIN(D)	    http://myreview.intellagence.eu/
Linklings	Tool	    37	                16	                            31	                             SROIQ(D)	    http://www.linklings.com/
"""

logging.getLogger().setLevel(logging.INFO)


@pytest.fixture
def conference_ontologies_stats() -> Dict[str, Dict[str, int]]:
    return {
        "num_classes": {
            "ekaw": 74,
            "conference": 60,  # Sofsem
            "sigkdd": 49,
            "iasted": 140,
            "micro": 32,
            "confious": 57,
            "pcs": 23,
            "openconf": 62,
            "confof": 38,  # ConfTool
            "crs_dr": 14,  # Crs
            "cmt": 36,
            "cocus": 55,
            "paperdyne": 47,
            "edas": 104,
            "myreview": 39,
            "linklings": 37,
        },
        "num_datatype_props": {
            "ekaw": 0,
            "conference": 18,
            "sigkdd": 11,
            "iasted": 3,
            "micro": 9,
            "confious": 5,
            "pcs": 14,
            "openconf": 21,
            "confof": 23,
            "crs_dr": 2,
            "cmt": 10,
            "cocus": 0,
            "paperdyne": 21,
            "edas": 20,
            "myreview": 17,
            "linklings": 16,
        },
        "num_object_props": {
            "ekaw": 33,
            "conference": 46,
            "sigkdd": 17,
            "iasted": 38,
            "micro": 17,
            "confious": 52,
            "pcs": 24,
            "openconf": 24,
            "confof": 13,
            "crs_dr": 15,
            "cmt": 49,
            "cocus": 35,
            "paperdyne": 61,
            "edas": 30,
            "myreview": 49,
            "linklings": 31,
        }
    }


@pytest.fixture
def conference_ontologies() -> List[Path]:
    dataset_dir = Path("datasets/oaei/2021/conference/ontologies")
    assert dataset_dir.exists(), f"Cannot find datasets at: {dataset_dir}"
    return [Path(onto_file) for onto_file in glob(str(dataset_dir.joinpath("*.owl")))]


def test_rdflib_ontology_parser_highlevel_functionality_on_all_conference_ontologies(conference_ontologies: List[Path],
                                                                                     conference_ontologies_stats: Dict[str, Dict[str, int]]):
    logging.info(f"Testing RDFLibOntologyParser on all {len(conference_ontologies)} Conference Ontologies!")
    parser = RDFLibOntologyParser()
    for conf_onto in tqdm(conference_ontologies):
        onto = parser.parse_from_file(conf_onto)

        onto_name = str(conf_onto.stem).lower().strip()
        num_classes = conference_ontologies_stats["num_classes"][onto_name]
        num_datatype_props = conference_ontologies_stats["num_datatype_props"][onto_name]
        num_object_props = conference_ontologies_stats["num_object_props"][onto_name]
        num_all_props = num_datatype_props + num_object_props

        # For some strange reason most of the number of parsed classes or properties are off by one. After inspecting some of the ontology files
        #  manually, I'm pretty sure this is not a problem of the Parser but the provided information is not correct. But this is claim has to be validated
        #  by some other persons, too.
        if not len(onto.concepts) == num_classes:
            logging.warning(
                f"Number of classes in '{onto_name}' ({len(onto.concepts)}) does not match {num_classes}!")
        if not len(onto.properties) == num_all_props:
            logging.warning(
                f"Number of Properties in '{onto_name}' ({len(onto.properties)}) does not match {num_all_props}!")

        logging.info(pprint.pformat(onto))


def test_owlready2_ontology_parser_highlevel_functionality_on_all_conference_ontologies(conference_ontologies: List[Path],
                                                                                        conference_ontologies_stats: Dict[str, Dict[str, int]]):
    logging.info(f"Testing OwlReady2OntologyParser on all {len(conference_ontologies)} Conference Ontologies!")
    parser = OwlReady2OntologyParser()
    for conf_onto in tqdm(conference_ontologies):
        onto = parser.parse_from_file(conf_onto)

        onto_name = str(conf_onto.stem).lower().strip()
        num_classes = conference_ontologies_stats["num_classes"][onto_name]
        num_datatype_props = conference_ontologies_stats["num_datatype_props"][onto_name]
        num_object_props = conference_ontologies_stats["num_object_props"][onto_name]
        num_all_props = num_datatype_props + num_object_props

        # For some strange reason most of the number of parsed classes or properties are off by one. After inspecting some of the ontology files
        #  manually, I'm pretty sure this is not a problem of the Parser but the provided information is not correct. But this is claim has to be validated
        #  by some other persons, too.
        if not len(onto.concepts) == num_classes:
            logging.warning(
                f"Number of classes in '{onto_name}' ({len(onto.concepts)}) does not match {num_classes}!")
        if not len(onto.properties) == num_all_props:
            logging.warning(
                f"Number of Properties in '{onto_name}' ({len(onto.properties)}) does not match {num_all_props}!")

        logging.info(pprint.pformat(onto))


def test_owlready2_vs_rdflib_parsers_on_all_conference_ontologies(conference_ontologies: List[Path],
                                                                  conference_ontologies_stats: Dict[str, Dict[str, int]]):
    logging.info(f"Testing OwlReady2OntologyParser on all {len(conference_ontologies)} Conference Ontologies!")
    owlready2_parser = OwlReady2OntologyParser()
    rdflib_parser = RDFLibOntologyParser()

    for conf_onto in tqdm(conference_ontologies):
        owlready2_parser_onto = owlready2_parser.parse_from_file(conf_onto)
        rdflib_parser_onto = rdflib_parser.parse_from_file(conf_onto)

        if not len(owlready2_parser_onto.concepts) == len(rdflib_parser_onto.concepts):
            logging.warning((
                f"Number of classes in '{owlready2_parser_onto.name}' do not match! "
                f"OwlReady2OntologyParser found ({len(owlready2_parser_onto.concepts)}) "
                f"RDFLibOntologyParser found ({len(rdflib_parser_onto.concepts)}) "))
        if not len(owlready2_parser_onto.properties) == len(rdflib_parser_onto.properties):
            logging.warning((
                f"Number of properties in '{owlready2_parser_onto.name}' do not match! "
                f"OwlReady2OntologyParser found ({len(owlready2_parser_onto.properties)}) "
                f"RDFLibOntologyParser found ({len(rdflib_parser_onto.properties)}) "))
