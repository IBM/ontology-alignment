from data.preprocessing.owlready2_ontology_parser import OwlReady2OntologyParser
from data.model.ontology import Ontology
import logging
import sys
sys.path.append("../src")


logging.getLogger().setLevel(logging.INFO)


def test_ontology_persistence_roundtrip():
    iasted_onto = OwlReady2OntologyParser().parse_from_file("datasets/oaei/2021/conference/ontologies/iasted.owl")
    iasted_onto_fn = iasted_onto.persist("/tmp")
    iasted_onto_2 = Ontology.load_from_file(iasted_onto_fn)
    assert iasted_onto_2 == iasted_onto and iasted_onto == iasted_onto_2 and iasted_onto is not iasted_onto_2

    esda_onto = OwlReady2OntologyParser().parse_from_file("datasets/oaei/2021/conference/ontologies/edas.owl")
    esda_onto_fn = esda_onto.persist("/tmp")
    esda_onto_2 = Ontology.load_from_file(esda_onto_fn)
    assert esda_onto_2 == esda_onto and esda_onto == esda_onto_2 and esda_onto is not esda_onto_2

    assert esda_onto != iasted_onto and esda_onto_2 != iasted_onto_2
