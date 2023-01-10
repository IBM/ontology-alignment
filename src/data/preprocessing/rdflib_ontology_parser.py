import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Tuple, Union

from rdflib import Graph
from rdflib.namespace import OWL, RDF
from rdflib.query import Result, ResultRow
from rdflib.term import BNode, URIRef, XSDToPython

from data.model.concept import Concept
from data.model.ontology import Ontology
from data.model.property import Property
from data.preprocessing.ontology_parser import IOntologyParser


class RDFLibOntologyParser(IOntologyParser):
    _g: Graph

    def __init__(self) -> None:
        super().__init__()
        logging.warning(("This parser does not implement hierarchies, labels, comments for Properties. "
                         "Please consider using the 'OwlReady2OntologyParser', which is also faster by some magnitudes!"))

    def parse_from_file(self, input_file: Union[str, Path]) -> Ontology:
        input_file = Path(input_file)
        assert input_file.exists() and input_file.is_file(), f"Cannot read Ontology from: {input_file}"

        logging.info(f"Parsing Ontology from {input_file} ...")
        start_parsing = time.time_ns()
        try:
            start = time.time_ns()
            self._g = Graph().parse(str(input_file))
            logging.debug(f"RDFLib Parsing took: {(time.time_ns() - start) * 1e-9} seconds")

            start = time.time_ns()
            roots, concepts = self._get_concept_hierarchy()
            logging.debug(f"Getting all {len(concepts)} Concepts took: {(time.time_ns() - start) * 1e-9} seconds")

            start = time.time_ns()
            props = self._get_all_properties(concepts)
            logging.debug(f"Getting all {len(props)} Properties took: {(time.time_ns() - start) * 1e-9} seconds")

            start = time.time_ns()
            self._set_concept_properties(concepts, props)
            logging.debug(f"Setting Properties for Concepts took: {(time.time_ns() - start) * 1e-9} seconds")

            onto = Ontology(input_file=input_file,
                            concepts=concepts,
                            properties=props)
            logging.debug(f"Parsing Ontology took: {(time.time_ns() - start_parsing) * 1e-9} seconds")

        except Exception as e:
            msg = f"Cannot parse Ontology {input_file} because:\n{e}"
            logging.error(msg)
            raise SystemExit(msg)

        return onto

    def _get_root_concepts(self) -> Dict[str, Concept]:
        root_results = self._execute_sparql_query("""
            SELECT DISTINCT ?concept
            WHERE {
              {
                SELECT DISTINCT ?concept
                WHERE {
                  { ?concept rdf:type owl:Class . } UNION
                  { ?concept  rdf:type rdfs:Class . }
                  FILTER NOT EXISTS {
                    { ?concept rdfs:subClassOf ?super . } UNION
                    { ?sub rdfs:subClassOf ?concept . }
                  }
                }
              }
              UNION
              {
                SELECT DISTINCT ?concept
                WHERE {
                  { ?concept rdf:type owl:Class . } UNION
                  { ?concept  rdf:type rdfs:Class . }
                  FILTER EXISTS {
                      { ?subClass rdfs:subClassOf ?concept . } UNION
                      { ?concept rdfs:subClassOf owl:Thing . }
                  }
                }
              }
            }
        """)
        roots: Dict[str, Concept] = dict()
        for rr in root_results:
            uri = rr.asdict()["concept"]  # type: ignore
            if isinstance(uri, BNode):
                logging.debug(f"Skipping BNode Root Concept for {uri}")
                continue
            elif isinstance(uri, URIRef):
                qname = self._g.namespace_manager.compute_qname(uri)
                uri = str(uri)
                root = Concept(name=qname[2], uri=uri)
                root.comments = self._get_concept_comments(root)
                root.labels = self._get_concept_labels(root)
                if uri in roots:
                    msg = f"Duplicate root Concept with URI {root.uri}"
                    logging.error(msg)
                    raise SystemExit(msg)
                roots[uri] = root
            else:
                logging.debug(f"Skipping unknown root Concept {uri} of type {type(uri)}")
                continue
        return roots

    def _get_concept_hierarchy(self) -> Tuple[Dict[str, Concept], Dict[str, Concept]]:
        roots = self._get_root_concepts()

        all_concepts: Dict[str, Concept] = dict(roots)
        for root in roots.values():
            self._get_child_concepts_recursively(parent=root, all_concepts=all_concepts)

        return roots, all_concepts

    def _get_child_concepts_recursively(self, parent: Concept, all_concepts: Dict[str, Concept]) -> None:
        children_result = self._execute_sparql_query(f"""
            SELECT ?child
            WHERE {{
                {{ ?child rdf:type owl:Class . }} UNION {{ ?child  rdf:type rdfs:Class . }}
                ?child rdfs:subClassOf <{str(parent.uri)}>
            }}
        """)

        for cr in children_result:
            uri = str(cr.asdict()["child"])  # type: ignore
            if uri in all_concepts:
                child = all_concepts[uri]
            else:
                qname = self._g.namespace_manager.compute_qname(uri)
                child = Concept(name=qname[2], uri=uri)
                child.comments = self._get_concept_comments(child)
                child.labels = self._get_concept_labels(child)
                all_concepts[uri] = child

            parent.children[child.uri] = child
            child.parents[parent.uri] = parent

            self._get_child_concepts_recursively(parent=child, all_concepts=all_concepts)

    def _get_concept_comments(self, concept: Concept) -> List[str]:
        comments = []
        comment_result = self._execute_sparql_query(f"""
            SELECT ?comment
            WHERE {{
                <{str(concept.uri)}> rdfs:comment ?comment .
            }}
        """)
        for cr in comment_result:
            comments.append(str(cr.comment.toPython()))  # type: ignore
        return comments

    def _get_concept_labels(self, concept: Concept) -> List[str]:
        labels = []
        label_result = self._execute_sparql_query(f"""
            SELECT ?label
            WHERE {{
                <{str(concept.uri)}> rdfs:label ?label .
            }}
        """)
        for lr in label_result:
            labels.append(str(lr.label.toPython()))  # type: ignore
        return labels

    def _is_xsd_datatype(self, uri: Union[URIRef, Any]) -> bool:
        if not isinstance(uri, URIRef):
            return False
        try:
            return str(self._g.namespace_manager.qname(uri)).startswith("xsd:")
        except Exception:
            return False

    def _is_union(self, bn: Union[BNode, Any]) -> bool:
        if not isinstance(bn, BNode):
            return False
        return len(list(self._g.objects(bn, OWL.unionOf))) == 1

    def _get_domain_or_range_from_result_row(self, uri: Union[URIRef, BNode], concepts: Dict[str, Concept]) -> List[Union[Concept, Any]]:
        result = []
        if uri and uri in concepts:
            result.append(concepts[uri])
        elif self._is_union(uri):
            result = self._get_union_members(uri, concepts)  # type: ignore
        elif self._is_xsd_datatype(uri):
            result.append(XSDToPython[uri])  # type: ignore
        else:
            logging.debug(f"Cannot handle {uri} of type {type(uri)} for Domain or Range of Property!")

        return result

    def _create_property_from_result_row(self, rr: ResultRow, concepts: Dict[str, Concept]) -> Property:
        pr_dict: DefaultDict[str, Union[URIRef, BNode]] = defaultdict(lambda: None, rr.asdict())   # type: ignore

        property_uri = pr_dict["property"]
        property_qname = self._g.namespace_manager.compute_qname(property_uri)

        prop = Property(name=property_qname[2],
                        uri=str(property_uri))

        domain_uri = pr_dict["domain"]
        domain = self._get_domain_or_range_from_result_row(domain_uri, concepts)
        prop.domain = domain  # type: ignore

        range_uri = pr_dict["range"]
        range = self._get_domain_or_range_from_result_row(range_uri, concepts)
        prop.range = range

        return prop

    def _get_all_properties(self, concepts: Dict[str, Concept]) -> Dict[str, Property]:
        prop_results = self._execute_sparql_query(""" 
            SELECT DISTINCT ?domain ?property ?range
            WHERE {
                { ?property rdf:type owl:ObjectProperty . } UNION
                { ?property rdf:type owl:DatatypeProperty . }
                {
                    SELECT ?range ?domain
                    WHERE {
                        OPTIONAL { ?property rdfs:range ?range . }
                        OPTIONAL { ?property rdfs:domain ?domain . }
                    }
                }
            }
        """)
        props = dict()
        for pr in prop_results:
            prop = self._create_property_from_result_row(pr, concepts)  # type: ignore
            props[prop.uri] = prop

        return props

    def _set_concept_properties(self, concepts: Dict[str, Concept], props: Dict[str, Property]) -> None:
        for p in props.values():
            for d in p.domain:
                if isinstance(d, Concept):
                    d.outgoing_properties[p.uri] = p

            for r in p.range:
                if isinstance(r, Concept):
                    r.incoming_properties[p.uri] = p

    def _execute_sparql_query(self, query: str) -> Result:
        return self._g.query(query, processor="sparql")

    def _get_rdf_collection_member_uris_recursively(self, collection_head_node: BNode, members: List[str]) -> None:
        first = list(self._g.objects(collection_head_node, RDF.first))
        assert len(first) == 1, "Exactly one first required!"
        members.append(str(first[0]))  # type: ignore

        rest = list(self._g.objects(collection_head_node, RDF.rest))
        assert len(first) == 1, "Exactly one rest required!"
        if rest[0] != RDF.nil and isinstance(rest[0], BNode):
            self._get_rdf_collection_member_uris_recursively(collection_head_node=rest[0], members=members)

    def _get_union_members(self, union_node: BNode, concepts: Dict[str, Concept]) -> List[Union[Concept, Any]]:
        if not self._is_union(union_node):
            logging.error("Node is not a Union!")
            return []

        member_uris = []
        head_node = next(iter(self._g.objects(union_node, OWL.unionOf)))
        self._get_rdf_collection_member_uris_recursively(head_node, member_uris)  # type: ignore

        return [XSDToPython[uri] if self._is_xsd_datatype(uri) else concepts[uri] for uri in member_uris]
