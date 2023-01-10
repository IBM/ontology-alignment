from datetime import date
import logging
import time
from pathlib import Path
from typing import Dict, Union, Tuple

from data.model.concept import Concept
from data.model.ontology import Ontology
from data.model.property import Property
from data.preprocessing.ontology_parser import IOntologyParser

import owlready2
from owlready2.entity import ThingClass
from owlready2.class_construct import LogicalClassConstruct, OneOf


class OwlReady2OntologyParser(IOntologyParser):

    def parse_from_file(self, input_file: Union[str, Path]) -> Ontology:
        input_file = Path(input_file)
        assert input_file.exists() and input_file.is_file(
        ) and input_file.suffix == ".owl", f"Cannot read OWL Ontology from: {input_file}"

        logging.info(f"Parsing Ontology from {input_file} ...")
        start_parsing = time.time_ns()
        try:
            start = time.time_ns()
            self._onto = owlready2.get_ontology(str(input_file)).load()
            logging.debug(f"OWLReady2 Parsing took: {(time.time_ns() - start) * 1e-9} seconds")

            start = time.time_ns()
            concepts, incomplete_props = self._get_concept_hierarchy()
            logging.debug(f"Getting all {len(concepts)} Concepts took: {(time.time_ns() - start) * 1e-9} seconds")

            start = time.time_ns()
            props = self._get_all_properties(concepts, incomplete_props)
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

    def _get_concept_hierarchy(self) -> Tuple[Dict[str, Concept], Dict[str, Property]]:
        concepts: Dict[str, Concept] = dict()
        incomplete_props: Dict[str, Property] = dict()
        for clazz in self._onto.classes():
            c = Concept(name=clazz.name, uri=clazz.iri)

            #  build incomplete Properties here since the domain information is only available from the class properties
            for cp in clazz.get_class_properties():
                # when range is also empty, we dont know if the class is in the domain or range
                if len(cp.domain) == 0 and len(cp.range) != 0:
                    if cp.iri in incomplete_props:
                        prop = incomplete_props[cp.iri]
                    else:
                        prop = Property(name=cp.name, uri=cp.iri)
                        incomplete_props[cp.iri] = prop
                    prop.domain.append(c)

            # parents
            for supclazz in clazz.is_a:
                if isinstance(supclazz, ThingClass):
                    if supclazz.iri in concepts:
                        parent = concepts[supclazz.iri]
                    else:
                        parent = Concept(name=supclazz.name, uri=supclazz.iri)
                    c.parents[parent.uri] = parent

            # childred
            for subclazz in clazz.subclasses():
                if isinstance(subclazz, ThingClass):
                    if subclazz.iri in concepts:
                        child = concepts[subclazz.iri]
                    else:
                        child = Concept(name=subclazz.name, uri=subclazz.iri)
                    c.children[child.uri] = child

            # comments
            for comment in clazz.comment:
                c.comments.append(str(comment))

            # labels
            for label in clazz.label:
                c.labels.append(str(label))

            concepts[clazz.iri] = c

        return concepts, incomplete_props

    def _get_all_properties(self, concepts: Dict[str, Concept], incomplete_props: Dict[str, Property]) -> Dict[str, Property]:
        # shallow copy! this method is not idempotent, do not run more than once!
        properties: Dict[str, Property] = dict(incomplete_props)
        for prop in self._onto.properties():
            p = Property(name=prop.name, uri=prop.iri) if prop.iri not in properties else properties[prop.iri]

            # parents
            try:
                for supprop in prop.is_a:
                    if supprop.iri != prop.iri and supprop.namespace.name != "owl":
                        if supprop.iri in properties:
                            parent = properties[supprop.iri]
                        else:
                            parent = Property(name=supprop.name, uri=supprop.iri)
                        p.parents[parent.uri] = parent
            except TypeError:
                logging.debug(f"Cannot parse parents of Propery {prop.iri}")

            # children
            try:
                for subprop in prop.subclasses():
                    if subprop.iri != prop.iri and subprop.namespace.name != "owl":
                        if subprop.iri in properties:
                            child = properties[subprop.iri]
                        else:
                            child = Property(name=subprop.name, uri=subprop.iri)
                        p.children[child.uri] = child
            except TypeError:
                logging.debug(f"Cannot parse parents of Propery {prop.iri}")

            # range
            try:
                for r in prop.range:
                    if isinstance(r, ThingClass):
                        p.range.append(concepts[r.iri])
                        concepts[r.iri].incoming_properties[p.uri] = p
                    elif isinstance(r, LogicalClassConstruct):
                        for cc in r.get_Classes():
                            p.range.append(concepts[cc.iri])
                            concepts[cc.iri].incoming_properties[p.uri] = p
                    elif isinstance(r, OneOf):
                        if isinstance(r.instances, int):
                            p.range.append(r)
                        else:
                            for cc in r.instances:
                                if isinstance(cc, (str, int, bool, date, float)):
                                    p.range.append(cc)
                                else:
                                    p.range.append(concepts[cc.iri])
                                    concepts[cc.iri].incoming_properties[p.uri] = p
                    elif r is not None:
                        p.range.append(str(r))
            except TypeError:
                logging.debug(f"Cannot parse Range of Propery {p.uri}")

            # domain
            try:
                for d in prop.domain:
                    if isinstance(d, ThingClass):
                        p.domain.append(concepts[d.iri])
                        concepts[d.iri].outgoing_properties[p.uri] = p
                    elif isinstance(d, LogicalClassConstruct):
                        for cc in d.get_Classes():
                            p.domain.append(concepts[cc.iri])
                            concepts[cc.iri].outgoing_properties[p.uri] = p
                    elif d is not None:
                        p.domain.append(d)
            except TypeError:
                logging.debug(f"Cannot parse Range of Propery {p.uri}")

            # get domain and range from parents
            for parent in p.parents.values():
                p.range.extend([r for r in parent.range if r not in p.range and r is not None])
                p.domain.extend([d for d in parent.domain if d not in p.domain and d is not None])

            # set domain and range for children
            for child in p.children.values():
                child.range.extend([r for r in p.range if r not in child.range and r is not None])
                child.domain.extend([d for d in p.domain if d not in child.domain and d is not None])

            properties[p.uri] = p

            # inverse
            try:
                inv = prop.get_inverse_property()
                if inv is not None:
                    p.inverse = Property(name=inv.name, uri=inv.iri) \
                        if inv.iri not in properties else properties[inv.iri]
            except AttributeError as e:
                logging.debug(f"Cannot get inverse property of {prop.iri} because: {e}")

        return properties
