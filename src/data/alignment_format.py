from data.model.ontology import Ontology
from data.model.alignment import Alignment
from typing import List

# Taken from: https://github.com/Remorax/VeeAlign/blob/master/src/SEALS-OAEI.py


def write_alignment_format(onto_a: Ontology, onto_b: Ontology, alis: List[Alignment], threshold: float = .6, only_positives: bool = True):
    alignment_format_rdf = f"""\
<?xml version='1.0' encoding='utf-8' standalone='no'?>
<rdf:RDF xmlns='http://knowledgeweb.semanticweb.org/heterogeneity/alignment#'
         xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'
         xmlns:xsd='http://www.w3.org/2001/XMLSchema#'
         xmlns:align='http://knowledgeweb.semanticweb.org/heterogeneity/alignment#'>
    <Alignment>

        <xml>yes</xml>
        <level>0</level>
        <type>**</type>

        <onto1>
            <Ontology rdf:about="{onto_a.name}">
            <location>{onto_a.input_file}</location>
            </Ontology>
        </onto1>
        <onto2>
            <Ontology rdf:about="{onto_b.name}">
            <location>{onto_b.input_file}</location>
            </Ontology>
        </onto2>
        """

    for ali in alis:
        if only_positives and ali.similarity < threshold:
            continue
        mapping = f"""
        <map>
            <Cell>
            <entity1 rdf:resource='{ali.a_uri}'/>
            <entity2 rdf:resource='{ali.b_uri}'/>
            <relation>=</relation>
            <measure rdf:datatype='http://www.w3.org/2001/XMLSchema#float'>{"1.0" if ali.similarity >= threshold else "0.0" }</measure>
            </Cell>
        </map>"""

        alignment_format_rdf += mapping

    alignment_format_rdf += """
    </Alignment>
</rdf:RDF>"""
    return alignment_format_rdf.strip()
