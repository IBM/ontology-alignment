import argparse
import logging
from pathlib import Path

from data.preprocessing.rdflib_ontology_parser import RDFLibOntologyParser
from data.preprocessing.owlready2_ontology_parser import OwlReady2OntologyParser
from data.preprocessing.pseudo_sentence_generator import PseudoSentenceGenerator
from omegaconf import OmegaConf

if __name__ == "__main__":

    logging.getLogger().setLevel(logging.INFO)

    ap = argparse.ArgumentParser()

    ap.add_argument("-of", "--onto_file",
                    help="Path to the ontology file for which to create the Pseudo Sentence Cache",
                    type=str)
    ap.add_argument("-out", "--output_dir",
                    help="Path to the output directory to persist Pseudo Sentence Cache",
                    type=str)
    ap.add_argument("-cfg", "--psg_config",
                    help="Path to the config file for the Pseudo Sentence Generator. If not provided default config will be used.",
                    type=str,
                    default=None)
    ap.add_argument("--parser",
                    default="OwlReady2OntologyParser",
                    const="OwlReady2OntologyParser",
                    nargs="?",
                    choices=["OwlReady2OntologyParser", "RDFLibOntologyParser"],
                    help="The OntologyParser used to parse the Ontology File. Default is 'OwlReady2OntologyParser'")

    opts = ap.parse_args()

    outdir = Path(opts.output_dir)
    assert outdir.exists(), f"Cannot read {outdir}"

    if opts.parser == "RDFLibOntologyParser":
        parser = RDFLibOntologyParser()
    else:
        parser = OwlReady2OntologyParser()
    logging.info(f"Using {opts.parser} to parse Ontology File...")

    onto_file = Path(opts.onto_file)
    logging.info(f"Parsing {onto_file} ...")
    onto = parser.parse_from_file(onto_file)

    config = None
    if opts.psg_config:
        logging.info(f"Using PSG config from: {opts.psg_config}")
        config = OmegaConf.load(opts.psg_config).pseudo_sentence_generator

    psg = PseudoSentenceGenerator(config=config)
    psg.build_psg_cache(outdir=outdir, onto=onto, persist=True)
