import argparse
import logging
from glob import glob
from pathlib import Path

from data.preprocessing.rdflib_ontology_parser import RDFLibOntologyParser
from data.preprocessing.owlready2_ontology_parser import OwlReady2OntologyParser
from data.preprocessing.pseudo_sentence_generator import PseudoSentenceGenerator
from omegaconf import OmegaConf
from tqdm import tqdm

if __name__ == "__main__":

    logging.getLogger().setLevel(logging.INFO)

    ap = argparse.ArgumentParser()

    ap.add_argument("-od", "--ontos_dir",
                    help="Path to the directory that contains the ontologies",
                    type=str)
    ap.add_argument("-out", "--output_dir",
                    help="Path to the output directory to persist Pseudo Sentence Caches",
                    type=str)
    ap.add_argument("-cfgd", "--psg_configs_dir",
                    help="Path to the directory that contains the config files for the Pseudo Sentence Generator",
                    type=str)
    ap.add_argument("--onto_file_ext",
                    help="File extension of the Ontology files. Default is '.owl'",
                    type=str,
                    default=None)
    ap.add_argument("--parser",
                    default="OwlReady2OntologyParser",
                    const="OwlReady2OntologyParser",
                    nargs="?",
                    choices=["OwlReady2OntologyParser", "RDFLibOntologyParser"],
                    help="The OntologyParser used to parse the Ontology File. Default is 'OwlReady2OntologyParser'")

    opts = ap.parse_args()

    output_dir = Path(opts.output_dir)
    assert output_dir.exists(), f"Cannot read {output_dir}"

    ontos_dir = Path(opts.ontos_dir)
    assert ontos_dir.exists(), f"Cannot read: {ontos_dir}"
    onto_files = [Path(onto_file) for onto_file in glob(
        str(ontos_dir.joinpath(f"*{opts.onto_file_ext if opts.onto_file_ext else '.owl'}")))]
    logging.info(f"Found {len(onto_files)} Ontologies at {ontos_dir}")

    psg_configs_dir = Path(opts.psg_configs_dir)
    assert psg_configs_dir.exists(), f"Cannot find PseudoSentenceGenerator configs at: {psg_configs_dir}"
    cfg_files = [Path(cfg) for cfg in glob(str(psg_configs_dir.joinpath("*.yaml")))]
    logging.info(f"Found {len(onto_files)} PSG Configs at {psg_configs_dir}")

    if opts.parser == "RDFLibOntologyParser":
        parser = RDFLibOntologyParser()
    else:
        parser = OwlReady2OntologyParser()
    logging.info(f"Using {opts.parser} to parse Ontology File...")

    logging.info(f"Generating {len(cfg_files)* len(onto_files)} PseudoSentenceCaches...")

    for cfg in tqdm(cfg_files):
        psg = PseudoSentenceGenerator(OmegaConf.load(cfg).pseudo_sentence_generator)

        for onto_fn in onto_files:
            onto = parser.parse_from_file(onto_fn)

            fn, _ = psg.build_psg_cache(str(output_dir), onto, persist=True)

    logging.info(f"Generating {len(cfg_files)* len(onto_files)} PseudoSentenceCaches... Done")
