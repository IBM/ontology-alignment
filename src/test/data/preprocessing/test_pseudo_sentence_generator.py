import logging
from typing import List

from data.preprocessing.pseudo_sentence_generator import PseudoSentenceGenerator
from data.preprocessing.owlready2_ontology_parser import OwlReady2OntologyParser
from pathlib import Path
from glob import glob
from tqdm import tqdm
from omegaconf import OmegaConf

import pytest

@pytest.fixture
def conference_ontologies() -> List[Path]:
    dataset_dir = Path("datasets/oaei/2021/conference/ontologies")
    assert dataset_dir.exists(), f"Cannot find datasets at: {dataset_dir}"
    return [Path(onto_file) for onto_file in glob(str(dataset_dir.joinpath("*.owl")))]


@pytest.fixture
def psg_parameter_configs() -> List[Path]:
    cfg_dir = Path("configurations/pseudo_sentence_generator")
    assert cfg_dir.exists(), f"Cannot find PseudoSentenceGenerator configs at: {cfg_dir}"
    return [Path(cfg) for cfg in glob(str(cfg_dir.joinpath("*.yaml")))]


def test_psg_cache_persistence_roundtrip(conference_ontologies: List[Path], psg_parameter_configs: List[Path]):
    logging.info((f"Testing PseudoSentenceGenerator all {len(psg_parameter_configs)} PSG Parameter Configurations"
                  f" on all {len(conference_ontologies)} Conference Ontologies with"))
    parser = OwlReady2OntologyParser()

    for cfg in tqdm(psg_parameter_configs):
        psg = PseudoSentenceGenerator(OmegaConf.load(cfg).pseudo_sentence_generator)

        for onto_fn in conference_ontologies:
            onto = parser.parse_from_file(onto_fn)

            fn, psc = psg.build_psg_cache("/tmp", onto, persist=True)
            psc2 = psg.read_psg_cache_of_onto("/tmp", onto)
            psc3 = psg.read_psg_cache(fn)

            assert all(psc == psc2) and all(psc == psc3) and all(psc2 == psc3) and (psc is not psc2 and psc is not psc3 and psc2 is not psc3)
