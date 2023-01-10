import argparse
import json
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
from pprint import pformat
from typing import Dict, Tuple
import nbformat as nbf


import pandas as pd
from model.nlfoa import NLFOA
from util import set_random_seeds
from data.model.ontology import Ontology
from data.model.predicted_alignment_rankings import PredictedAlignmentRankings, PREDICTED_ALIGNMENT_RANKINGS_FILE_EXTENSION
from data.model.reference_alignments import ReferenceAlignments, REFERENCE_ALIGNMENTS_FILE_EXTENSION
from data.preprocessing.rdflib_ontology_parser import RDFLibOntologyParser
from data.preprocessing.owlready2_ontology_parser import OwlReady2OntologyParser
from data.preprocessing.pseudo_sentence_generator import PSEUDO_SENTENCE_CACHE_FILE_EXTENSION, PseudoSentenceGenerator
from data.preprocessing.reference_alignments_parser import RDFLibAlignmentFormatReferenceAlignmentsParser
from model.pseudo_sentence_encoder import PseudoSentenceEncoder, SBertPSE
from model.similarity import compute_similarities
from omegaconf import DictConfig, OmegaConf
from scoring import compute_score_dict


def _load_inputs(config: DictConfig) -> Tuple[Ontology, Ontology, ReferenceAlignments, pd.DataFrame, pd.DataFrame]:
    # try to load PSCs for the two Ontologies. If not present, build and persist
    if config.io.ontology_parser == "RDFLibOntologyParser":
        parser = RDFLibOntologyParser()
    else:
        parser = OwlReady2OntologyParser()
    logging.info(f"Using {opts.parser} to parse Ontology File...")
    try:
        onto_a = Ontology.load_from_directory(config.io.onto_cache_dir, Path(config.io.onto_a).stem)
    except FileNotFoundError as e:
        logging.info(f"{e}")
        onto_a = parser.parse_from_file(config.io.onto_a)
        onto_a.persist(config.io.onto_cache_dir)
    try:
        onto_b = Ontology.load_from_directory(config.io.onto_cache_dir, Path(config.io.onto_b).stem)
    except FileNotFoundError as e:
        logging.info(f"{e}")
        onto_b = parser.parse_from_file(config.io.onto_b)
        onto_b.persist(config.io.onto_cache_dir)

    ref_alignments = RDFLibAlignmentFormatReferenceAlignmentsParser().parse_from_file(config.io.ref_alignments, onto_a, onto_b)

    logging.info(f"Using PSG config: {pformat(str(config.io.psg_config))}")
    assert Path(config.io.psg_config).exists(), \
        f"Cannot read Pseudo Sentence Generator config at: {config.io.psg_config}"
    psg_config = OmegaConf.load(config.io.psg_config).pseudo_sentence_generator
    psg = PseudoSentenceGenerator(psg_config)

    # try to load PSCs for the two Ontologies. If not present, build and persist
    onto_a_psc = psg.read_psg_cache_of_onto(config.io.ps_cache_dir, onto_a)
    if onto_a_psc is None:
        _, onto_a_psc = psg.build_psg_cache(config.io.ps_cache_dir, onto_a, persist=True)
    onto_b_psc = psg.read_psg_cache_of_onto(config.io.ps_cache_dir, onto_b)
    if onto_b_psc is None:
        _, onto_b_psc = psg.build_psg_cache(config.io.ps_cache_dir, onto_b, persist=True)

    return onto_a, onto_b, ref_alignments, onto_a_psc, onto_b_psc  # type: ignore


def _load_pseudo_sent_encoder(config: DictConfig) -> PseudoSentenceEncoder:
    if config.model.type == "sbert_baseline":
        logging.info("Loading SBert Baseline PseudoSentenceEncoder... ")
        encoder = SBertPSE(config.model.pertrained_name_or_path)
    elif config.model.type == "nlfoa":
        logging.info("Loading NLFOA PseudoSentenceEncoder... ")
        encoder = NLFOA(model_path=config.model.pertrained_name_or_path)
    else:
        raise NotImplementedError("Currenty only PseudoSentenceEncoder of type 'sbert_baseline' and 'nlfoa' are implemented!")

    encoder.to_device(config.model.device)
    # print(f"Max Seq Len: {encoder.model.max_seq_length}")
    logging.info("Loading PseudoSentenceEncoder... done")
    return encoder


def run_oa_eval(config: DictConfig,
                onto_a: Ontology,
                onto_b: Ontology,
                ref_alignments: ReferenceAlignments,
                onto_a_psc: pd.DataFrame,
                onto_b_psc: pd.DataFrame) -> Tuple[PredictedAlignmentRankings, Dict[str, float]]:

    encoder = _load_pseudo_sent_encoder(config)

    all_ps = pd.concat((onto_a_psc, onto_b_psc)).reset_index(drop=True)
    logging.info(f"Total number of Pseudo Sentences: {len(onto_a_psc)} + {len(onto_b_psc)} = {len(all_ps)}")

    logging.info("Computing dense vector representations... ")
    reps = encoder.encode(all_ps["pseudo_sentence"], device=config.model.device)

    logging.info(f"Computing '{config.similarity}' similarity between all {len(onto_a_psc) * len(onto_b_psc)} pairs...")
    sims = compute_similarities(reps_a=reps[:len(onto_a_psc)], reps_b=reps[len(onto_a_psc):], metric=config.similarity)

    logging.info("Building PredictionRankings from Similarity Tensor... ")
    pred_rankings = PredictedAlignmentRankings(onto_a=onto_a,
                                               onto_b=onto_b,
                                               a_uris=list(onto_a_psc["uri"].values),
                                               b_uris=list(onto_b_psc["uri"].values),
                                               similarity_scores=sims.numpy())
    logging.info("Building PredictionRankings from Similarity Tensor... done")

    logging.info("Computing scores...")
    score_dict = compute_score_dict(ref_alignments, pred_rankings, threshold=config.threshold)
    logging.info("Computing scores... done")
    return pred_rankings, score_dict


def _generate_inspect_data_notebook(results_dir: Path) -> None:
    logging.info("Generating Notebook to inspect data...")
    nb = nbf.v4.new_notebook()

    nb['cells'] = []

    intro_cell_text = """\
# This Notebook lets you inspect predicted Alignment Rankings and generated Pseudo Sentences
    """
    nb['cells'].append(nbf.v4.new_markdown_cell(intro_cell_text))

    load_data_cell_code = f"""\
import pandas as pd
from glob import glob

predicted_rankings = pd.read_parquet(glob("*{PREDICTED_ALIGNMENT_RANKINGS_FILE_EXTENSION}")[0])
print(f"Found Predicted Alignment Rankings of shape: {{predicted_rankings.shape}} ")

reference_alignments = pd.read_parquet(glob("*{REFERENCE_ALIGNMENTS_FILE_EXTENSION}")[0])
print(f"Found Reference Alignments with {{len(reference_alignments)}} alignments")

pseudo_sentences = {{
    onto_a: pd.read_parquet(fn)
    for onto_a, fn in map(lambda f: (f.split("_")[0], f), glob("*{PSEUDO_SENTENCE_CACHE_FILE_EXTENSION}"))
}}

print("Found the following PseudoSenteces for")
for onto_name, psc in pseudo_sentences.items():
    print(f"Ontology {{onto_name}} with {{len(psc)}} pseudo sentences")
"""
    nb['cells'].append(nbf.v4.new_code_cell(load_data_cell_code))

    inspect_rankings_cell_text = """\
# Inspect Predicted Rankings
"""
    inspect_rankings_cell_code = """\
predicted_rankings.head()
    """
    nb['cells'].append(nbf.v4.new_markdown_cell(inspect_rankings_cell_text))
    nb['cells'].append(nbf.v4.new_code_cell(inspect_rankings_cell_code))

    inspect_refs_cell_text = """\
# Inspect Reference Alignments
"""
    inspect_refs_cell_code = """\
reference_alignments.head()
    """
    nb['cells'].append(nbf.v4.new_markdown_cell(inspect_refs_cell_text))
    nb['cells'].append(nbf.v4.new_code_cell(inspect_refs_cell_code))

    inspect_ps_onto_a_cell_text = """\
# Inspect Pseudo Sentences of Ontology A
    """
    inspect_ps_onto_a_cell_code = """\
pseudo_sentences[list(pseudo_sentences.keys())[0]].head()
"""
    nb['cells'].append(nbf.v4.new_markdown_cell(inspect_ps_onto_a_cell_text))
    nb['cells'].append(nbf.v4.new_code_cell(inspect_ps_onto_a_cell_code))

    inspect_ps_onto_b_cell_text = """
# Inspect Pseudo Sentences of Ontology B
"""
    inspect_ps_onto_b_cell_code = """\
pseudo_sentences[list(pseudo_sentences.keys())[1]].head()
"""
    nb['cells'].append(nbf.v4.new_markdown_cell(inspect_ps_onto_b_cell_text))
    nb['cells'].append(nbf.v4.new_code_cell(inspect_ps_onto_b_cell_code))

    fn = results_dir.joinpath("inspect_data.ipynb")
    with open(fn, 'w') as f:
        nbf.write(nb, f)

    logging.info(f"Persited Data Inspection Notebook at: {fn}")


if __name__ == "__main__":
    start_t = time.time_ns()
    ap = argparse.ArgumentParser(
        description="This script runs an Ontology Alignment evaluation experiment defined in the provided run configuration.")
    ap.add_argument("-rc", "--run_config", help="Path to the run evaluation configuration", type=str)
    ap.add_argument("--parser",
                    default="OwlReady2OntologyParser",
                    const="OwlReady2OntologyParser",
                    nargs="?",
                    choices=["OwlReady2OntologyParser", "RDFLibOntologyParser"],
                    help="The OntologyParser used to parse the Ontology File. Default is 'OwlReady2OntologyParser'")

    opts = ap.parse_args()

    run_config = Path(opts.run_config)
    assert run_config.exists(), f"Cannot read run config at {opts.run_config}"
    config = OmegaConf.load(str(run_config)).eval_run_config

    # create results directory
    time_str = str(datetime.now()).replace(" ", "_")
    results_dir = Path(config.io.results_base_dir).joinpath(f"{run_config.stem}_{time_str}")
    if not results_dir.exists():
        results_dir.mkdir(parents=True)

    # setup logging for file (DEBUG) and stderr/stdout (INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(str(results_dir.joinpath("output.log")))
    file_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)

    set_random_seeds(1312)

    logging.info(f"Using run config: {opts.run_config}")
    logging.info(f"{json.dumps(OmegaConf.to_container(config), sort_keys=False, indent=2)}")

    eval_run_cfg_fn = str(results_dir.joinpath('eval_run_config.yaml'))
    logging.info(f"Copying config to results dir: {eval_run_cfg_fn}")
    shutil.copy(opts.run_config, eval_run_cfg_fn)

    psg_params_fn = str(results_dir.joinpath('psg_params.yaml'))
    logging.info(f"Copying PSG config to results dir: {psg_params_fn}")
    shutil.copy(config.io.psg_config, psg_params_fn)

    onto_a, onto_b, ref_alignments, onto_a_psc, onto_b_psc = _load_inputs(config)

    pred_rankings, score_dict = run_oa_eval(config, onto_a, onto_b, ref_alignments, onto_a_psc, onto_b_psc)

    # persist score dict in results dir
    scores_file = str(results_dir.joinpath("scores.json"))
    with open(scores_file, "w") as f:
        json.dump(score_dict, f, indent=2, sort_keys=False)
    logging.info(f"Persisted scores at: {scores_file}")

    # persist predicted rankings in results dir
    logging.info("Persisting predicted rankings...")
    _, rankings_file = pred_rankings.to_dataframe(results_dir)
    logging.info(f"Persisted predicted rankings at: {rankings_file}")

    # persist reference alignments in results dir
    logging.info("Persisting Reference Alignments...")
    _, alignments_file = ref_alignments.to_dataframe(results_dir)
    logging.info(f"Persisted Reference Alignments at: {alignments_file}")

    # persist PseudoSentenceCaches in results dir
    logging.info("Persisting PseudoSentences...")
    onto_a_psc_fn = results_dir.joinpath(f"{onto_a.name}{PSEUDO_SENTENCE_CACHE_FILE_EXTENSION}")
    onto_a_psc.to_parquet(f"{onto_a_psc_fn}")
    logging.info(f"Persisted PseudoSentences for {onto_a.name} at: {onto_a_psc_fn}")
    onto_b_psc_fn = results_dir.joinpath(f"{onto_b.name}{PSEUDO_SENTENCE_CACHE_FILE_EXTENSION}")
    onto_b_psc.to_parquet(f"{onto_b_psc_fn}")
    logging.info(f"Persisted PseudoSentences for {onto_b.name} at: {onto_b_psc_fn}")

    # generating notebook to inspect data
    _generate_inspect_data_notebook(results_dir)

    logging.info("============== Evaluation Scores =============")
    logging.info(f"{json.dumps(score_dict, indent=2, sort_keys=False)}")
    logging.info("==============================================")

    logging.info(f"Finished Evaluation Experiment in {(time.time_ns() - start_t) * 1e-9} seconds!")
