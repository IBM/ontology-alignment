import argparse
from glob import glob
from pathlib import Path
import yaml
from typing import Dict, Any, Optional, Union, List, Tuple

import logging

EVAL_RUN_CFG_BASE = {
    "eval_run_config": {
        "io": {
            "onto_a": "FILEPATH",
            "onto_b": "FILEPATH",
            "ref_alignments": "FILEPATH",
            "psg_config": "FILEPATH",
            "onto_cache_dir": "DIRPATH",
            "ps_cache_dir": "DIRPATH",
            "results_base_dir": "DIRPATH",
            "ontology_parser": "OwlReady2OntologyParser"  # OwlReady2OntologyParser or RDFLibOntologyParser
        },
        "model": {
            "type": "sbert_baseline",
            "pertrained_name_or_path": "bert-base-uncased",  # any Transformer from huggingface
            "device": "cuda:0"
        },
        "similarity": "cosine",  # or dot
        "threshold": "0.6"  # for evaluation. If the computed similariy between two concepts is less than the threshold, it is considered as NO_MATCH
    }
}

AVAILABLE_ONTO_PARSERS = ["OwlReady2OntologyParser", "RDFLibOntologyParser"]  # RDFLibOntologyParser is deprecated
AVAILABLE_MODEL_TYPES = ["sbert_baseline", "nlfoa"]
AVAILABLE_DEVICES = ["cpu", "cuda", "cuda:0", "cuda:1"]
AVAILABLE_SIMILARITIES = ["cosine", "dot"]


def _generate_eval_run_cfg(onto_a: Path,
                           onto_b: Path,
                           ref_alignments: Path,
                           psg_config: Path,
                           onto_cache_dir: Path,
                           ps_cache_dir: Path,
                           results_base_dir: Path,
                           ontology_parser: str = "OwlReady2OntologyParser",
                           model_type: str = "sbert_baseline",
                           model_pertrained_name_or_path: str = "bert-base-uncased",
                           model_device: str = "cuda:1",
                           similarity: str = "cosine",
                           threshold: float = .6) -> Dict[str, Any]:

    assert onto_a.exists(), f"Cannot read: {onto_a}"
    assert onto_b.exists(), f"Cannot read: {onto_b}"
    assert ref_alignments.exists(), f"Cannot read: {ref_alignments}"
    assert psg_config.exists(), f"Cannot read: {psg_config}"
    assert onto_cache_dir.exists(), f"Cannot read: {onto_cache_dir}"
    assert ps_cache_dir.exists(), f"Cannot read: {ps_cache_dir}"
    if not results_base_dir.exists():
        results_base_dir.mkdir(parents=True)
    assert ontology_parser in AVAILABLE_ONTO_PARSERS, f"Unknown Ontology Parser! Available are: {AVAILABLE_ONTO_PARSERS}"
    assert model_type in AVAILABLE_MODEL_TYPES, f"Unknown Model Type! Available are: {AVAILABLE_MODEL_TYPES}"
    assert model_device in AVAILABLE_DEVICES, f"Unknown Device! Available are: {AVAILABLE_DEVICES}"
    assert similarity in AVAILABLE_SIMILARITIES, f"Unknown Similarity Measure! Available are: {AVAILABLE_SIMILARITIES}"

    cfg = EVAL_RUN_CFG_BASE.copy()

    cfg["eval_run_config"]["io"]["onto_a"] = str(onto_a)
    cfg["eval_run_config"]["io"]["onto_b"] = str(onto_b)
    cfg["eval_run_config"]["io"]["ref_alignments"] = str(ref_alignments)
    cfg["eval_run_config"]["io"]["psg_config"] = str(psg_config)
    cfg["eval_run_config"]["io"]["onto_cache_dir"] = str(onto_cache_dir)
    cfg["eval_run_config"]["io"]["ps_cache_dir"] = str(ps_cache_dir)
    cfg["eval_run_config"]["io"]["ontology_parser"] = str(ontology_parser)
    cfg["eval_run_config"]["io"]["results_base_dir"] = str(results_base_dir)

    cfg["eval_run_config"]["model"]["type"] = str(model_type)
    cfg["eval_run_config"]["model"]["pertrained_name_or_path"] = str(model_pertrained_name_or_path)
    cfg["eval_run_config"]["model"]["device"] = str(model_device)

    cfg["eval_run_config"]["similarity"] = str(similarity)
    cfg["eval_run_config"]["threshold"] = float(threshold)

    return cfg


def _get_available_reference_alignment_files(alignments_dir: Union[str, Path]) -> List[Path]:
    alignments_dir = Path(alignments_dir)
    assert alignments_dir.exists() and alignments_dir.is_dir(), f"Cannot read: {alignments_dir}"

    return [Path(ali) for ali in glob(str(alignments_dir.joinpath("*.rdf")))]


def _get_ontologies_from_reference_alignment_files(ontos_dir: Union[str, Path], alignments: List[Path]) -> List[Tuple[Path, Path]]:
    ontos_dir = Path(ontos_dir)
    assert ontos_dir.exists() and ontos_dir.is_dir(), f"Cannot read: {ontos_dir}"

    ontos: List[Tuple[Path, Path]] = []
    for ali in alignments:
        ontos_in_alignment = ali.stem.split("-")
        onto_a = ontos_dir.joinpath(f"{ontos_in_alignment[0]}.owl")
        onto_b = ontos_dir.joinpath(f"{ontos_in_alignment[1]}.owl")

        assert onto_a.exists(), f"Cannot read: {onto_a}"
        assert onto_b.exists(), f"Cannot read: {onto_b}"

        ontos.append((onto_a, onto_b))

    return ontos


def _get_available_psg_config_files(psg_config_dir: Union[str, Path]) -> List[Path]:
    psg_config_dir = Path(psg_config_dir)
    assert psg_config_dir.exists(), f"Cannot read: {psg_config_dir}"

    return [Path(cfg) for cfg in glob(str(psg_config_dir.joinpath("*.yaml")))]


def generate_evaluation_run_configs(alignments_dir: Union[str, Path],
                                    ontos_dir: Union[str, Path],
                                    psg_config_dir: Union[str, Path],
                                    onto_cache_dir: Union[str, Path],
                                    ps_cache_dir: Union[str, Path],
                                    results_base_dir: Union[str, Path],
                                    pretrained_model: str,
                                    parser: str,
                                    model_type: str,
                                    device: str,
                                    similarity: str,
                                    threshold: float,
                                    output_base_dir: Optional[Union[str, Path]] = None) -> Dict[str, Dict[str, Any]]:
    alignments_dir = Path(alignments_dir)
    ontos_dir = Path(ontos_dir)
    psg_config_dir = Path(psg_config_dir)
    onto_cache_dir = Path(onto_cache_dir)
    ps_cache_dir = Path(ps_cache_dir)
    results_base_dir = Path(results_base_dir)

    alis = _get_available_reference_alignment_files(alignments_dir)
    ontos = _get_ontologies_from_reference_alignment_files(ontos_dir, alis)
    unique_ontos = set()
    for a, b in ontos:
        unique_ontos.add(a)
        unique_ontos.add(b)
    logging.info(f"Found {len(alis)} reference alignment files comprising {len(unique_ontos)} Ontologies")

    psg_configs = _get_available_psg_config_files(psg_config_dir)
    logging.info(f"Found {len(psg_configs)} PSG configs")

    logging.info(f"Using the following pretrained model: {pretrained_model}")

    num_eval_run_cfgs = len(alis) * len(psg_configs)
    logging.info((f"Generating {len(alis)} * {len(psg_configs)} = {num_eval_run_cfgs} combinations"
                  f" of Evaluation Run Config Files at {output_base_dir}"))

    cfgs: Dict[str, Dict[str, Any]] = dict()
    for ali, (onto_a, onto_b) in zip(alis, ontos):
        for psg_cfg in psg_configs:
            cfg_name = f"{model_type}_{pretrained_model}_{psg_cfg.stem}_for_{ali.stem}-alignments.yaml"
            cfg = _generate_eval_run_cfg(onto_a=onto_a,
                                         onto_b=onto_b,
                                         ref_alignments=ali,
                                         psg_config=psg_cfg,
                                         onto_cache_dir=onto_cache_dir,
                                         ps_cache_dir=ps_cache_dir,
                                         results_base_dir=results_base_dir,
                                         ontology_parser=parser,
                                         model_type=model_type,
                                         model_pertrained_name_or_path=pretrained_model,
                                         model_device=device,
                                         similarity=similarity,
                                         threshold=threshold)
            cfgs[cfg_name] = cfg
            if output_base_dir is not None:
                fn = Path(output_base_dir).joinpath(ali.stem).joinpath(cfg_name)
                logging.debug(f"Persisting Evaluation Run Config at {fn}")
                if not fn.parent.exists():
                    fn.parent.mkdir(parents=True)
                with open(str(fn), "w") as f:
                    yaml.dump(cfg, f)

    return cfgs


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    ap = argparse.ArgumentParser(
        description="This script generates all possible combinations of Evaluation Run Config Files")

    ap.add_argument("alignments_dir",
                    help="Path to the directory containing the reference alignments")

    ap.add_argument("ontos_dir",
                    help="Path to the directory containing the ontologies")

    ap.add_argument("psg_config_dir",
                    help="Path to the PseudoSentenceGenerator configuration files used in the generated evaluation run configs")

    ap.add_argument("output_base_dir",
                    help="Path where the generated Evaluation Run Config Files are stored")

    ap.add_argument("--onto_cache_dir",
                    default="data/onto_caches",
                    nargs="?",
                    help="Path where the Ontologies get cached")

    ap.add_argument("--ps_cache_dir",
                    default="data/ps_caches",
                    nargs="?",
                    help="Path where the PseudoSentences get cached")

    ap.add_argument("--results_base_dir",
                    default="data/eval_results",
                    nargs="?",
                    help="Path where the evaluation results are stored")

    ap.add_argument("--model_type",
                    default="sbert_baseline",
                    const="sbert_baseline",
                    nargs="?",
                    choices=AVAILABLE_MODEL_TYPES,
                    help="Type of the model")

    ap.add_argument("--pretrained_model",
                    default="bert-base-uncased",
                    nargs="?",
                    help="The pretrained model used. Can be either a path to a NLFOA or the name of a pretrained LM on huggingface, e.g., bert-base-uncased")

    ap.add_argument("--parser",
                    default="OwlReady2OntologyParser",
                    const="OwlReady2OntologyParser",
                    nargs="?",
                    choices=AVAILABLE_ONTO_PARSERS,
                    help="The OntologyParser used to parse the Ontology Files. Default is 'OwlReady2OntologyParser'")

    ap.add_argument("--device",
                    default="cuda:0",
                    const="cuda:0",
                    nargs="?",
                    choices=AVAILABLE_DEVICES)

    ap.add_argument("--similarity",
                    default="cosine",
                    const="cosine",
                    nargs="?",
                    choices=AVAILABLE_SIMILARITIES)

    ap.add_argument("--threshold",
                    default=.6,
                    type=float,
                    nargs="?")

    opts = ap.parse_args()

    logging.info("Generating Evaluation Run Config Files using the following inputs:")
    logging.info(f"{opts}")

    generate_evaluation_run_configs(**vars(opts))
