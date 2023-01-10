import logging
from pathlib import Path
from pprint import pformat

import pandas as pd
from api.config import config
from api.models.rdf_alignment_format import RDFAlignmentFormat
from api.models.inference_api_settings import InferenceApiSettings
from data.alignment_format import write_alignment_format
from data.model.alignment import Alignment
from data.model.ontology import Ontology
from data.model.predicted_alignment_rankings import PredictedAlignmentRankings
from data.model.reference_alignments import ReferenceAlignments
from data.preprocessing.owlready2_ontology_parser import OwlReady2OntologyParser
from data.preprocessing.pseudo_sentence_generator import PseudoSentenceGenerator
from data.preprocessing.reference_alignments_parser import RDFLibAlignmentFormatReferenceAlignmentsParser
from fastapi import APIRouter, Form, Response
from omegaconf import OmegaConf
from model.nlfoa import NLFOA
from model.similarity import compute_similarities
from typing import Optional, Tuple
from datetime import datetime

inference = APIRouter()
tags = ["inference"]


def _build_psg(psg_config_file: Optional[str] = None) -> PseudoSentenceGenerator:
    if psg_config_file is None:
        psg_config_file = str(config.io.psg_config)

    assert Path(psg_config_file).exists(), \
        f"Cannot read Pseudo Sentence Generator config at: {psg_config_file}"

    psg_cfg = OmegaConf.load(psg_config_file).pseudo_sentence_generator
    logging.info(f"Using PSG config: {pformat(str(psg_cfg))}")
    return PseudoSentenceGenerator(psg_cfg)


def _load_nlfoa(model_pertrained_name_or_path: Optional[str] = None) -> NLFOA:
    if model_pertrained_name_or_path is None:
        model_pertrained_name_or_path = config.model.pertrained_name_or_path

    logging.info(f"Loading NLFOA from {model_pertrained_name_or_path}")
    nlfoa = NLFOA(model_path=model_pertrained_name_or_path)
    nlfoa.to_eval()
    nlfoa.to_device(config.model.device)
    return nlfoa


def _set_scores_threshold(scores_threshold: Optional[float] = None) -> float:
    if scores_threshold is None:
        scores_threshold = float(config.threshold)
    logging.info(f"Using threshold {scores_threshold}")

    return scores_threshold


def _parse_ontologies(source: str, target: str) -> Tuple[Ontology, Ontology]:
    # quick and dirty: write all input to (tmp) files and parse with Onto and Ali Parsers (to do it in-memory would require extending the parsers)
    onto_a_tmp_file = Path("/tmp/onto_a.owl")
    onto_b_tmp_file = Path("/tmp/onto_b.owl")

    if onto_a_tmp_file.exists():
        onto_a_tmp_file.unlink()
    if onto_b_tmp_file.exists():
        onto_b_tmp_file.unlink()

    with open(onto_a_tmp_file, 'w') as f:
        f.write(source)

    with open(onto_b_tmp_file, 'w') as f:
        f.write(target)

    onto_parser = OwlReady2OntologyParser()
    onto_a = onto_parser.parse_from_file(onto_a_tmp_file)
    onto_b = onto_parser.parse_from_file(onto_b_tmp_file)

    logging.info(f"Onto A: {onto_a}")
    logging.info(f"Onto B: {onto_b}")

    return onto_a, onto_b


def _parse_input_alignment(input_alignment: str, onto_a: Ontology, onto_b: Ontology) -> Optional[ReferenceAlignments]:
    if input_alignment is not None:
        ali_tmp_file = Path("/tmp/alis.rdf")
        if ali_tmp_file.exists():
            ali_tmp_file.unlink()
        
        with open(ali_tmp_file, 'w') as f:
            f.write(input_alignment)
        ali_parser = RDFLibAlignmentFormatReferenceAlignmentsParser()
        alis = ali_parser.parse_from_file(ali_tmp_file, onto_a, onto_b)

        logging.info(f"Alignments: {alis}")
        return alis
    return None


psg: PseudoSentenceGenerator = _build_psg()
nlfoa: NLFOA = _load_nlfoa()
threshold: float = _set_scores_threshold()


@inference.post('/set_inference_parameters', tags=["settings"])
def set_inference_parameters(settings: InferenceApiSettings):
    global psg
    global nlfoa
    global threshold
    if settings.psg_config is not None:
        psg = _build_psg(settings.psg_config)
    if settings.model_pertrained_name_or_path is not None:
        nlfoa = _load_nlfoa(settings.model_pertrained_name_or_path)
    if settings.scores_threshold is not None:
        threshold = _set_scores_threshold(settings.scores_threshold)


# This API matches the specifications from https://dwslab.github.io/melt/6_matcher_packaging/swagger_ui_melt.html
@inference.post(
    "/match",
    responses={
        200: {"model": RDFAlignmentFormat,
              "description": ("The alignment in the [alignment format](https://moex.gitlabpages.inria.fr/alignapi/format.html) "
                              "either as file (application/xml in case of multipart request) or as file URL (represented as a"
                              " string in case of form-urlencoded request)")},
        400: {"model": str,
              "description": "Some errors on the client side(like not providing a source or target OR formatted in the wrong way)."},
        500: {"model": str, "description": "Any server errors."},
    },
    tags=["match"],
    summary="Computes the alignment between the given ontologies/knowledge graphs as URLs",
    response_model_by_alias=True,
)
def match_post(
    source: str = Form(None, description=("The URI of the source ontology/knowledge graph. The format of the file depends on the matcher."
                                          " Possible formats rdf/xml, turtle, n3. "
                                          "The URI can be a file URI pointing to a local file or an external URI.")),
    target: str = Form(None, description=("The URI of the target ontology/knowledge graph. The format of the file depends on the matcher."
                                          " Possible formats rdf/xml, turtle, n3. "
                                          "The URI can be a file URI pointing to a local file or an external URI.")),
    input_alignment: str = Form(None, description=("The URI of the input alignment which is optional. The format needs to be the [alignment format]"
                                                   "(https://moex.gitlabpages.inria.fr/alignapi/format.html). "
                                                   "The URI can be a file URI pointing to a local file or an external URI.")),
    parameters: str = Form(None, description=("The URI of the parameters which is optional. Currently supported formats are JSON and YAML."
                                              " The parameters are usually only key value pairs. Some keys are already defined in "
                                              "[MELT](https://github.com/dwslab/melt/blob/master/matching-base/src/main/java/de/uni_mannheim/informatik/dws/melt/matching_base/ParameterConfigKeys.java)."
                                              " The URI can be a file URI pointing to a local file or an external URI."))):
    # using explicit global vars since they can be set via the set_inference_parameters endoint!
    global psg
    global nlfoa
    global threshold

    onto_a, onto_b = _parse_ontologies(source, target)

    # ali_parser = RDFLibAlignmentFormatReferenceAlignmentsParser()
    # alis = ali_parser.parse_from_file('datasets/oaei/2021/anatomy/alignments/mouse-human.rdf', onto_a, onto_b)
    # top1_alis = [Alignment(ali.a_uri, ali.b_uri, 1.0) for ali in alis.alignments]

    # ali_rdf = write_alignment_format(onto_a, onto_b, top1_alis, threshold)
    # with open(Path('/tmp').joinpath("alignments.rdf"), "w") as f:
    #     f.write(ali_rdf)

    # return Response(content=ali_rdf, media_type="application/xml")
    
    alis = _parse_input_alignment(input_alignment, onto_a, onto_b)

    results_dir = Path(config.io.results_base_dir).joinpath(str(datetime.now()).replace(' ', '_'))

    _, onto_a_psc = psg.build_psg_cache(results_dir, onto_a, persist=True)
    _, onto_b_psc = psg.build_psg_cache(results_dir, onto_b, persist=True)

    all_ps = pd.concat((onto_a_psc, onto_b_psc)).reset_index(drop=True)
    logging.info(f"Total number of Pseudo Sentences: {len(onto_a_psc)} + {len(onto_b_psc)} = {len(all_ps)}")

    logging.info("Computing dense vector representations... ")
    reps = nlfoa.encode(all_ps["pseudo_sentence"], device=config.model.device)

    logging.info(f"Computing '{config.similarity}' similarity between all {len(onto_a_psc) * len(onto_b_psc)} pairs...")
    sims = compute_similarities(reps_a=reps[:len(onto_a_psc)], reps_b=reps[len(onto_a_psc):], metric=config.similarity)

    logging.info("Building PredictionRankings from Similarity Tensor... ")
    pred_rankings = PredictedAlignmentRankings(onto_a=onto_a,
                                               onto_b=onto_b,
                                               a_uris=list(onto_a_psc["uri"].values),
                                               b_uris=list(onto_b_psc["uri"].values),
                                               similarity_scores=sims.numpy())
    logging.info("Building PredictionRankings from Similarity Tensor... done")

    # persist predicted rankings in results dir
    logging.info("Persisting predicted rankings...")
    _, rankings_file = pred_rankings.to_dataframe(results_dir)
    logging.info(f"Persisted predicted rankings at: {rankings_file}")

    top1_alis = []
    for a_uri in pred_rankings.a_uris:
        b_uri, sim = pred_rankings.get_sorted_rankings_for_uri(a_uri)[0]
        top1_alis.append(Alignment(a_uri, b_uri, sim))

    ali_rdf = write_alignment_format(onto_a, onto_b, top1_alis, threshold, only_positives=True)
    with open(results_dir.joinpath("alignments.rdf"), "w") as f:
        f.write(ali_rdf)

    return Response(content=ali_rdf, media_type="application/xml")
