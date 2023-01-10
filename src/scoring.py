from typing import Dict, List

import logging
import numpy as np
import pandas as pd

from data.model.predicted_alignment_rankings import PredictedAlignmentRankings
from data.model.reference_alignments import ReferenceAlignments
import sklearn.metrics as metrics


def compute_score_dict(refs: ReferenceAlignments,
                       preds: PredictedAlignmentRankings,
                       threshold: float = 0.9) -> Dict[str, float]:
    ranking_scores = compute_ranking_scores(refs, preds)
    classification_scores = compute_classification_scores(refs, preds, threshold)
    return {**ranking_scores, **classification_scores}


def compute_classification_scores(refs: ReferenceAlignments,
                                  preds: PredictedAlignmentRankings,
                                  threshold: float = 0.9,
                                  zero_division=0.) -> Dict[str, float]:

    logging.info(f"Computing Classification Scores with Treshold {threshold}!")
    refs_df: pd.DataFrame = refs.to_dataframe()  # type: ignore

    y_true = refs_df['b_uri'].to_list()  # all b_uris from reference alignments
    y_pred = []
    for a_uri in refs_df["a_uri"]:
        top1_b_uri, score = preds.get_sorted_rankings_for_uri(a_uri)[0]
        if score >= threshold:
            y_pred.append(top1_b_uri)
        else:
            y_pred.append("NO_MATCH")

    scores = {s: -1. for s in ["micro_precision",
                               "micro_recall",
                               "micro_f1",
                               "macro_precision",
                               "macro_recall",
                               "macro_f1",
                               "accuracy"]}

    for s in scores.keys():
        logging.debug(f"Computing {s} for {len(y_true)} Ground Truth prediction")
        if s == "micro_precision":
            scores[s] = metrics.precision_score(y_true,
                                                y_pred,
                                                average='micro',
                                                zero_division=zero_division)  # type: ignore
        elif s == "micro_recall":
            scores[s] = metrics.recall_score(y_true,
                                             y_pred,
                                             average='micro',
                                             zero_division=zero_division)  # type: ignore
        elif s == "micro_f1":
            scores[s] = metrics.f1_score(y_true,
                                         y_pred,
                                         average='micro',
                                         zero_division=zero_division)  # type: ignore
        elif s == "macro_precision":
            scores[s] = metrics.precision_score(y_true,
                                                y_pred, average='macro',
                                                zero_division=zero_division)  # type: ignore
        elif s == "macro_recall":
            scores[s] = metrics.recall_score(y_true, y_pred,
                                             average='macro',
                                             zero_division=zero_division)  # type: ignore
        elif s == "macro_f1":
            scores[s] = metrics.f1_score(y_true,
                                         y_pred,
                                         average='macro',
                                         zero_division=zero_division)  # type: ignore
        elif s == "accuracy":
            scores[s] = metrics.accuracy_score(y_true, y_pred)  # type: ignore

    return scores


def compute_ranking_scores(refs: ReferenceAlignments,
                           preds: PredictedAlignmentRankings,
                           top_k: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:

    # ground truth ranks of the b_uris of the reference alignments in the predicted rankings
    y_true = [preds.b_uris.index(b_uri) for b_uri in map(lambda a: a.b_uri, refs.alignments)]
    # indices of the a_uris of the reference alignments in the similarity score tensor
    a_uri_indices = [preds.a_uris.index(a_uri) for a_uri in map(lambda a: a.a_uri, refs.alignments)]
    y_score = preds.similarity_scores[a_uri_indices]

    scores = dict()
    for k in top_k:
        scores[f"hits@{k}"] = metrics.top_k_accuracy_score(y_true, y_score, k=k, labels=list(range(len(preds.b_uris))))

    pred_ranks = []
    for i, ranks in enumerate(y_score.argsort(axis=1)[:, ::-1]):
        pred_ranks += [np.where(ranks == y_true[i])[0][0] + 1]
    pred_ranks = np.asarray(pred_ranks)
    scores["mrr"] = np.mean(1/pred_ranks)

    return scores
