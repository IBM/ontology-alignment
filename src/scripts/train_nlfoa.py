import argparse
import importlib
import json
import logging
import shutil
import time
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from data.datasets.dynamic_pseudo_sentence_alignment_dataset import DynamicPseudoSentenceAlignmentDataset
from data.model.alignment import Alignment
from data.preprocessing.owlready2_ontology_parser import OwlReady2OntologyParser
from data.preprocessing.pseudo_sentence_generator import default_config
from model.nlfoa import NLFOA
from omegaconf import DictConfig, OmegaConf
from sentence_transformers import LoggingHandler
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader
from util import set_random_seeds


def _load_dataframes(cfg: DictConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logging.info("Loading DataFrames")
    train_set_fn = Path(cfg.io.train_set)
    assert train_set_fn.exists(), f"Cannot read train set at: {train_set_fn}"
    train_set = pd.read_parquet(train_set_fn)

    test_set_fn = Path(cfg.io.test_set)
    assert test_set_fn.exists(), f"Cannot read test set at: {test_set_fn}"
    test_set = pd.read_parquet(test_set_fn)

    val_set_fn = Path(cfg.io.val_set)
    assert val_set_fn.exists(), f"Cannot read validate set at: {val_set_fn}"
    val_set = pd.read_parquet(val_set_fn)

    return train_set, test_set, val_set


def _create_dynamic_ps_datasets(train_df: pd.DataFrame, test_df: pd.DataFrame, val_df: pd.DataFrame, cfg: DictConfig) \
    -> Tuple[DynamicPseudoSentenceAlignmentDataset,
             DynamicPseudoSentenceAlignmentDataset,
             DynamicPseudoSentenceAlignmentDataset]:

    logging.info("Creating Dynamic PseudoSentence Datasets!")

    # load ontologies
    onto_parser = OwlReady2OntologyParser()
    ontos_glob = cfg.training.dynamic_pseudo_sentences.ontologies_glob
    onto_paths = glob(str(ontos_glob))
    if len(onto_paths) == 0:
        raise SystemExit(f"Cannot find any Ontology at {ontos_glob}")
    logging.debug(f"Loading {len(onto_paths)} Ontologies from {ontos_glob}!")
    ontologies = [onto_parser.parse_from_file(f) for f in onto_paths]

    # create alignments
    train_alis = [Alignment(row.uri_a, row.uri_b, row.similarity) for _, row in train_df.iterrows()]
    test_alis = [Alignment(row.uri_a, row.uri_b, row.similarity) for _, row in test_df.iterrows()]
    val_alis = [Alignment(row.uri_a, row.uri_b, row.similarity) for _, row in val_df.iterrows()]

    # load psg config
    psg_config = OmegaConf.load(cfg.training.dynamic_pseudo_sentences.psg_config).pseudo_sentence_generator

    train_ds = DynamicPseudoSentenceAlignmentDataset(ontologies,
                                                     train_alis,
                                                     psg_config,
                                                     cfg.training.dynamic_pseudo_sentences.shuffle_ps)
    test_ds = DynamicPseudoSentenceAlignmentDataset(ontologies,
                                                    test_alis,
                                                    psg_config,
                                                    cfg.training.dynamic_pseudo_sentences.shuffle_ps)
    val_ds = DynamicPseudoSentenceAlignmentDataset(ontologies,
                                                   val_alis,
                                                   psg_config,
                                                   cfg.training.dynamic_pseudo_sentences.shuffle_ps)

    return train_ds, test_ds, val_ds


def _prepare_data(train_df: pd.DataFrame,
                  test_df: pd.DataFrame,
                  val_df: pd.DataFrame,
                  cfg: DictConfig) -> Tuple[DataLoader, List[InputExample]]:
    logging.info("Preparing data...")

    if "dynamic_pseudo_sentences" in cfg.training:
        train_data, _, val_data = _create_dynamic_ps_datasets(train_df, test_df, val_df, cfg)
        val_samples = [val_ps for val_ps in val_data]
    else:
        logging.debug("Using static, pre-generated PseudoSentences!")
        train_data = [InputExample(str(idx), [row.pseudo_sentence_a, row.pseudo_sentence_b], row.similarity)
                      for idx, row in train_df.iterrows()]
        val_samples = [InputExample(str(idx), [row.pseudo_sentence_a, row.pseudo_sentence_b], row.similarity)
                       for idx, row in val_df.iterrows()]

    train_batch_size = cfg.training.train_batch_size
    shuffle = cfg.training.shuffle_dataloaders

    train_dl = DataLoader(train_data, batch_size=train_batch_size, shuffle=shuffle)  # type: ignore

    return train_dl, val_samples


def _instantiate_nlfoa(config: DictConfig) -> NLFOA:
    logging.info("Instantiating NLFOA model... ")
    special_characters = list(default_config.special_characters.values()) if config.model.add_special_tokens else []

    if "word_embedding_model" in config.model:
        nlfoa = NLFOA(special_characters=special_characters,
                      word_embedding_model=config.model.word_embedding_model,
                      max_seq_length=config.model.max_seq_length,
                      pooling_mode=config.model.pooling_mode,
                      device=config.model.device)
    elif "sts_model" in config.model:
        nlfoa = NLFOA(special_characters=special_characters,
                      sts_model=config.model.sts_model,
                      pooling_mode=config.model.pooling_mode,
                      device=config.model.device)
    else:
        msg = "Cannot instantiate NLFOA model because neither a word_embedding_model nor an sts_model was provided!"
        logging.error(msg)
        raise SystemExit(msg)
    return nlfoa


def run_training(config: DictConfig, results_dir: Path) -> None:
    train_df, test_df, val_df = _load_dataframes(config)
    train_dl, val_samples = _prepare_data(train_df, test_df, val_df, config)

    model_save_path = results_dir.joinpath("model")
    nlfoa = _instantiate_nlfoa(config)
    nlfoa.save(model_save_path)

    loss_module = importlib.import_module("sentence_transformers.losses")
    loss = getattr(loss_module, config.training.loss)(model=nlfoa.model)

    evaluator_module = importlib.import_module("sentence_transformers.evaluation")
    evaluator = getattr(evaluator_module, config.training.evaluator).from_input_examples(val_samples, name='nlfoa-val')

    optimizer_module = importlib.import_module("torch.optim")
    optimizer = getattr(optimizer_module, config.training.optimizer)

    num_epochs = config.training.epochs
    warmup_steps = np.ceil(len(train_dl) * num_epochs * config.training.warmup_steps)

    nlfoa.model.fit(train_objectives=[(train_dl, loss)],  # type: ignore
                    evaluator=evaluator,
                    epochs=num_epochs,
                    scheduler=config.training.warmup_scheduler,
                    warmup_steps=warmup_steps,
                    optimizer_class=optimizer,
                    optimizer_params={'lr':  config.training.lr},
                    weight_decay=config.training.weight_decay,
                    evaluation_steps=config.training.evaluation_steps,
                    output_path=str(model_save_path),
                    use_amp=config.training.use_amp,
                    checkpoint_path=str(model_save_path.joinpath("checkpoints")),
                    checkpoint_save_steps=config.training.checkpoint_save_steps)


if __name__ == "__main__":
    start_t = time.time_ns()
    ap = argparse.ArgumentParser(
        description="This script trains a NLFOA model based on the provided training configuration.")
    ap.add_argument("-tc", "--train_config", help="Path to the training configuration", type=str)

    opts = ap.parse_args()

    train_config = Path(opts.train_config)
    assert train_config.exists(), f"Cannot read training config at {opts.train_config}"
    config = OmegaConf.load(str(train_config)).nlfoa_training_config

    # create results directory
    time_str = str(datetime.now()).replace(" ", "_")
    results_dir = Path(config.io.results_dir).joinpath(f"{train_config.stem}_{time_str}")
    if not results_dir.exists():
        results_dir.mkdir(parents=True)

    # setup logging for file (DEBUG) and stderr/stdout (DEBUG)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(str(results_dir.joinpath("output.log")))
    file_handler.setLevel(logging.DEBUG)
    sbert_handler = LoggingHandler()
    sbert_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(sbert_handler)
    root_logger.addHandler(file_handler)

    set_random_seeds(config.random_seed)

    logging.info(f"Using training config: {opts.train_config}")
    logging.info(f"{json.dumps(OmegaConf.to_container(config), sort_keys=False, indent=2)}")

    train_cfg_fn = str(results_dir.joinpath('training_config.yaml'))
    logging.info(f"Copying training config to results dir: {train_cfg_fn}")
    shutil.copy(opts.train_config, train_cfg_fn)

    results_dir = run_training(config, results_dir)

    logging.info(f"Finished Training in {(time.time_ns() - start_t) * 1e-9} seconds!")
