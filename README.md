# Ontology Alignment

A tool for matching or aligning of ontologies with transformer language models.

## Install

The code in this repo is written in `Python v3.8.5` and uses `poetry` as dependency management tool.

1. Create the basic `conda` environment: `conda env create -f environment.yml`
2. Install python dependencies: `poetry install`

## Code Structure Overview

### configurations
- `/configurations/api`: configuration for the inference API to run MELT Evaluations

- `/configurations/eval_run_cfgs`: configurations used to run evaluation experiments with pretrained models. Used with `/src/scripts/run_oa_eval.py`

- `/configurations/pseudo_sentence_generator`: configurations for the PseudoSentenceGenerator

- `/configurations/train_cfgs`: configurations to train NLFOA models. Used with `/src/scripts/train_nlfoa.py`


### datasets
- `/datasets/veealign`: contains 'raw' veealign datasets, i.e., ontologies in OWL and alignments in RDF

- `/datasets/oaei`: contains 'raw' oaei datasets, i.e., ontologies in OWL and alignments in RDF

- `/datasets/nlfoa`: contains 'transformed' datasets used to train models. The datasets are in DataFrames and can be read with `pd.read_parquet()`. The dataframes contain precomputed pseudo sentences from a default PSG Config, a pair of URIs of two aligned entities and a similarity score that is either `1.0` or `0.0`.

### oaMeltEval
- `/oaMeltEval/src`: Java code to run evaluations with the MELT Framework

### src
- `/src/api`: code for the inference API to run MELT evaluations

- `/src/data/datasets/dynamic_pseudo_sentence_alignment_dataset.py`: PyTorch dataset to dynamically generate pseudo sentences on the fly during training

- `/src/data/model`: holds the abstract ontology data model that it used throughout the project

- `/src/data/preprocessing`: parsers for OWL Ontologies, Reference Alignments in RDF, and a minimalistic JSONL parser. All the parsers read the respective files and build an abstract datamodel, i.e., the one defined in `/data/model`.

- `/src/data/alignment_format.py`: dummy script to generate Alignment Format files from alignment instances (used within OAEI and MELT etc.)

- `/src/model`: (simple) code to create, train, load NLFOA models using the SentenceTransformer library

- `/src/scripts`: various scripts (most of which are not really used anymore by myself). 

- `/src/scripts/train_nlfoa.py`: script to train NLFOA models. This reads training configurations as in the `/configurations/train_cfgs` directory

- `/src/scripts/run_oa_eval.py`: script to evaluate NLFOA models on a reference alignments file (not MELT, not custom splits!). This reads training configurations as in the `/configurations/train_cfgs` directory

- `/src/test`: old tests for the parsers

- `/src/scoring.py`: code to compute scores from results of the `run_oa_eval.py` script using sklearn

### scratches
contains IPYNBs used by me to test code and run some experiments etc.

## Eval Configuration Details

### Folders and File Naming conventions
Files like in the `configuration/eval_run_cfgs` directory. Basically each file there describes a evaluation experiment using a trained model on some referene alignments.
In the direcotry are many of these configs in different subfolders named after which reference alignments and/or model are used in the experiment.

The single config files also follow a naming convention. E.g.: The filename 

`nlfoa_va-combo-tn_dynamic-psg_params_name_desc_1hop_props_no_domain_and_range_english_no_special_tokens_sts-all-mpnet-base-v2.yaml` 

says:
- a nlfoa model trained on the VA Combo dataset (`nlfoa_va-combo`) with twice the number of negative samples than positive samples is used (`tn`)
- the pseudo sentence configuration `name_desc_1hop_props_no_domain_and_range_english` was used during training. This file can be found in the pseudo sentence config directory
- the tokenizer of the model was NOT exptended by special tokens (`no_special_tokens`)
- the NLFOA was initialized with an `all-mpnet-v2-base` STS model


### Config Fields
Each file looks like the following:

```yaml
eval_run_config:
  io:
    onto_a: relative path to the first ontology in the reference alignments
    onto_b: relative path to the second ontology in the reference alignments
    ref_alignments: path to the ref alignments
    ontology_parser: the name of the parser used to read the ontology files
    psg_config: relative path to the PSG Config used to generate pseudo sentences
    onto_cache_dir: relative path to the directory where ontologies are cached (so that they dont need to be parsed again)
    ps_cache_dir: relative path to the directory where pseudo sentences for the two ontologies are cached so that they dont need to be generated again
    results_base_dir: relative path to the directory where the results are stored

  model:
    type: type of the model. only use nlfoa
    pertrained_name_or_path: relative path to the model
    device: device on which the model gets loaded

  similarity: similarity method used to compute the sim. between two pseudo sentence embeddings. either cosine or dot
  threshold: for evaluation. If the computed similariy between two concepts is less than the threshold, it is considered as NO_MATCH
```


## Pseudo Sentence Generator Configs
Files like in the `configuration/pseudo_sentence_generator` directory. Basically each file there describes what to include in a pseudo sentence for a given concept or relation.

The file names give hints about what is included and the directory where the files are located doesnt matter.

The fields of the configs are self-explanatory.

##  Training Configs
Files like in the `configuration/train_cfgs` directory. Basically each file there describes the training process for an NLFOA model.

The files and sub-directories follow the same naming convention as the evaluation scripts.


### Config Fields

There are two different config types. One uses dynamic pseudo sentences, i.e., they are generated on the fly during training and can therefore be shuffled or modified. For these configs only the URIs are used from the train/test/val dataframes. The other config type uses static, i.e., precomputed, pseudo sentence from the train/test/val dataframes.

It is recommended to only use Dynamic Pseudo Sentence training since the overhead of generating the PS dynamically is relatively small.

**Dynamic PseudoSentences**
The files all look like this. 

```yaml
nlfoa_training_config:
  random_seed: random seed that is used for rando, numpy, torch, and cuda
  io:
    train_set: relative path to the training dataframe
    test_set: relative path to the test dataframe
    val_set: relative path to the val dataframe
    results_dir: relative path where the model etc are stored
  model:
    sts_model: name of the sts model that is used as init. name has to exist on huggingface
    add_special_tokens: if true add special tokens from the PSG config to the tokenizer
    pooling_mode: Should be 'mean' and is the strategy for the pooling layer to generate a single embedding for a sentence
    device: cuda:0
  training:
    train_batch_size: 32
    test_batch_size: 32
    val_batch_size: 32
    shuffle_dataloaders: true
    dynamic_pseudo_sentences:
      ontologies_glob: relative path with glob pattern to find all OWL files that are referenced by the URIs in the dataframes.
      psg_config: relative path to the PSG Config used to generate the pseudo sents
      shuffle_ps: if True the PS will get shuffled
    epochs: 20
    evaluation_steps: 1000
    checkpoint_save_steps: 1000
    loss: CosineSimilarityLoss
    warmup_scheduler: WarmupLinear
    warmup_steps: 0.1
    optimizer: AdamW
    lr: 2.0e-05
    weight_decay: 0.01
    evaluator: EmbeddingSimilarityEvaluator
    use_amp: true
```


