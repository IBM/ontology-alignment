import logging
import random
import re

import numpy as np
import torch


def match_url(input: str):
    return re.findall(r'_https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)', input)


def set_random_seeds(seed: int):
    logging.info(f"Setting random, numpy, torch, and torch.cuda random seeds to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.random.manual_seed(seed)
