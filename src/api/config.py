import os
import logging
from pathlib import Path
from omegaconf import OmegaConf

logging.getLogger().setLevel(logging.INFO)

cfg_file = Path(os.environ.get('OASYS_API_CONFIG', 'configurations/api/config.yaml'))
assert cfg_file.exists() and cfg_file.is_file(), f"Cannot read API config from: {cfg_file}"

config = OmegaConf.load(cfg_file).api
logging.info(f"Using config: {cfg_file}")
