import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from typing import List, Optional, Tuple

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig  ## comes up along with hydra
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase

from src import utils


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="test_target.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    print(cfg)
    dm_func = hydra.utils.instantiate(cfg.datamodule)

    print(dm_func)

    print(dm_func())
if __name__ == "__main__":
    main()
