import os
from typing import List

import hydra
from omegaconf import DictConfig
from lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers.logger import Logger
import pandas as pd

from ai4eo_mapyourcity import utils

log = utils.get_logger(__name__)


def test(config: DictConfig) -> None:
    """Contains minimal example of the testing pipeline.
    Evaluates given checkpoint on a testset.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        None
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    # Convert relative ckpt path to absolute path if necessary
    if not os.path.isabs(config.ckpt_path):
        config.ckpt_path = os.path.join(hydra.utils.get_original_cwd(), config.ckpt_path)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    valid_dataloader = datamodule.valid_dataloader()
    test_dataloader = datamodule.test_dataloader()

    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)

    # Init lightning loggers
    logger: List[logger.Logger] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(config.trainer, logger=logger)

    # Log hyperparameters
    trainer.logger.log_hyperparams({"ckpt_path": config.ckpt_path})

    # save predictions for validation set
    log.info("Starting prediction!")
    trainer.predict(model=model, dataloaders=valid_dataloader, ckpt_path=ckpt_path)

    valid_predictions = pd.DataFrame(model.valid_predictions)
    valid_predictions.to_csv(f'valid_predictions_fold_{datamodule.dataset_options["fold"]}.csv', index=False)

    log.info("Starting testing!")
    trainer.test(model=model, dataloaders=test_dataloader, ckpt_path=ckpt_path)

    test_predictions = pd.DataFrame(model.test_predictions)
    test_predictions.to_csv(f'test_predictions_fold_{datamodule.dataset_options["fold"]}.csv', index=False)
