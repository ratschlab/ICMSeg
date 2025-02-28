import os
import torch
import pickle
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, Callback, ModelCheckpoint
from lightning_fabric.utilities.seed import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import LearningRateMonitor

import time
import pandas as pd
import pprint

from utils.options import make_task_name
from models.model_init import model_init

from data_loader import CMLDataModule, setup_labeled_dataloader
import wandb

from torch.utils.data import DataLoader

from utils.evaluations import simple_test, get_embeddings

from typing import Dict, Any


class ICMSegTrainer:
    """
    Generic class for operations on the model.
    """

    def __init__(self, hparams: Dict[str, Any]) -> None:
        seed_everything(hparams["seed"])

        self.gpu = torch.cuda.device_count()

        self.params = hparams

        self.__create_task_name()

        self.data_module = CMLDataModule(self.params)
        
        # self.dataloader_labeled = setup_labeled_dataloader(self.params)
        self.params["num_classes"] = self.data_module.num_classes

        if self.params["continue"]:
            self.params["ckpt_file"] = self.__get_checkpoint_name(
                self.params["meta_folder"]
            )

        self.model = model_init(self.params)
        self.set_up_trainer()

    def __get_checkpoint_name(self, parent_folder: str) -> str:
        ts = 0
        latest_ckpt = None
        for folder in [f for f in os.listdir(parent_folder) if "-" in f]:
            path = os.path.join(parent_folder, folder)
            for ckpt in [f for f in os.listdir(path) if ".ckpt" in f]:
                ckpt_path = os.path.join(path, ckpt)
                filename_ts = os.path.getmtime(ckpt_path)
                print(filename_ts, ckpt_path)
                if filename_ts > ts:
                    latest_ckpt = ckpt_path
                    ts = filename_ts

        print(f"Found checkpoint:\n{latest_ckpt}\n")

        return latest_ckpt

    def __create_task_name(self) -> None:
        """
        Automatically creates a name for the task and the output folders from the key
        parameters of the model.
        """

        self.params["meta_folder"], self.params["wandb_task"] = make_task_name(
            self.params
        )

        self.params[
            "meta_folder"
        ] = f"{self.params['path_out']}/{self.params['meta_folder']}"

        self.params[
            "full_path_out"
        ] = f"{self.params['path_out']}/{self.params['wandb_task']}"

        if not os.path.exists(self.params["full_path_out"]):
            os.makedirs(self.params["full_path_out"])

        print("\n" + "".join(["="] * 150))
        print("Saving results to:", self.params["full_path_out"])
        print("".join(["="] * 150), "\n")

        with open(f'{self.params["full_path_out"]}/params.pickle', "wb") as handle:
            pickle.dump(self.params, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(f'{self.params["full_path_out"]}/params.txt', "w") as log_file:
                pprint.pprint(self.params, log_file)

    def set_up_trainer(self) -> None:
        """
        Sets up pytorch lightening trainer for the model.
        """
        self.lr_monitor = LearningRateMonitor(logging_interval="step")

        self.set_logger()

        callbacks = [
            LearningRateMonitor(logging_interval='step'),
            TQDMProgressBar(refresh_rate=0),
            ModelCheckpoint(
                monitor=f'Val ({self.params["val_domains"]})/dice_foreground',
                mode="min",
                save_top_k=1,
                dirpath=self.params["full_path_out"],
                filename='best'
            ),
        ]

        if self.params["early_stop"] > 0:
            early_stop_callback = EarlyStopping(
                monitor="Loss/val",
                min_delta=0.00,
                patience=self.params["early_stop"],
                verbose=False,
                mode="min",
            )
            callbacks.append(early_stop_callback)

        if self.gpu > 1:
            print('Using distributed training.')
            self.trainer = pl.Trainer(devices=self.gpu,
                                    accelerator='gpu',
                                    max_epochs=self.params['num_epochs'],
                                    # max_epochs=-1,
                                    # max_steps=self.params['num_steps'],
                                    profiler=None,
                                    callbacks=callbacks,
                                    logger=self.logger,
                                    check_val_every_n_epoch=self.params['check_val_every_n_epoch'],
                                    # check_val_every_n_epoch=None,
                                    # val_check_interval=self.params['check_val_every_n_step'],
                                    precision=self.params['precision'],
                                    strategy="ddp",
                                    sync_batchnorm=True,
                                    use_distributed_sampler=True,
                                    )
        else:
            self.trainer = pl.Trainer(devices=self.gpu,
                                    accelerator='gpu',
                                    max_epochs=self.params['num_epochs'],
                                    # max_epochs=-1,
                                    # max_steps=self.params['num_steps'],
                                    profiler=None,
                                    callbacks=callbacks,
                                    logger=self.logger,
                                    check_val_every_n_epoch=self.params['check_val_every_n_epoch'],
                                    # check_val_every_n_epoch=None,
                                    # val_check_interval=self.params['check_val_every_n_step'],
                                    precision=self.params['precision'],
                                    )


    def set_logger(self):
        """
        Set up wandb logger.
        """
        self.logger = WandbLogger(
            project=self.params["project_name"],
            name=self.params["wandb_task"],
            # save_dir=self.params["full_path_out"],
            settings=wandb.Settings(init_timeout=600, _service_wait=600),
        )

    def train(self):
        """
        Calls the lightning trainer, then tests and saves mode.
        """
        self.model.train()
        self.trainer.fit(self.model, datamodule=self.data_module)
        self.trainer.save_checkpoint(f"{self.params['full_path_out']}/last.ckpt")

        self.test()

        print("Saved results to:\n", self.params["full_path_out"])

    def test(self, plot_latent: bool = True):
        """
        Performs the test and saves the model.
        """
        ts = time.time()

        self.model.eval()

        self.data_module.setup(stage="test")
        self.trainer.test(self.model, self.data_module)

        time_min = (time.time() - ts) / 60
        print(f"Time (min) = {time_min:.1f}")
