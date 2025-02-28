import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import wandb

from torchmetrics.functional.classification import dice

from .losses import DiceLoss
from .utils import cosine_annealing
from data_loader import get_domains

from typing import List, Tuple, Dict, Any


class BasicSegmentationModel(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()  # makes hparams available

        self.test_domains, _ = get_domains(self.hparams.test_domains)
        self.val_domains, _ = get_domains(self.hparams.val_domains)

        if self.hparams.loss_weight is None:
            self.weights = None
        else:
            self.weights = torch.Tensor([self.hparams.loss_weight]).cuda()

        self._test_dice_structures = {}
        
        self.automatic_optimization = False

    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], Dict[str, Any]]:
        # lr_factor = self.hparams.batch_size / 128
        learning_rate = self.hparams.learning_rate

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            # betas=(0.9, 0.98),
            weight_decay=self.hparams.weight_decay,
        )

        self.lr_scheduler = {}

        len_train_loader = (
            len(self.trainer.datamodule.train_dataset[-1]) // (self.hparams.batch_size * len(self.hparams.training_domains.split('+')))
        ) 

        # total_steps = len_train_loader * self.hparams.num_epochs
        total_steps = len_train_loader * 50
        # total_steps = self.hparams.num_steps

        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=total_steps, eta_min=1e-7
        )

        if self.hparams.lr_scheduler == "coswarmup":
            warm_up_iters = 10 * len_train_loader
            sch = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda step: cosine_annealing(
                    step,
                    total_steps,
                    1,  # since lr_lambda computes multiplicative factor
                    1e-6 / learning_rate,
                    warmup_steps=warm_up_iters,
                ),
            )

        elif self.hparams.lr_scheduler == "cosine":
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=total_steps, eta_min=1e-7
            )
   
        else:
            raise NotImplementedError

        self.lr_scheduler = {
            "scheduler": sch,
            "name": "learning rate",
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [self.lr_scheduler]

    def _step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, str], loss_name: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        batch_input, target, name = batch

        output = self.forward(batch_input)

        loss = self._loss(output, target)

        if loss_name != "test":
            self.log(f"Loss/{loss_name}", loss.item(), prog_bar=False)

        if self.current_epoch == 0:
            self.log_image_with_target_prediction_loss(
                output, batch_input, target, f"Training"
            ) 

        return output, target, loss, name

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        _, _, loss = self._step(batch, "train")

        return loss
        
    def log_image_with_target_prediction_loss(
        self,
        outputs: torch.Tensor,
        imgs: torch.Tensor,
        targets: torch.Tensor,
        name: str,
    ):
        # the following code logs images with targets, predictions and losses
        # to wandb. (the values are attached as captions)
        self.logger: WandbLogger

        predictions = outputs.softmax(dim=1).argmax(1).cpu()
        num_classes = self.hparams.num_classes

        class_labels = dict(
            zip(range(num_classes), [str(i) for i in range(num_classes)])
        )

        random_idx = np.random.choice(len(outputs), 10, replace=True)

        for idx in random_idx:
            img_to_plot = imgs[idx].cpu()
            target = targets[idx].cpu().numpy()
            prediction = predictions[idx].numpy()

            img_to_plot = img_to_plot.permute(1, 2, 0)
            wandb.log(
                {
                    f"{name}": wandb.Image(
                        img_to_plot.numpy(),
                        masks={
                            "predictions": {
                                "mask_data": prediction,
                                "class_labels": class_labels,
                            },
                            "ground_truth": {
                                "mask_data": target,
                                "class_labels": class_labels,
                            },
                        },
                    )
                }
            )

    def _loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.hparams.loss_name == "dice":
            return DiceLoss()(prediction, target)
        elif self.hparams.loss_name == "cross_entropy":
            return F.cross_entropy(prediction, target)
        elif self.hparams.loss_name == "dice_and_cross_entropy":
            return F.cross_entropy(prediction, target) + DiceLoss()(prediction, target)
        else:
            raise NotImplementedError

    def _metrics(
        self, output: torch.Tensor, target: torch.Tensor
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if torch.is_tensor(output):
            output, target = output.cpu(), target.cpu()

        prediction = output.softmax(dim=1).argmax(1)

        metrics = {}
        
        dice_structures = dice(prediction, 
                               target.int(), 
                               average='none', 
                               num_classes=self.hparams.num_classes)

        mask_names = [str(i) for i in range(output.shape[1])]

        metrics["dice_avg"] = dice_structures.nanmean().item()
        metrics["dice_foreground"] = dice_structures[1:].nanmean().item()
        dice_structures = dict(zip(mask_names, dice_structures.numpy()))

        return metrics, dice_structures

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> None:
        outputs, target, loss = self._step(batch, "val")

        name = f"{self.val_domains[0]}"

        batch_size = len(target)
        important_idx = 256 // batch_size
        if batch_idx == important_idx:
            inputs, _ = batch
            self.log_image_with_target_prediction_loss(
                outputs, inputs, target, f"val_images ({name})"
            )

        metrics, dice_structures = self._metrics(outputs.cpu(), target.cpu())
        for k in metrics:
            self.log(f"Val ({name})/{k}", metrics[k], prog_bar=True)

        print(name)        
        print(dice_structures)
        print('------------')

        for k in dice_structures:
            self.log(f"Val {name}/Dice - {k}", dice_structures[k], prog_bar=False)

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        outputs, target, _ = self._step(batch, "test")

        metrics, dice_structures = self._metrics(outputs, target)

        name = f"{self.test_domains[dataloader_idx]}"
        if not (name in self._test_dice_structures):
            self._test_dice_structures[name] = {}

        for m in dice_structures:
            if m in self._test_dice_structures[name]:
                self._test_dice_structures[name][m].append(dice_structures[m])
            else:
                self._test_dice_structures[name][m] = [dice_structures[m]]

        for m in metrics:
            self.log(f"Test ({name})/{m}", metrics[m], prog_bar=False)

        if batch_idx == 0:
            inputs, _ = batch
            self.log_image_with_target_prediction_loss(
                outputs, inputs, target, f"TEST ({self.test_domains[dataloader_idx]})"
            )

    def on_test_epoch_end(self) -> None:
        for name in self._test_dice_structures:
            print(f"Test per structure {name}")
            labels = []
            values = []
            for m in self._test_dice_structures[name]:
                dice_structures = np.array(self._test_dice_structures[name][m])
                print(
                    f"{m}:\t{dice_structures.mean():.3f} +/- {dice_structures.std():.3f}"
                )
                # self.log(f'Total Test ({name})/{m}', dice_structures.mean(), prog_bar=True)
                labels.append(m)
                values.append(dice_structures.mean())

            data = [[label, val] for (label, val) in zip(labels, values)]
            table = wandb.Table(data=data, columns=["label", "dice"])
            wandb.log(
                {
                    f"Test {name}": wandb.plot.bar(
                        table, "label", "dice", title=f"Total Test ({name})"
                    )
                }
            )