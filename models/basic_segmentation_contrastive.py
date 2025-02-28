import torch
import torch.nn as nn
from typing import Tuple

from pytorch_lightning.loggers import WandbLogger
import wandb

from data_loader import get_domains
from .basic_segmentation import BasicSegmentationModel
from .losses import NTXentLossSegm

class BasicSegmentationContrastModel(BasicSegmentationModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()  # makes hparams available
        self.layers = [int(l) for l in self.hparams.layers.split('-')]
        self.hidden_dim = 256

        self.test_domains, _ = get_domains(self.hparams.test_domains)
        self.val_domains, _ = get_domains(self.hparams.val_domains)

        if self.hparams.loss_weight is None:
            self.weights = None
        else:
            self.weights = torch.Tensor([self.hparams.loss_weight]).cuda()

        self._test_dice_structures = {}
                
        self.set_pheads_and_losses()
        
        self.automatic_optimization = False

    def set_pheads_and_losses(self):
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        if self.hparams.loss_contrast == 'NTXentLossSegm':
            self._loss_contrast = NTXentLossSegm(memory_bank_size=0, 
                                temperature=self.hparams.temperature,
                                gather_distributed=True)
            self.projection_head = torch.nn.Sequential(*[nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1), 
                                nn.BatchNorm2d(self.hidden_dim),
                            nn.ReLU(),
                            nn.Conv2d(self.hidden_dim, self.hparams.out_dim, kernel_size=1), 
                                nn.BatchNorm2d(self.hparams.out_dim)])
        else:
            raise NotImplementedError, f"Loss contrast {self.hparams.loss_contrast} not implemented"

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:

        optimizer = self.optimizers()
        optimizer.zero_grad()
        
        (x, x_style), target = batch
        output, latent1 = self.forward(x)
        
        output_style, latent2 = self.forward(x_style)

        loss_source = 0
        num_losses = 0
        if self.hparams.loss_source > 0:
            loss_source = self._loss(output, target) 
            self.log(f"Loss-segmentation-source/train", loss_source.item(), prog_bar=False)
            num_losses += 1

        loss_style = 0
        if self.hparams.loss_style > 0:            
            loss_style = self._loss(output_style, target)
            self.log(f"Loss-segmentation-style/train", loss_style.item(), prog_bar=False)
            num_losses += 1

        loss_semg = (self.hparams.loss_source * loss_source + self.hparams.loss_style * loss_style)/num_losses
        self.log(f"Loss-segmentation/train", loss_semg.item(), prog_bar=False)

        loss_contrast = self._loss_contrast(latent2, latent1)
        self.log(f"Loss-contrastive/train", loss_contrast.item(), prog_bar=False)

        loss = self.hparams.loss_contrast_weight * loss_contrast + loss_semg
        self.manual_backward(loss)
        
        # optimizer step
        optimizer.step()
        
        # scheduler step
        sch = self.lr_schedulers()
        if (batch_idx + 1) % 1 == 0:
            sch.step()
        
        self.log(f"Loss/train", loss.item(), prog_bar=False)
        
        if self.current_epoch == 0:
            self.log_image_with_target_prediction_loss(
                output, x, target, f"Training/images"
            )
            self.log_image_with_target_prediction_loss(
                output_style, x_style.detach(), target, f"Training/images_style"
            )

        ## uncomment for debugging
        # if self.global_step == 0:
        #     self.log_image_with_target_prediction_loss_pair(
        #         output, x, x_style, target, f"Training-pair"
        #     )

        # return loss

    def log_image_with_target_prediction_loss_pair(
        self,
        outputs: torch.Tensor,
        imgs: torch.Tensor,
        imgs_style: torch.Tensor,
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
        for idx, (img, img_style, target, prediction) in enumerate(
            zip(imgs, imgs_style, targets, predictions)
        ):
            img_to_plot = img.cpu().permute(1, 2, 0)
            img_to_plot_style = img_style.cpu().permute(1, 2, 0)

            wandb.log(
                {
                    f"{name}": wandb.Image(
                        img_to_plot.numpy(),
                        masks={
                            "ground_truth": {
                                "mask_data": target.cpu().numpy(),
                                "class_labels": class_labels,
                            },
                        },
                    ),
                    f"{name}-style": wandb.Image(
                        img_to_plot_style.numpy(),
                        masks={
                            "ground_truth": {
                                "mask_data": target.cpu().numpy(),
                                "class_labels": class_labels,
                            },
                        },
                    ),
                }
            )

            if idx >= 31:
                break