import torch
import numpy as np
import pytorch_lightning as pl

from dataset import ImageDataset
from torch.utils.data import DataLoader

from utils.factory import get_model, get_loss, get_optimizer

from utils.factory import get_transform


class LightningModuleReg(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.net = get_model(self.cfg.model)
        self.loss = get_loss(self.cfg.loss)

        self.mean = [0.4914, 0.4822, 0.4465]
        self.std = [0.2023, 0.1994, 0.2010]

        self.transforms = {
            "train": get_transform(cfg.transform.train),
            "val": get_transform(cfg.transform.val),
        }


    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        correct, labels_size = self.__calc_correct(y_hat, y)
        return {
            'val_loss': loss,
            "val_corr": correct,
            "labels_size": labels_size}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        labels_size = np.array([x['labels_size'] for x in outputs]).sum()
        val_acc = np.array([x['val_corr']
                            for x in outputs]).sum() / labels_size
        metrics = {'val_loss_mean': float(val_loss_mean.cpu().numpy()),
                   'val_acc': float(val_acc)}
        self.log_dict(metrics)

    def configure_optimizers(self):
        optimizer, scheduler = get_optimizer(
            cfg=self.cfg.optimizer, model_params=self.net.parameters())
        return [optimizer], [scheduler]

    def train_dataloader(self):
        train_dataset = ImageDataset(
            dataset_cfg=self.cfg,
            transform=self.transforms["train"],
            mode="train"
        )
        train_dataloader = DataLoader(train_dataset, batch_size=self.cfg.data.dataloader.batch_size,
                                      num_workers=self.cfg.data.dataloader.num_workers, shuffle=True,
                                      drop_last=True, pin_memory=True)
        return train_dataloader

    def val_dataloader(self):
        val_dataset = ImageDataset(
            dataset_cfg=self.cfg,
            transform=self.transforms["val"],
            mode="val"
        )
        val_dataloader = DataLoader(val_dataset, batch_size=self.cfg.data.dataloader.batch_size,
                                    num_workers=self.cfg.data.dataloader.num_workers,
                                    pin_memory=True)
        return val_dataloader

    def test_dataloader(self):
        return self.val_dataloader()

    # # learning rate warm-up
    # def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, second_order_closure=None, on_tpu=False,
    #                    using_native_amp=False, using_lbfgs=False):
    #     # warm up lr
    #     if self.trainer.global_step < 500:
    #         lr_scale = min(1., float(self.trainer.global_step + 1) / 500.)
    #         for pg in optimizer.param_groups:
    #             pg['lr'] = lr_scale * self.hparams.learning_rate
    #
    #     # update params
    #     optimizer.step()
    #     optimizer.zero_grad()

    def __calc_correct(self, outputs, labels):
        _, predicted_indexes = torch.max(outputs.data, 1)
        labels_size = labels.size(0)
        correct = (predicted_indexes == labels).sum().item()
        return correct, labels_size


class LightningModuleInference(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.net = get_model(self.cfg.model)

    def forward(self, x):
        return self.net(x)
