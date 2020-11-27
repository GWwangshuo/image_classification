from urllib.parse import urlparse

import mlflow
from pytorch_lightning.trainer import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only

from pl_module import LightningModuleReg
from utils.util import get_parser, read_yaml


class MyLogger(LightningLoggerBase):
    def __init__(self):
        super().__init__()

    def name(self):
        return "MyLogger"

    @property
    def version(self):
        return mlflow.active_run().info.run_id

    def experiment(self):
        return mlflow

    @rank_zero_only
    def log_hyperparams(self, params):
        super().log_hyperparams(params)

    @rank_zero_only
    def log_metrics(self, metrics, step):
        mlflow.log_metrics(metrics, step)


def train(cfg):
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.callback.checkpoint.monitor,
        save_last=cfg.callback.checkpoint.save_last,
        save_top_k=cfg.callback.checkpoint.save_top_k,
        mode=cfg.callback.checkpoint.mode,
        save_weights_only=cfg.callback.checkpoint.save_weights_only,
        filename=cfg.callback.checkpoint.filename,
        dirpath=urlparse(mlflow.get_artifact_uri()).path
    )

    trainer = Trainer(
        checkpoint_callback=True,
        callbacks=[checkpoint_callback],
        logger=MyLogger(),
        accelerator=cfg.general.accelerator,
        max_epochs=cfg.general.epoch,
        gpus=cfg.general.gpus,
        precision=cfg.general.precision,
        amp_backend=cfg.general.amp_backend,
        amp_level=cfg.general.amp_level,
        accumulate_grad_batches=cfg.general.acc_grad,
        fast_dev_run=True if cfg.general.debug else False,
        limit_train_batches=0.25 if cfg.general.debug else 1.0,
        limit_val_batches=0.25 if cfg.general.debug else 1.0,
        resume_from_checkpoint=cfg.general.resume_from_checkpoint,
    )

    pl_module = LightningModuleReg(cfg=cfg)
    trainer.fit(model=pl_module)


if __name__ == "__main__":
    args = get_parser().parse_args()

    cfg = read_yaml(path=args.config)
    # cfg.data.dataset.val_k = args.k_fold_num

    seed_everything(seed=cfg.general.seed)

    mlflow.pytorch.autolog()
    mlflow.log_artifact(args.config)

    train(cfg)
