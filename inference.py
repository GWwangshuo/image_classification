from pathlib import Path

import cv2
import torch
from addict import Dict
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score)
from pytorch_lightning.trainer import seed_everything

from pl_module import LightningModuleInference
from dataset import ImageDataset
from utils.factory import get_transform
from utils.util import get_parser, read_yaml


TARGET_EXTENSION = [
    ".jpg",
    ".png",
    ".tiff"
]

TEST_ROOT_PATH = "../data/CIFAR10/test"
OUTPUT_PATH = "../data/CIFAR10/output"
CHECKPOINT = "./mlruns/0/4dbc974a3b814a66a50ff35cbf2614ca/artifacts/epoch=28-val_loss_mean=0.486-val_acc=0.842.pth.ckpt"


def inference(cfg):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    # define model
    pl_module = LightningModuleInference.load_from_checkpoint(
        CHECKPOINT, cfg=cfg
    ).eval().to(device)

    # define transform
    transform = get_transform(cfg.transform.val)

    # define Dataset and Dataloader
    dataset_cfg = Dict({"root": TEST_ROOT_PATH})
    dataset = ImageDataset(dataset_cfg, transform)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.data.dataloader.batch_size // 2,
        num_workers=cfg.data.dataloader.num_workers,
        shuffle=cfg.data.dataloader.shuffle,
    )

    output_path = Path(OUTPUT_PATH)
    output_path.mkdir(parents=True, exist_ok=True)

    labels, predicts = [], []
    for imgs, targets in tqdm(dataloader):
        input = imgs.to(device, non_blocking=True)
        outputs = pl_module(input)
        _, predicted_indexes = torch.max(outputs.data, 1)
        targets = targets.cpu().numpy().tolist()
        predict_idx = predicted_indexes.cpu().numpy().tolist()
        labels.extend(targets)
        predicts.extend(predict_idx)
    calc_eval(labels, predicts)


def calc_eval(labels, predicts):
    matrix = confusion_matrix(labels, predicts)
    acc = accuracy_score(labels, predicts)
    precision = precision_score(labels, predicts, average="weighted")
    recall = recall_score(labels, predicts, average="weighted")
    f1 = f1_score(labels, predicts, average="weighted")
    print(matrix)
    print("acc      :", acc)
    print("precision:", precision)
    print("recall   :", recall)
    print("f1       :", f1)


if __name__ == "__main__":
    args = get_parser().parse_args()

    cfg = read_yaml(path=args.config)

    seed_everything(seed=cfg.general.seed)
    inference(cfg)
