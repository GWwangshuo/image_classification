import argparse
from pathlib import Path

from tqdm import tqdm
from PIL import Image
from torchvision import datasets


DATASET_LIST = [
    "CIFAR10",
    "CIFAR100",
    "MNIST",
]


def get_parser():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--root",
        default="../data/",
        type=str, help="download location")
    arg("--dataset",
        default="MNIST",
        type=str, help="dataset name: [CIFAR10, MNIST]")
    return parser


def build_dataset(dataset_name, root):
    if dataset_name in DATASET_LIST:
        train_dataset = getattr(datasets, dataset_name)(
            root=root, train=True, download=True)
        test_dataset = getattr(datasets, dataset_name)(
            root=root, train=False, download=True)
        return (train_dataset, test_dataset)
    raise Exception("not found dataset.")


def save_data(dataset, root_path, mode):
    mode_root_path = root_path / mode
    mode_root_path.mkdir(parents=True, exist_ok=True)
    for idx in tqdm(range(len(dataset))):
        data, target = dataset[idx]
        target_path = mode_root_path / str(target)
        if not target_path.exists():
            target_path.mkdir(parents=True, exist_ok=True)
        data.save(str(target_path / "{}.png".format(idx)))


def main(args):
    root_path = Path(args.root)
    dataset_path = root_path / args.dataset
    train_dataset, test_dataset = build_dataset(
        args.dataset,
        str(root_path))
    # save train data
    save_data(train_dataset, dataset_path, "train")
    save_data(test_dataset, dataset_path, "test")


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
