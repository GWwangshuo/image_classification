import sys
import argparse

import yaml
from addict import Dict


def get_parser():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config",
        default="./configs/sample.yaml",
        type=str, help="config path")
    # arg("-k", "--k_fold_num",
    #     default=0, type=int,
    #     help="validation index for cross-validation")
    return parser


def read_yaml(path):
    try:
        with open(path) as file:
            obj = yaml.safe_load(file)
    except Exception as e:
        print('Exception occurred while loading YAML...', file=sys.stderr)
        print(e, file=sys.stderr)
        sys.exit(1)
    return Dict(obj)
