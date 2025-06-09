"""
This module provides some helper methods for getting configuration data.
The configuration data can be used with all defaults or it can be customized
on a user-basis.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import torch


@dataclass
class UserConfig:
    """
    This class contains fields that may be changed
    by anyone running these notebooks. Nothing here
    adversly affects the model, except for the batch
    size and the number of epochs to train it for.
    """
    # For initializing random number generators
    seed: int = 0

    # Where to save the best ONNX model
    best_model_path: str = 'best_model.onnx'

    # The batch size to use for training and validation
    batch_size: int = 4

    # The name of the model
    model_name: str = 'neuron_unet'

    # The version of the model
    model_version: int = 1

    # Where to generate imagery and where to load it from during training
    imagery_dir: str = '../data/imagery'

    # Where to find the SWC models during data generation
    swc_dir: str = '../data/swc'

    # The directory containing the web files to deploy alongside the ONNX model
    demo_dir: str = '../demo'

    # Whether or not to cache the datasets in memory (only use if you have a 32 GB of RAM)
    cache_data: bool = False

    # The number of epochs to train the model
    num_epochs: int = 100

    # The name to give the MLFlow experiment
    mlflow_experiment_name: str = 'neuron_unet'

    # The MLFlow tracking URI to use (automatically determined if left blank)
    mlflow_tracking_uri: str = ''

    # Where to put the tensorboard summaries (automatically determined if left blank)
    tensorboard_dir: str = ''

    # What libtorch device to use (automatically determined if left blank)
    device: str = ''


def open_user_config(path: str = 'user_config.json') -> UserConfig:
    """
    Opens a user configuration file, if it exists.
    If it does not exist, a config object is returned with all the defaults.

    :param path: The path to the user configuration file.
    """
    p = Path(path)
    cfg = UserConfig()
    if p.exists():
        with open(path, 'r') as f:
            cfg = UserConfig(**json.load(f))

    # auto detect default
    if cfg.mlflow_tracking_uri == '':
        if Path('/phoenix/mlflow').exists():
            cfg.mlflow_tracking_uri = '/phoenix/mlflow'
        else:
            cfg.mlflow_tracking_uri = 'mlruns'

    # auto detect default
    if cfg.tensorboard_dir == '':
        if Path('/phoenix/tensorboard').exists():
            cfg.tensorboard_dir = '/phoenix/tensorboard/tensorlogs'
        else:
            cfg.tensorboard_dir = 'runs'

    # auto detect default
    if cfg.device == '':
        if torch.cuda.is_available():
            cfg.device = 'cuda:0'
        else:
            cfg.device = 'cpu'

    return cfg
