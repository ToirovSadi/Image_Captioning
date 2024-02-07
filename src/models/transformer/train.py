"""
This script trains a Transformer model for image captioning using the specified configuration file.
"""

## import all libraries
import torch
import math
import lightning as L
import sys
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
print("Root Dir:", root_dir)
sys.path.insert(0, root_dir)
os.chdir(root_dir)

from model import Transformer
from utils import get_config, get_datasets, get_dataloaders

import argparse
p = argparse.ArgumentParser()
p.add_argument('--config_file', type=str, default='src/models/transformer/baseline_config.yaml')
args = p.parse_args()

config = get_config(args.config_file)
train_dataset, val_dataset = get_datasets(config)
train_dataloader, val_dataloader = get_dataloaders(train_dataset, val_dataset, config)

## Make training deterministic
L.seed_everything(config['seed'])
torch.set_float32_matmul_precision("medium")

## Define the model
output_dim = len(train_dataset.vocab)
config['model']['output_dim'] = output_dim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if config['pretrain_path']:
    print("loading model from checkpoint")
    model = Transformer.load_from_checkpoint(config['pretrain_path'], device=device)
else:
    model = Transformer(
        vocab=train_dataset.vocab,
        config=config,
        device=device
    )

log_dir = os.path.join(config['logs_dir'], config['model']['model_name'])

# get logger and callbacks
tb_logger = L.pytorch.loggers.TensorBoardLogger(save_dir=log_dir)
model_ckpt = L.pytorch.callbacks.ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    dirpath=log_dir,
    every_n_epochs=1,
    save_top_k=-1,
)

trainer_cfg = {
    "accelerator": "gpu",
    "logger": tb_logger,
    "devices": 1,
    # "fast_dev_run": True,
    "num_sanity_val_steps": 0,
    "max_epochs": config['train']['max_epochs'],
    "max_steps": config['train']['max_steps'],
    "log_every_n_steps": 100, # TODO
    "gradient_clip_val": 2.0,
    "check_val_every_n_epoch": 1,
    "callbacks": [model_ckpt],
    "default_root_dir": config['default_root_dir'],
}
trainer = L.Trainer(**trainer_cfg)

# fit the model
trainer.fit(
    model,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
    ckpt_path=config['model_resume'] if config['model_resume'] else None,
)