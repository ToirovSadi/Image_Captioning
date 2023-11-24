## import all libraries
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import math
import lightning as L
import os
os.chdir("..") # go to root dir

import config
from data import COCODataset
from data import collate_batch
from model import Transformer
from utils import generate_ckpt_name

# ImageNet mean and std
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

image_size = (224, 224)
train_transform = transforms.Compose([
    transforms.Resize((232, 232)),
    transforms.RandomCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])
test_transform = transforms.Compose([
    transforms.Resize((232, 232)),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

## Get the Datasets
train_dataset = COCODataset(
    root=config.TRAIN_DATASET,
    annotation_path=config.TRAIN_CAPTION,
    train=True,
    image_transform=train_transform,
    remove_idx=True,
    vocab_min_freq=5,
)
print("Train Dataset Size: ", len(train_dataset))
print("Vocab Size:", len(train_dataset.vocab))

val_dataset = COCODataset(
    root=config.VAL_DATASET,
    annotation_path=config.VAL_CAPTION,
    vocab=train_dataset.vocab,
    train=True,
    image_transform=test_transform,
    take_first=10_000,
    remove_idx=False,
    return_all_captions=True,
)
print("Val Dataset Size: ", len(val_dataset))

## Get the DataLoaders
train_dataloader = DataLoader(
    train_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=config.NUM_WORKERS,
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
    collate_fn=collate_batch,
)

## Make training determenistic
L.seed_everything(config.SEED)
torch.set_float32_matmul_precision("medium")

## Define the model
config.output_dim = len(train_dataset.vocab)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if config.PREPTRAIN_PATH is None:
    model = Transformer(
        input_dim=config.input_dim,
        embed_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
        output_dim=config.output_dim,
        num_heads=config.num_heads,
        num_layer=config.num_layer,
        vocab=train_dataset.vocab,
        dropout=config.dropout,
        max_length=config.MAX_SENT_SIZE,
        device=device,
    )
else:
    print("loading model from checkpoint")
    model = Transformer.load_from_checkpoint(
        config.PREPTRAIN_PATH,
        input_dim=config.input_dim,
        embed_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
        output_dim=config.output_dim,
        num_heads=config.num_heads,
        num_layer=config.num_layer,
        vocab=train_dataset.vocab,
        dropout=config.dropout,
        max_length=config.MAX_SENT_SIZE,
        device=device,
    )

    
# new_dir = generate_ckpt_name()
new_dir = '2023-11-23_13-19-38' #don't create new dir
log_dir = os.path.join(config.LOGS_DIR, new_dir)

# get logger and callbacks
tb_logger = L.pytorch.loggers.TensorBoardLogger(save_dir=log_dir)
model_ckpt = L.pytorch.callbacks.ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    dirpath=log_dir,
    every_n_epochs=1,
    save_top_k=-1,
)
ACCUMULATE_MAP = {
    0: math.ceil(128 / config.BATCH_SIZE),
    10: math.ceil(256 / config.BATCH_SIZE),
    15: math.ceil(512 / config.BATCH_SIZE),
    25: math.ceil(1024 / config.BATCH_SIZE),
}
grad_accum = L.pytorch.callbacks.GradientAccumulationScheduler(scheduling=ACCUMULATE_MAP)

trainer_cfg = {
    "accelerator": "gpu",
    "logger": tb_logger,
    # "fast_dev_run": True,
    "devices": 1,
    # "precision": "16-mixed",
    "max_epochs": config.MAX_EPOCHS,
    "log_every_n_steps": 100,
    "gradient_clip_val": 2.0,
    "check_val_every_n_epoch": 1,
    "callbacks": [model_ckpt, grad_accum],
    "default_root_dir": config.DEFAULT_ROOT_DIR,
    # "accumulate_grad_batches": 2,
}
trainer = L.Trainer(**trainer_cfg)

# fit the model
trainer.fit(
    model,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
    ckpt_path=config.MODEL_RESUME
)