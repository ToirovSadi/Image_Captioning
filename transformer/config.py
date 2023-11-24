# General
SEED = 42
MAX_SENT_SIZE = 20

# Training Stage
BATCH_SIZE = 128
NUM_WORKERS = 8
MAX_EPOCHS = -1
LR = 1e-3
LABEL_SMOOTHING = 0
WEIGHT_DECAY = 0

MODEL_RESUME = './transformer/logs/2023-11-24_04-07-11/epoch=3-step=12712.ckpt'
PREPTRAIN_PATH = './transformer/logs/2023-11-24_04-07-11/epoch=3-step=12712.ckpt'
LOGS_DIR = "./transformer/logs/"

## Model Hyperparameters
# Encoder
input_dim = 3
hidden_dim = 256
dropout = 0.3

# Decoder
num_heads = 4
num_layer = 2
output_dim = None # will be defined in train.py
embed_dim = 256


## files
DEFAULT_ROOT_DIR = 'transformer/'
TRAIN_DATASET = '/datasets/coco/train2014/'
TRAIN_CAPTION = '/datasets/coco/annotations/captions_train2014.json'

VAL_DATASET = '/datasets/coco/val2014/'
VAL_CAPTION = '/datasets/coco/annotations/captions_val2014.json'