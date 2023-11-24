# General
SEED = 42
MAX_SENT_SIZE = 20

# Training Stage
BATCH_SIZE = 64
NUM_WORKERS = 8
MAX_EPOCHS = -1
LR = 1e-3
LABEL_SMOOTHING = 0
WEIGHT_DECAY = 0

MODEL_RESUME = './seq2seq/logs/2023-11-24_18-31-11/epoch=2-step=19065.ckpt'
PREPTRAIN_PATH = './seq2seq/logs/2023-11-24_18-31-11/epoch=2-step=19065.ckpt'
LOGS_DIR = "./seq2seq/logs/"

## Model Hyperparameters
# general hyper parameters
input_dim = 3 # input channels
embed_dim = 512
hidden_dim = 512
output_dim = None # will be defined in train.py
dropout = 0.3
num_layers = 2


## files
DEFAULT_ROOT_DIR = 'seq2seq/'
TRAIN_DATASET = '/datasets/coco/train2014/'
TRAIN_CAPTION = '/datasets/coco/annotations/captions_train2014.json'

VAL_DATASET = '/datasets/coco/val2014/'
VAL_CAPTION = '/datasets/coco/annotations/captions_val2014.json'