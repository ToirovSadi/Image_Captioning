# Transformer Model

## baseline config

- General
  - SEED = 42
  - MAX_SENT_SIZE = 32
- Training Stage
  - BATCH_SIZE = 32
  - LR = 1e-4
  - MIN_LR = 1.e-5
  - MAX_EPOCHS = 30
  - LABEL_SMOOTHING = 0
  - WEIGHT_DECAY = 0
- Model Hyperparameters
  - embed_dim = 512
  - hidden_dim = 512
  - num_heads: 8
  - num_layers: 4
  - ff_expantion: 4
  - encoder_dropout = 0.5
  - decoder_dropout: 0.1

You can try to change the config file and run the train.py to train our model from scratch.
Of course, this model has very simple architecture, and yet is able to produce some meaningfull sentence.

## Results

| Experiment | epochs | train loss | val loss | bleu_score |
| :---:      | :---:  | :---:      | :---:    | :---:      |
| Baseline   | 30     | 1.818      | 2.194    | 0.262      |
