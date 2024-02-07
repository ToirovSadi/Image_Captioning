# Seq2Seq Model

## baseline config

- General
  - SEED = 42
  - MAX_SENT_SIZE = 32
- Training Stage
  - BATCH_SIZE = 32
  - LR = 3e-4
  - MAX_EPOCHS = 10
  - LABEL_SMOOTHING = 0
  - WEIGHT_DECAY = 0
- Model Hyperparameters
  - embed_dim = 512
  - hidden_dim = 512
  - dropout = 0.5
  - num_layers = 1

You can try to change the config file and run the train.py to train our model from scratch.
Of course, this model has very simple architecture, and yet is able to produce some meaningfull sentence.

## Results

| Experiment | epochs | train loss | val loss | bleu_score |
| :---:      | :---:  | :---:      | :---:    | :---:      |
| Baseline   | 10     | 1.862      | 2.099    | 0.264     |
| Baseline with weight_decay=0.01   | 10     | 2.024      | 2.07    | 0.272     |