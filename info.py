import json

model_info = {
    'transformer': {
        # general info
        'model_name': 'transformer',
        'file_name': 'transformer.ckpt',
        'model_link': 'https://drive.google.com/file/d/1zAL3S8P3Fi9n5QdPLXcwx-QuVmBux-Ny/view?usp=drive_link',
        'source_code': 'https://github.com',
        'dataset': 'COCO2014',
        'bleu_score': '0.14',
        
        # model specific info
        'max_len': 32,
        'hidden_dim': 512,
        'num_heads': 8,
        'num_layers': 4,
        'dropout': 0.1,
        'ff_expantion': 4,
        'batch_size': 32,
        'lr': 5.0e-5,
        'max_epochs': 30,
    },
    'seq2seq': {
        # general info
        'model_name': 'seq2seq',
        'file_name': 'seq2seq.ckpt',
        'model_link': 'https://drive.google.com/uc?id=1-3Yyf2c3zv4zGZd8D7XxVY3W8Y8Y1b3C',
        'source_code': '',
        'dataset': 'COCO2014',
        'bleu_score': '0.14',
        
        # model specific info
        'max_len': 32,
        'hidden_dim': 512,
        'dropout': 0.5,
        'batch_size': 32,
        'lr': 3.0e-4,
        'max_epochs': 10,
    },
}

json.dump(model_info, open('model_info.json', 'w'), indent=4)