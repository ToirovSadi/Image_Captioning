import gdown
import torch
import warnings
warnings.filterwarnings("ignore")

model_names = {
    'transformer': 'models/transformer.ckpt',
    'transformer_small': 'models/transformer_small.ckpt',
    'seq2seq': 'models/seq2seq.ckpt',
}

def get_predict_fn(args):
    model_name = args.model_name
    if model_name not in model_names:
        raise ValueError(f"Incorrect model name, available one {list(model_names.keys())}")
    
    if model_name == 'transformer' or model_name == 'transformer_small':
        from transformer.predict_model import predict
        return predict, model_names[model_name]
    if model_name == 'seq2seq':
        from seq2seq.predict_model import predict
        return predict, model_names[model_name]
    
    raise RuntimeError("model name not found!")

import argparse
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model_name', type=str, default='transformer')
    p.add_argument('--image_path', type=str)
    p.add_argument('--num_captions', type=int, default=1)
    p.add_argument('--max_steps', type=int, default=2000)
    args = p.parse_args()
    
    predict_fn, ckpt_path = get_predict_fn(args)
    
    captions = predict_fn(ckpt_path, args.image_path, num_candidates=args.num_captions, max_steps=args.max_steps)
    
    if type(captions) is not list:
        captions = [captions]
    
    if len(captions) == 1:
        print("caption:", captions[0])
    else:
        print("captions:")
        for i, caption in enumerate(captions):
            print(f"{i}) {caption}")
