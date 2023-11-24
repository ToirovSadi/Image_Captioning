import gdown
import torch

from transformer.predict_model import predict

import argparse
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model_name', type=str, default='transformer')
    p.add_argument('--image_path', type=str)
    p.add_argument('--num_captions', type=int, default=1)
    p.add_argument('--max_steps', type=int, default=2000)
    args = p.parse_args()
    
    ckpt_path = 'transformer/logs/2023-11-23_13-19-38/epoch=29-step=49665.ckpt'
    
    captions = predict(ckpt_path, args.image_path, num_candidates=args.num_captions, max_steps=args.max_steps)
    
    if len(captions) == 1:
        print("caption:", captions[0])
    else:
        print("captions:")
        for i, caption in enumerate(captions):
            print(f"{i}) {caption}")
    
#     model_name = args.model_name
#     if model_name is None:
#         choose_model_name()
#     get_model()
    
#     print(f"You have {num_predictions} number of predictions per program run, you can write your sentence in console and get answer from the model")
#     for _ in range(num_predictions):
#         toxic_sent = input("Your sentence: ")
#         try:
            
#             # in later versions this `use_encoder_out` will be removed so
#             # you can ignore it
#             if model_name.startwith('attention'):
#                 print(model.predict(toxic_sent, use_encoder_out=True))
#             else:
#                 print(model.predict(toxic_sent))
#         except Exception as e:
#             print("ERROR occured while trying to predict, msg:", e)
    