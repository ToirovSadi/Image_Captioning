from PIL import Image
import torch
from torchvision import transforms

import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
nltk.download('punkt', quiet=True)

from seq2seq import get_instance

# take the image and run it through the model and return the result
def predict(image_path: str, model_name: str) -> str:
    img = Image.open(image_path)
    # set up the transforms
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image_size = (224, 224)
    transform = transforms.Compose([
        transforms.Resize((232, 232)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    # load the model
    model = get_instance(load=False)
    res = model.predict(img, transform)
    
    # detokenize the output
    res = TreebankWordDetokenizer().detokenize(res)
    
    return res


import argparse
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--image-path', type=str)
    p.add_argument('--model', type=str)
    args = p.parse_args()
    
    print(predict(args.image_path, args.model))