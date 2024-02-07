from PIL import Image
import torch

import sys
import os
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, root_dir)

from .model import Seq2Seq
from .utils import get_transforms, post_proress, beam_search

def load_model(checkpoint, device):
    return Seq2Seq.load_from_checkpoint(checkpoint, map_location=device)

def predict(model, img, num_captions=1, device='cpu', postprocess=True, **kwds):
    if type(model) is str:
        model = load_model(model, device)
    
    if type(img) is str:
        img = Image.open(img).convert('RGB')

    img = get_transforms(img=img, train=False)
    img = img.unsqueeze(0).to(device)
    
    captions = beam_search(
        src=img,
        model=model,
        vocab=model.vocab,
        beam_width=5,
        num_candidates=num_captions,
        max_steps=2000,
        jaccard_threshold=0.5,
    )
    
    if postprocess:
        captions = [post_proress(caption) for caption in captions]
        
    return captions