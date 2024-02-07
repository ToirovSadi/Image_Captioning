from PIL import Image
import torch

import sys
import os
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, root_dir)

from .model import Seq2Seq
from .utils import get_transforms, post_proress, beam_search

def load_model(checkpoint, device):
    """
    Load the trained model from a checkpoint file.

    Args:
        checkpoint (str): Path to the checkpoint file.
        device (str): Device to load the model on.

    Returns:
        Seq2Seq: The loaded model.
    """
    return Seq2Seq.load_from_checkpoint(checkpoint, map_location=device)

def predict(model, img, num_captions=1, device='cpu', postprocess=True, **kwds):
    """
    Generate captions for an image using the trained model.

    Args:
        model (str or Seq2Seq): The model to use for prediction. If a string is provided, it is assumed to be the path to the checkpoint file.
        img (str or PIL.Image.Image): The image to generate captions for. If a string is provided, it is assumed to be the path to the image file.
        num_captions (int): The number of captions to generate for the image. Default is 1.
        device (str): Device to perform the prediction on. Default is 'cpu'.
        postprocess (bool): Whether to apply post-processing to the generated captions. Default is True.
        **kwds: Additional keyword arguments to be passed to the beam_search function.

    Returns:
        list: A list of generated captions for the image.
    """
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
        num_candidates=num_captions,
        **kwds
    )
    
    if postprocess:
        captions = [post_proress(caption) for caption in captions]
        
    return captions