from PIL import Image

import sys
import os
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, root_dir)

from .model import Transformer
from .utils import get_transforms, post_proress, beam_search

def load_model(checkpoint, device):
    """
    Load the Transformer model from a checkpoint file.

    Args:
        checkpoint (str): Path to the checkpoint file.
        device (str): Device to load the model on.

    Returns:
        Transformer: Loaded Transformer model.
    """
    return Transformer.load_from_checkpoint(checkpoint, map_location=device, device=device)

def predict(model, img, num_captions=1, device='cpu', postprocess=True, **kwds):
    """
    Generate captions for an image using the Transformer model.

    Args:
        model (Transformer or str): The Transformer model or path to the checkpoint file.
        img (str or PIL.Image.Image): Path to the image file or PIL image object.
        num_captions (int): Number of captions to generate.
        device (str): Device to run the model on.
        postprocess (bool): Whether to post-process the generated captions.
        **kwds: Additional keyword arguments.

    Returns:
        list: List of generated captions.
    """
    if type(model) is str:
        model = load_model(model, device)
        model.decoder.device = device
    
    if type(img) is str:
        img = Image.open(img).convert('RGB')

    img = get_transforms(img=img, train=False)
    img = img.unsqueeze(0).to(device)
    
    captions = beam_search(
        src=img,
        model=model,
        vocab=model.vocab,
        num_candidates=num_captions,
        **kwds,
    )
    
    if postprocess:
        captions = [post_proress(caption) for caption in captions]
    
    return captions