from torchvision import transforms
from PIL import Image
import pickle
import torch

from .model import Seq2Seq
from .utils import beam_search
from .utils import post_proress

def load_model(checkpoint, device):
    return Seq2Seq.load_from_checkpoint(checkpoint, map_location=device, device=device)

def predict(model, img, device='cpu', postprocess=True, **kwds):
    if type(model) is str:
        model = load_model(model, device)
    if type(img) is str:
        img = Image.open(img).convert('RGB')
        
    # pass the image through the transfoms
    # ImageNet mean and std
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    image_size = (224, 224)
    transform = transforms.Compose([
        transforms.Resize((232, 232)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    img = transform(img).unsqueeze(0)
    max_sent_size = model.max_sent_size
    
    model.eval()
    with torch.no_grad():
        img_feat = model.encoder(img)
        caption = model.decoder.sample(img_feat, max_sent_size)
        caption = caption.cpu().squeeze(0).numpy()
    
    caption = model.vocab.lookup_tokens(caption)
    if '<eos>' in caption:
        caption = caption[:caption.index('<eos>')]
    
    if postprocess:
        caption = post_proress(caption)
        
    return caption