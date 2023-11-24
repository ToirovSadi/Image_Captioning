from torchvision import transforms
from PIL import Image

from .model import Transformer
from .utils import beam_search
from .utils import post_proress

def load_model(checkpoint, device):
    return Transformer.load_from_checkpoint(checkpoint, map_location=device, device=device).eval()

def predict(model, img, num_candidates, max_steps=1000, device='cpu', postprocess=True):
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
    
    captions = beam_search(
        src=img,
        model=model,
        vocab=model.decoder.vocab,
        beam_width=5,
        num_candidates=num_candidates,
        max_steps=max_steps,
        max_candidates_coef=5,
    )
    
    if postprocess:
        captions = [post_proress(c) for c in captions]
        
    return captions