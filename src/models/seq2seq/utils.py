import torch
from queue import PriorityQueue
from datetime import datetime
import yaml
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
nltk.download('punkt', quiet=True)
from torchvision.models.resnet import ResNet152_Weights

from torch.utils.data import DataLoader
from torchvision import transforms
from src.data.dataset import COCODataset
from src.data.dataset import collate_batch

def remove_specials(x):
    """
    Remove special tokens from a list of tokens.

    Args:
        x (list): List of tokens.

    Returns:
        list: List of tokens with special tokens removed.
    """
    specials = ['<unk>', '<sos>', '<eos>', '<pad>']
    for sp in specials:
        while sp in x:
            x.remove(sp)
    return x

def get_config(file_name):
    """
    Read and parse a YAML configuration file.

    Args:
        file_name (str): Path to the YAML configuration file.

    Returns:
        dict: Parsed configuration data.
    """
    print("reading config file:", file_name)
    with open(file_name) as f:
        data = yaml.safe_load(f)
    return data

def post_proress(caption, detokenize=True):
    """
    Perform post-processing on a caption.

    Args:
        caption (list): List of tokens representing the caption.
        detokenize (bool, optional): Whether to detokenize the caption. Defaults to True.

    Returns:
        str: Processed caption.
    """
    # remove specials
    caption = remove_specials(caption)
    
    # detokenize (join the tokens)
    return TreebankWordDetokenizer().detokenize(caption)

def get_transforms(img=None, train=True):
    """
    Get the image transformations for preprocessing.

    Args:
        img (PIL.Image, optional): Input image. Defaults to None.
        train (bool, optional): Whether the transformations are for training or testing. Defaults to True.

    Returns:
        torchvision.transforms.Compose: Composed image transformations.
    """
    # ImageNet mean and std
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image_size = (224, 224)
    
    transform = transforms.Compose([
        transforms.Resize((232, 232)),
        (transforms.RandomCrop if train else transforms.CenterCrop)(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    if img is not None:
        return transform(img)
    
    return transform

def get_datasets(config):
    """
    Get the training and validation datasets.

    Args:
        config (dict): Configuration data.

    Returns:
        tuple: Tuple containing the training and validation datasets.
    """
    ## Get the Datasets
    train_dataset = COCODataset(
        root=config['datasets']['train_dataset'],
        annotation_path=config['datasets']['train_caption'],
        train=True,
        image_transform=get_transforms(train=True),
        remove_idx=True,
        max_sent_size=config['max_sent_size'],
    )
    train_dataset.build_vocab(
        root=config['default_root_dir'] + 'vocab',
        min_freq=config['min_freq'],
        load_from_file=True,
    )

    print("Train Dataset Size: ", len(train_dataset))
    print("Vocab Size:", len(train_dataset.vocab))

    val_dataset = COCODataset(
        root=config['datasets']['val_dataset'],
        annotation_path=config['datasets']['val_caption'],
        vocab=train_dataset.vocab,
        train=True,
        image_transform=get_transforms(train=False),
        take_first=10_000,
        remove_idx=False,
        return_all_captions=True,
        max_sent_size=config['max_sent_size'],
    )
    print("Val Dataset Size: ", len(val_dataset))
    
    return train_dataset, val_dataset

def get_dataloaders(train_dataset, val_dataset, config):
    """
    Get the training and validation dataloaders.

    Args:
        train_dataset (COCODataset): Training dataset.
        val_dataset (COCODataset): Validation dataset.
        config (dict): Configuration data.

    Returns:
        tuple: Tuple containing the training and validation dataloaders.
    """
    ## Get the DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=config['train']['num_workers'],
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=False,
        num_workers=config['train']['num_workers'],
        collate_fn=collate_batch,
    )
    return train_dataloader, val_dataloader

def jaccard_index(a: list[str], b: list[str]) -> float:
    """
    Calculate the Jaccard index between two lists of strings.

    Args:
        a (list): First list of strings.
        b (list): Second list of strings.

    Returns:
        float: Jaccard index.
    """
    a = set(a)
    b = set(b)
    
    return float(len(a.intersection(b)) / len(a.union(b)))

### Beam Search Implementation
class BeamSearchNode:
    def __init__(self, word_id, log_prob, length):
        self.word_id = word_id
        self.log_prob = log_prob
        self.length = length
        
    def eval(self, alpha=0.75):
        return self.log_prob / (self.length ** alpha)
    
    def __lt__(self, other):
        return self.log_prob < other.log_prob

def beam_search(
    src,
    model,
    vocab=None,
    beam_width=5,
    num_candidates=1,
    max_steps=2000,
    max_candidates_coef=3,
    jaccard_threshold=0.5,
):
    """
    Perform beam search to generate captions for an input image.

    Args:
        src (torch.Tensor): Input image.
        model: Model used for caption generation.
        vocab (Vocab, optional): Vocabulary object. Defaults to None.
        beam_width (int, optional): Width of the beam. Defaults to 5.
        num_candidates (int, optional): Number of candidate captions to generate. Defaults to 1.
        max_steps (int, optional): Maximum number of steps for beam search. Defaults to 2000.
        max_candidates_coef (int, optional): Maximum number of candidates to consider. Defaults to 3.
        jaccard_threshold (float, optional): Jaccard index threshold for filtering duplicate captions. Defaults to 0.5.

    Returns:
        list: List of generated captions.
    """
    if vocab is None:
        if hasattr(model, "vocab"):
            vocab = model.vocab
        elif hasattr(model.decoder, "vocab"):
            vocab = model.decoder.vocab
        else:
            raise ValueError("vocab not specified")
    
    qsize = 1
    model.eval()
    candidates = []
    q = PriorityQueue()
    device = model.device
    max_size = model.max_sent_size
    max_candidates = num_candidates * max_candidates_coef
    
    # get encoder outputs
    with torch.no_grad():
        img_feat = model.encoder(src)
    
    sos_idx = vocab['<sos>']
    eos_idx = vocab['<eos>']
    decoder_input = torch.tensor([[sos_idx]], device=device)
    
    # create first node
    node = BeamSearchNode(decoder_input, 0, 1)
    q.put((-node.eval(), node))
    
    with torch.no_grad():
        while not q.empty():
            if qsize > max_steps:
                break
            
            score, node = q.get()
            
            if node.word_id[:, -1] == eos_idx and node.length > 1:
                candidates.append((score, node))
                if len(candidates) >= max_candidates:
                    break
                continue
            if node.length > max_size:
                continue
            
            # add <eos>, it's will be removed in node.word_id
            decoder_input = torch.cat((node.word_id, torch.tensor([[eos_idx]], device=device)), dim=1)
            preds = model.decoder(img_feat, decoder_input)
            preds = torch.log_softmax(preds[:, -1, :], dim=1)
            # preds.shape: [1, output_dim]
            topk, indices = torch.topk(preds, beam_width)
            # topk.shape: [1, beam_width]
            
            # add these topk to the queue
            for i in range(beam_width):
                next_decoder_in = indices[0][i].view(1, -1)
                next_decoder_in = torch.cat((node.word_id, next_decoder_in), dim=1)
                next_prob = topk[0][i].item()
                next_node = BeamSearchNode(
                    next_decoder_in,
                    node.log_prob + next_prob,
                    node.length + 1
                )
                q.put((-next_node.eval(), next_node))
            qsize += beam_width - 1
    
    while len(candidates) < max_candidates and not q.empty():
        candidates.append(q.get())
    
    res = []
    for score, node in candidates:
        ans = node.word_id.cpu().detach().numpy()[0]
        
        ans = vocab.lookup_tokens(ans)
        ans = remove_specials(ans)
        res.append((score, ans))
    
    # sort and take first num_candidates
    candidates = sorted(res, key=lambda x: x[0])

    result = []
    ignored = []
    for _, tokens in candidates:
        if len(result) == num_candidates:
            break
        
        good = True
        for r in result:
            if jaccard_index(tokens, r) > jaccard_threshold:
                good = False
        
        if good:
            result.append(tokens)
        else:
            ignored.append(tokens)
    
    if len(result) < num_candidates:
        result.extend(ignored[:num_candidates - len(result)])
    
    return result