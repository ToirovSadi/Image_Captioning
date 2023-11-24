import json
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator

import nltk
from nltk import word_tokenize
nltk.download('punkt', quiet=True)

class COCODataset(Dataset):
    def __init__(
        self,
        root,
        annotation_path,
        train=True,
        image_transform=None,
        take_first=None,
        shuffle=True,
        vocab=None,
        max_sent_size=20,
        remove_idx=True,
        vocab_max_tokens=20_000,
        vocab_min_freq=2,
        return_all_captions=False,
    ):
        self.root = root
        self.train = train
        self.image_transform = image_transform
        self.vocab = vocab
        self.max_sent_size = max_sent_size
        self.return_all_captions = return_all_captions
        
        data = json.load(open(annotation_path))
        
        images = pd.DataFrame(data['images'])
        annotations = pd.DataFrame(data['annotations'])
        
        # merge images and annotations
        self.df = images.merge(annotations, left_on='id', right_on='image_id')
        if take_first:
            self.df = self.df[:take_first]
        
        if shuffle:
            self.df = self.df.sample(frac=1)
        
        self.df['caption_list'] = self.df['caption'].str.lower().apply(word_tokenize)
        if self.vocab is None:
            self.build_vocab(vocab_max_tokens, vocab_min_freq)
        
        idx = []
        for i in self.df.index:
            if len(self.df['caption_list'][i]) > self.max_sent_size-2: # include <sos>, <eos>
                idx.append(i)
        if remove_idx:
            self.df.drop(idx, axis=0, inplace=True)
        
        self.df.reset_index(drop=True, inplace=True)
        self.all_caption_dict = dict()
        for idx in self.df.index:
            img_id = self.df['image_id'][idx]
            
            if not img_id in self.all_caption_dict:
                self.all_caption_dict[img_id] = []
            
            self.all_caption_dict[img_id].append(self.df['caption_list'][idx])
    
    def __getitem__(self, idx):
        image_path = self.root + self.df['file_name'][idx]
        image_id = self.df['image_id'][idx]
        
        image = self._load_image(image_path)
        if self.image_transform:
            image = self.image_transform(image)

        if self.train:
            caption = self.df['caption_list'][idx]
            caption = self._preprocess_caption(caption)
            
            if self.return_all_captions:
                # get all captions (for calculating bleu score)
                all_captions = [self._add_padding(cap) for cap in self.all_caption_dict[image_id]]
                
                return image, caption, all_captions
            
            return image, caption
        
        return image
        
    
    def __len__(self):
        return len(self.df)
    
    
    def _load_image(self, image_path):
        return Image.open(image_path).convert('RGB')

    def _add_padding(self, caption):
        caption = ['<sos>'] + caption[:self.max_sent_size-2] + ['<eos>']
        while len(caption) < self.max_sent_size:
            caption.append('<pad>')  
        return caption
    
                            
    def _preprocess_caption(self, caption):
        caption = self._add_padding(caption)
        caption = self.vocab.lookup_indices(caption)
        return torch.tensor(caption)
    
    
    def build_vocab(self, max_tokens=20_000, min_freq=2):
        def yield_tokens(df):
            for i in df.index:
                captions = df['caption_list'][i]
                yield captions

        self.vocab_specials = ['<unk>', '<pad>', '<sos>', '<eos>']

        self.vocab = build_vocab_from_iterator(
            yield_tokens(self.df),
            min_freq=min_freq,
            specials=self.vocab_specials,
            max_tokens=max_tokens,
        )
        self.vocab.set_default_index(self.vocab_specials.index('<unk>'))

        
# collate batch function if you set return_all_captions=True in dataset
def collate_batch(batch):
    images = []
    captions = []
    all_captions = []
    for img, cap, all_cap in batch:
        images.append(img)
        captions.append(cap)
        all_captions.append(all_cap)
    
    images = torch.stack(images, dim=0)
    captions = torch.stack(captions, dim=0)
    
    return images, captions, all_captions