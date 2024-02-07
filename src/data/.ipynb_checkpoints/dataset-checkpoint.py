import json
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset

import nltk
from nltk import word_tokenize
nltk.download('punkt', quiet=True)


from .vocab import Vocab

class COCODataset(Dataset):
    """
    A custom dataset class for COCO dataset used in image captioning.

    Args:
        root (str): The root directory of the dataset.
        annotation_path (str): The path to the annotation file.
        train (bool): Whether the dataset is for training or not. Default is True.
        image_transform (callable): A function/transform to apply on the image. Default is None.
        take_first (int): Number of samples to take from the dataset. Useful for sanity check or batch overfitting. Default is None.
        shuffle (bool): Whether to shuffle the dataset. Default is True.
        vocab (Vocab): The vocabulary object. Default is None.
        max_sent_size (int): Maximum size of the caption in terms of tokens. Default is 20.
        remove_idx (bool): Whether to remove captions that have more than `max_sent_size` tokens. Default is True.
        return_all_captions (bool): Whether to return all captions for each image. Default is False.

    Attributes:
        df (DataFrame): The merged DataFrame of images and annotations.
        all_caption_dict (dict): A dictionary mapping image IDs to their captions.

    Methods:
        __getitem__(self, idx): Retrieves the image and caption at the given index.
        __len__(self): Returns the length of the dataset.
        _load_image(self, image_path): Loads and returns the image from the given path.
        _add_padding(self, caption): Truncates the caption and adds padding.
        _preprocess_caption(self, caption): Converts tokens to indices.
        build_vocab(self, root, max_tokens=None, min_freq=2, load_from_file=True): Builds the vocabulary object.

    """

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
        return_all_captions=False,
    ):
        # Initialize the dataset attributes
        self.root = root
        self.train = train
        self.image_transform = image_transform
        self.vocab = vocab
        self.max_sent_size = max_sent_size
        self.take_first = take_first
        self.remove_idx = remove_idx
        self.return_all_captions = return_all_captions
        
        # Load the annotation data
        data = json.load(open(annotation_path))
        
        # Create DataFrames for images and annotations
        images = pd.DataFrame(data['images'])
        annotations = pd.DataFrame(data['annotations'])
        
        # Merge images and annotations
        self.df = images.merge(annotations, left_on='id', right_on='image_id')
        
        # Take only the first `take_first` samples if specified
        if take_first:
            self.df = self.df[:take_first]
        
        # Shuffle the dataset if specified
        if shuffle:
            self.df = self.df.sample(frac=1)
        
        # Apply simple preprocessing to the captions
        self.df['caption_list'] = self.df['caption'].str.lower().apply(word_tokenize)
        
        # Remove captions that have more than `max_sent_size` tokens
        if remove_idx:
            idx = []
            for i in self.df.index:
                if len(self.df.loc[i, 'caption_list']) > self.max_sent_size-2: # include <sos>, <eos>
                    idx.append(i)

            self.df.drop(idx, axis=0, inplace=True)
        
        # Reset indices
        self.df.reset_index(drop=True, inplace=True)
        
        # Map image IDs to their captions
        if self.return_all_captions:
            self.all_caption_dict = dict()
            for idx in self.df.index:
                img_id = self.df.loc[idx, 'image_id']

                if not img_id in self.all_caption_dict:
                    self.all_caption_dict[img_id] = []

                self.all_caption_dict[img_id].append(self.df.loc[idx, 'caption_list'])
    
    def __getitem__(self, idx):
        """
        Retrieves the image and caption at the given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the image and caption.

        """
        image_path = self.root + self.df.loc[idx, 'file_name']
        image_id = self.df.loc[idx, 'image_id']
        
        image = self._load_image(image_path)
        if self.image_transform:
            image = self.image_transform(image)

        if self.train:
            caption = self.df.loc[idx, 'caption_list']
            caption = self._preprocess_caption(caption)
            
            if self.return_all_captions:
                # Get all captions (for calculating bleu score)
                all_captions = [self._add_padding(cap) for cap in self.all_caption_dict[image_id]]
                
                return image, caption, all_captions
            
            return image, caption
        
        return image
        
    
    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.

        """
        return len(self.df)
    
    def _load_image(self, image_path):
        """
        Loads and returns the image from the given path.

        Args:
            image_path (str): The path to the image.

        Returns:
            PIL.Image.Image: The loaded image.

        """
        return Image.open(image_path).convert('RGB')

    def _add_padding(self, caption):
        """
        Truncates the caption and adds padding.

        Args:
            caption (list): The list of tokens representing the caption.

        Returns:
            list: The padded caption.

        """
        caption = ['<sos>'] + caption[:self.max_sent_size-2] + ['<eos>']
        return caption + ['<pad>'] * (self.max_sent_size - len(caption))
    
    def _preprocess_caption(self, caption):
        """
        Converts tokens to indices.

        Args:
            caption (list): The list of tokens representing the caption.

        Returns:
            torch.Tensor: The tensor containing the indices of the caption.

        """
        caption = self._add_padding(caption)
        caption = self.vocab.lookup_indices(caption)
        return torch.tensor(caption)
    
    def build_vocab(self, root, max_tokens=None, min_freq=2, load_from_file=True):
        """
        Builds the vocabulary object.

        Args:
            root (str): The directory to save the vocabulary.
            max_tokens (int): The maximum number of tokens in the vocabulary. Default is None.
            min_freq (int): The minimum frequency of tokens to be included in the vocabulary. Default is 2.
            load_from_file (bool): Whether to load the vocabulary from file if it exists. Default is True.

        """
        self.vocab = Vocab(
            root=root, # where to save the vocab
            min_freq=min_freq,
            max_tokens=max_tokens,
            file_name=f"take_first={self.take_first}_max_length={self.max_sent_size}_remove_idx={self.remove_idx}",
            load_from_file=load_from_file,
        )
        self.vocab.add_captions(self.df['caption_list'].tolist())
        self.vocab.save_vocab()


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