import os
import pickle
from collections import Counter

class Vocab:
    """
    A class representing a vocabulary for text data.

    Args:
        root (str): The root directory for saving and loading the vocabulary file.
        min_freq (int): The minimum frequency of a token to be included in the vocabulary.
        max_tokens (int): The maximum number of tokens to be included in the vocabulary.
        specials (list[str]): A list of special tokens to be included in the vocabulary.
        file_name (str): The name of the vocabulary file.
        load_from_file (bool): Whether to load the vocabulary from a file.

    Attributes:
        idx (int): The current index for adding tokens to the vocabulary.
        loaded (bool): Whether the vocabulary has been loaded from a file.
        token2idx (dict): A dictionary mapping tokens to their corresponding indices.
        idx2token (dict): A dictionary mapping indices to their corresponding tokens.
        unk_token (str): The token representing unknown words.
        unk_idx (int): The index of the unknown token in the vocabulary.

    Methods:
        __getitem__(self, token): Get the index of a token in the vocabulary.
        __len__(self): Get the size of the vocabulary.
        lookup_tokens(self, indices): Convert a list of indices to their corresponding tokens.
        lookup_indices(self, tokens): Convert a list of tokens to their corresponding indices.
        add_token(self, token): Add a token to the vocabulary.
        add_tokens(self, tokens): Add multiple tokens to the vocabulary.
        add_captions(self, captions): Add tokens from a list of captions to the vocabulary.
        load_vocab(self): Load the vocabulary from a file.
        save_vocab(self): Save the vocabulary to a file.

    """

    def __init__(
        self,
        root='.',
        min_freq=2,
        max_tokens=None,
        specials=['<unk>', '<pad>', '<sos>', '<eos>'],
        file_name=None,
        load_from_file=True,
    ):
        # Initialize the attributes
        self.root = root
        self.min_freq = 0 if min_freq is None else min_freq
        self.max_tokens = max_tokens
        self.file_name = file_name
        self.load_from_file = load_from_file
        self.specials = specials
        self.idx = 0
        self.loaded = False
        
        self.token2idx = dict()
        self.idx2token = dict()
        
        if specials is not None:
            for token in specials:
                self.add_token(token)
            self.unk_token = '<unk>'
            self.unk_idx = self.token2idx.get('<unk>', -1)
        
        if self.file_name is None:
            self.file_name = ''
        else:
            self.file_name += '_'
        
        if self.load_from_file:
            self.load_vocab()

    def __getitem__(self, token):
        """
        Get the index of a token in the vocabulary.

        Args:
            token (str): The token to get the index for.

        Returns:
            int: The index of the token in the vocabulary, or the index of the unknown token if the token is not found.

        """
        return self.token2idx.get(token, self.unk_token)
    
    def __len__(self):
        """
        Get the size of the vocabulary.

        Returns:
            int: The size of the vocabulary.

        """
        return len(self.token2idx)
   
    def lookup_tokens(self, indices):
        """
        Convert a list of indices to their corresponding tokens.

        Args:
            indices (list[int]): The list of indices to convert.

        Returns:
            list[str]: The list of tokens corresponding to the given indices.

        """
        return [self.idx2token.get(idx, self.unk_token) for idx in list(indices)]
    
    def lookup_indices(self, tokens):
        """
        Convert a list of tokens to their corresponding indices.

        Args:
            tokens (list[str]): The list of tokens to convert.

        Returns:
            list[int]: The list of indices corresponding to the given tokens.

        """
        return [self.token2idx.get(token, self.unk_idx) for token in list(tokens)]
 
    def add_token(self, token):
        """
        Add a token to the vocabulary.

        Args:
            token (str): The token to add.

        """
        if token in self.token2idx:
            return
        
        self.token2idx[token] = self.idx
        self.idx2token[self.idx] = token
        
        self.idx += 1
        
    def add_tokens(self, tokens):
        """
        Add multiple tokens to the vocabulary.

        Args:
            tokens (list[str]): The list of tokens to add.

        """
        for token in tokens:
            self.add_token(token)
        
    def add_captions(self, captions):
        """
        Add tokens from a list of captions to the vocabulary.

        Args:
            captions (list[list[str]]): The list of captions containing tokens to add.

        """
        if self.loaded: # no need to calculate this one more time
            return
        
        cnt = Counter()
        for caption in captions:
            cnt.update(caption)
        
        tokens = [token for token, freq in cnt.items() if freq >= self.min_freq]
        if self.max_tokens is not None:
            tokens = tokens[:self.max_tokens - len(self.specials)]
            
        for token in tokens:
            self.add_token(token)

    def load_vocab(self):
        """
        Load the vocabulary from a file.

        """
        file_name = self._get_vocab_file_name()
        path = os.path.join(self.root, file_name)
        if not os.path.exists(path):
            return
        
        with open(path, 'rb') as f:
            vocab = pickle.load(f)
            self.token2idx = vocab.token2idx
            self.idx2token = vocab.idx2token
            self.idx = vocab.idx

        self.loaded = True

    def save_vocab(self):
        """
        Save the vocabulary to a file.

        """
        file_name = self._get_vocab_file_name()
        path = os.path.join(self.root, file_name)
        if os.path.exists(path):
            return
        
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    def _get_vocab_file_name(self):        
        """
        Get the vocabulary file name based on the configuration.

        Returns:
            str: The vocabulary file name.

        """
        return self.file_name + f"min_freq={self.min_freq}_max_tokens={self.max_tokens}.pkl"