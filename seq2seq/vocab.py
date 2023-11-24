import os
import pickle
from collections import Counter

class Vocab:
    def __init__(
        self,
        root='.',
        min_freq=2,
        max_tokens=None,
        specials=['<unk>', '<pad>', '<sos>', '<eos>'],
        file_name=None,
        load_from_file=True,
    ):
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
    
    def add_token(self, token):
        if token in self.token2idx:
            return
        
        self.token2idx[token] = self.idx
        self.idx2token[self.idx] = token
        
        self.idx += 1
        
    def add_tokens(self, tokens: list[str]):
        for token in tokens:
            self.add_token(token)
        
    def add_captions(self, captions: list[list[str]]):
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
        
    def __getitem__(self, token):
        return self.token2idx.get(token, self.unk_token)
    
    def __len__(self):
        return len(self.token2idx)
    
    def lookup_tokens(self, indices: list[int]) -> list[str]:
        return [self.idx2token.get(idx, self.unk_token) for idx in list(indices)]
    
    def lookup_indices(self, tokens: list[str]) -> list[int]:
        return [self.token2idx.get(token, self.unk_idx) for token in list(tokens)]
    
    def load_vocab(self):
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
        file_name = self._get_vocab_file_name()
        path = os.path.join(self.root, file_name)
        if os.path.exists(path):
            return
        
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    def _get_vocab_file_name(self):        
        return self.file_name + f"min_freq={self.min_freq}_max_tokens={self.max_tokens}.pkl"