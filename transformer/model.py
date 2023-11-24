from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchtext.data.metrics import bleu_score

from random import random
import lightning as L

from . import config
from .utils import remove_specials

class Encoder(nn.Module):
    def __init__(self, input_channels, hidden_dim, dropout=0.3):
        super(Encoder, self).__init__()
        
        self.model = models.resnet50(weights='DEFAULT', progress=False)
        
        self.model.fc = nn.Linear(self.model.fc.in_features, hidden_dim)
        self.freeze()
        
    def forward(self, x):
        x = self.model(x)
        # x.shape: [batch_size, output_dim]
        return x
    
    def _req_grad(self, b):
        for name, params in self.named_parameters():
            if name.find('fc') != -1:
                break
            params.requires_grad = b
    
    def freeze(self):
        self._req_grad(False)
    
    def unfreeze(self):
        self._req_grad(True)

class FeedForward(nn.Module):
    """
        Linear layer after non-linearity
    """
    def __init__(self, n_embd):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd)
        )
    
    def forward(self, data):
        return self.model(data)
    
class AttentionHead(nn.Module):
    """
        Attention head of heads
    """
    def __init__(self, embed_size, head_size, batch_size, mask=False, dropout=0.2):
        super().__init__()
        self.mask = mask
        
        self.query = nn.Linear(embed_size, head_size)
        self.key = nn.Linear(embed_size, head_size)
        self.value = nn.Linear(embed_size, head_size)
        if mask:
            self.register_buffer('tril', torch.tril(torch.ones(batch_size, batch_size)))
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, data, crossData=None):
        """
            :param crossData: 
                * crossData = None -> It's Self Attention
                * else             -> It's Cross Attention
                
                
            B -> batch size
            T -> series length
            C -> size of embedding
        """
        B,T,C = data.shape
        query = self.query(data)
        if crossData is not None:
            key = self.key(crossData)
            v = self.value(crossData)
        else:
            key = self.key(data)
            v = self.value(data)
        
        wei = query @ key.transpose(1, 2) #* C**-0.5 # # (B, 1, C) @ (B, C, T) -> (B, 1, T)
        if self.mask:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        out = wei @ v # (B, 1, T) @ (B, T, C) -> (B, 1, C)
        return out
    
class MultipleAttention(nn.Module):
    """
        Multiple head for Attention
    """
    def __init__(self, embed_size, n_head, head_size, batch_size, mask=False, dropout=0.2):
        super().__init__()
        self.blockHead = nn.Sequential(*[AttentionHead(embed_size, head_size, batch_size, mask, dropout) for _ in range(n_head)])
        self.fc = nn.Linear(head_size * n_head, embed_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, data, crossData=None):
        heads = torch.cat([layer(data, crossData=crossData) for layer in self.blockHead], dim=-1)
        
        out = self.dropout(self.fc(heads))
        return out

        
class AttentionBlock(nn.Module):
    """
        Attention Block
    """
    def __init__(self, embed_size, n_head, batch_size, mask=False, dropout=0.2):
        super().__init__()
        head_size = embed_size // n_head
        self.sa = MultipleAttention(embed_size, n_head, head_size, batch_size, mask, dropout)
        self.ffwd = FeedForward(embed_size)
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)
        
        
    def forward(self, data, crossData=None):
        data = data + self.sa(self.ln1(data), crossData=crossData)
        data = data + self.ffwd(self.ln2(data))
        return data

class Decoder(nn.Module):
    def __init__(self, n_head, n_layer, batch_size, max_length, output_dim, embed_dim, hidden_dim, vocab, dropout, padding_idx, device):
        super(Decoder, self).__init__()
        self.device = device
        self.token_embed = nn.Embedding(output_dim, embed_dim, padding_idx)
        self.position_embed = nn.Embedding(max_length, embed_dim, padding_idx)
        self.output_dim = output_dim
        self.vocab = vocab
        self.hidden_dim = hidden_dim
        self.self_blocks = nn.Sequential(*[AttentionBlock(embed_dim, n_head, batch_size, mask=True) for _ in range(n_layer)])
        self.cros_blocks = nn.Sequential(*[AttentionBlock(embed_dim, n_head, batch_size) for _ in range(n_layer)])
        self.fc_out = nn.Linear(embed_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hidden):
        B, T = x.shape
        tok_emb = self.dropout(self.token_embed(x))
        pos_emb = self.position_embed(torch.arange(T, device=self.device))
        x = tok_emb + pos_emb
        x = self.self_blocks(x)
        
        query = hidden.unsqueeze(1)
        for layer in self.cros_blocks:
            x = layer(x, query)
        
        out = self.fc_out(x)
        return out
        
class Transformer(L.LightningModule):
    def __init__(
        self,
        input_dim,
        embed_dim,
        hidden_dim,
        output_dim,
        num_heads,
        num_layer,
        vocab,
        dropout,
        max_length,
        device,
    ):
        super(Transformer, self).__init__()
        self.save_hyperparameters()
        
        self.encoder = Encoder(
            input_dim,
            hidden_dim,
            dropout,
        )
        self.padding_idx = vocab['<pad>']
        self.decoder = Decoder(
            num_heads,
            num_layer,
            config.BATCH_SIZE,
            max_length,
            output_dim,
            embed_dim,
            hidden_dim,
            vocab,
            dropout,
            self.padding_idx,
            device
        )
        self.max_sent_size = max_length
        self.all_validation = []
        
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.padding_idx,
            label_smoothing=config.LABEL_SMOOTHING,
        )
        
        
    def forward(self, image_features, captions):
        
        batch_size, max_sent_size = captions.shape
        enc_outputs = self.encoder(image_features)
        out = self.decoder(captions[:, :-1], enc_outputs)
        return out

    def training_step(self, batch, batch_idx):
        img, caption = batch
        preds = self.forward(img, caption)
        
        output_dim = preds.size(-1)
        loss = self.loss_fn(
            preds.view(-1, output_dim),
            caption[:, 1:].reshape(-1),
        )
        self.log("train_loss", loss, prog_bar=True, on_step=True, logger=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        img, caption, all_captions = batch
        
        preds = self.forward(img, caption)
        output_dim = preds.size(-1)
        loss = self.loss_fn(
            preds.view(-1, output_dim),
            caption[:, 1:].reshape(-1),
        )
        
        self.all_validation.append((preds.argmax(-1), all_captions))
        self.log("val_loss", loss)
        return loss
    
    def on_validation_epoch_end(self):
        candidate_corpus = []
        references_corpus = []
        vocab = self.decoder.vocab
        for preds, all_captions in self.all_validation:
            for i in range(preds.size(0)):
                pred = preds[i].cpu().numpy()
                pred_token = vocab.lookup_tokens(pred)
                if '<eos>' in pred_token:
                    pred_token = pred_token[:pred_token.index('<eos>')]
                pred_token = remove_specials(pred_token)
                candidate_corpus.append(pred_token)
                
                all_caption = [remove_specials(cap) for cap in all_captions[i]]
                references_corpus.append(all_caption)
        
        bleu = bleu_score(candidate_corpus, references_corpus)
        self.log("bleu_score@4", bleu)
        self.all_validation.clear()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
        return optimizer
    
    
#     def predict(self, image, transform_image):
#         self.eval()
#         image = transform_image(image).to(self.device).unsqueeze(0)
#         with torch.no_grad():
#             enc_outputs = self.encoder(image)
#         vocab = self.decoder.vocab
#         res = [vocab['<sos>']]
#         hidden = enc_outputs.unsqueeze(0)
#         eos_idx = vocab['<eos>']
#         with torch.no_grad():
#             for t in range(self.max_sent_size):
#                 dec_input = torch.LongTensor([res[-1]]).to(self.device).unsqueeze(0)

#                 preds, hidden = self.decoder(dec_input, hidden)

#                 top1 = preds.argmax(-1)

#                 if top1 == eos_idx:
#                     break
#                 res.append(top1.cpu().detach().numpy()[0])
        
#         res = vocab.lookup_tokens(res[1:])
        
#         return res