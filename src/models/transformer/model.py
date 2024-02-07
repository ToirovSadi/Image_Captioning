import torch
import torch.nn as nn
from nltk.translate.bleu_score import corpus_bleu
from torchvision import models

from random import random
import lightning as L

from .utils import remove_specials

### Encoder
class Encoder(nn.Module):
    def __init__(self, input_channels, hidden_dim, dropout=0.5):
        super(Encoder, self).__init__()
        
        self.model = models.resnet101(weights='IMAGENET1K_V2', progress=False)
        
        self.model.fc = nn.Linear(self.model.fc.in_features, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.freeze()
        
    def forward(self, x):
        x = self.dropout(self.model(x)).unsqueeze(1)
        # x.shape: [batch_size, 1, output_dim]
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

        
### Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, heads, dropout, device):
        super(MultiHeadAttention, self).__init__()
        
        assert hidden_dim % heads == 0
        
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)
        self.fc_q = nn.Linear(hidden_dim, hidden_dim)
        self.fc_o = nn.Linear(hidden_dim, hidden_dim)
        
        self.heads = heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.heads_dim = hidden_dim // heads
        self.scale = torch.sqrt(torch.tensor(self.heads_dim, device=device))
        
        
    def forward(self, query, key, value, mask=None, return_attention=False):
        # (key, value, query).shape: [batch_size, max_sent_size, hidden_dim]
        
        batch_size, max_sent_size, hidden_dim = key.shape
            
        key = self.fc_k(key).view(batch_size, -1, self.heads, self.heads_dim).permute(0, 2, 1, 3)
        value = self.fc_v(value).view(batch_size, -1, self.heads, self.heads_dim).permute(0, 2, 1, 3)
        query = self.fc_q(query).view(batch_size, -1, self.heads, self.heads_dim).permute(0, 2, 1, 3)
        # key.shape: [batch_size, heads, max_sent_size, heads_dim]
        
        energy = torch.matmul(query, key.permute(0, 1, 3, 2)) / self.scale
        # energy.shape: [batch_size, heads, max_sent_size, max_sent_size]
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))
        
        attention = torch.softmax(energy, dim=-1)
        
        x = torch.matmul(self.dropout(attention), value).permute(0, 2, 1, 3).contiguous()
        # x.shape: [batch_size, max_sent_size, heads, heads_dim]
        x = self.fc_o(x.view(batch_size, -1, self.hidden_dim))
        if return_attention:
            return x, attention
        return x

### Decoder
class DecoderBlock(nn.Module):
    def __init__(self, hidden_dim, heads, ff_expantion, dropout, device):
        super(DecoderBlock, self).__init__()
        
        self.self_attention = MultiHeadAttention(hidden_dim, heads, dropout, device)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        self.encoder_attention = MultiHeadAttention(hidden_dim, heads, dropout, device)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_expantion),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim * ff_expantion, hidden_dim),
        )
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, x, enc_out, src_mask=None, trg_mask=None, return_attention=False):
        # x.shape: [batch_size, max_sent_size]
        # enc_out.shape: [batch_size, max_sent_size, hidden_dim]
        # src_mask: [batch_size, max_sent_size]
        # trg_mask: [batch_size, max_sent_size]
        
        _x = self.dropout(self.self_attention(x, x, x, trg_mask))
        # _x.shape: [batch_size, max_sent_size, hidden_dim]
        
        x = self.norm1(_x + x)
        # x.shape: [batch_size, max_sent_size, hidden_dim]
        
        _x, attention = self.encoder_attention(x, enc_out, enc_out, src_mask, return_attention=True)
        
        x = self.norm2(self.dropout(_x) + x)
        
        x = self.norm3(self.dropout(self.ff(x)) + x)
        
        if return_attention:
            return x, attention
        return x
        

class Decoder(nn.Module):
    def __init__(
        self,
        output_dim,
        hidden_dim,
        num_layers,
        heads,
        ff_expantion,
        dropout,
        device,
        max_size,
        vocab
    ):
        super(Decoder, self).__init__()
        self.padding_idx = vocab['<pad>']
        self.token_embedding = nn.Embedding(output_dim, hidden_dim, padding_idx=self.padding_idx)
        self.pos_embedding = nn.Embedding(max_size, hidden_dim)

        self.layers = nn.ModuleList([
            DecoderBlock(
                hidden_dim=hidden_dim,
                heads=heads,
                ff_expantion=ff_expantion,
                dropout=dropout,
                device=device,
            ) for _ in range(num_layers)])
        
        self.vocab = vocab
        self.device = device
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.scale = torch.sqrt(torch.tensor(hidden_dim, device=self.device))
        
    def forward(self, x, enc_out, src_mask=None, trg_mask=None, return_attention=False):
        # x.shape: [batch_size, max_sent_size]
        # src_mask.shape: [batch_size, max_sent_size]
        # trg_mask.shape: [batch_size, max_sent_size]
        batch_size, max_sent_size = x.shape
        
        emb = self.dropout(self.token_embedding(x))
        # emb.shape: [batch_size, max_sent_size, hidden_dim]
        
        pos = torch.arange(0, max_sent_size, device=self.device).reshape(1, max_sent_size).repeat(batch_size, 1)
        pos = self.pos_embedding(pos)
        x = emb * self.scale + pos
        
        for layer in self.layers:
            if return_attention:
                x, attention = layer(x, enc_out, src_mask, trg_mask, return_attention=return_attention)
            else:
                x = layer(x, enc_out, src_mask, trg_mask, return_attention=return_attention)
        
        x = self.fc_out(x)
        
        if return_attention:
            return x, attention
        
        return x

        
class Transformer(L.LightningModule):
    def __init__(
        self,
        vocab,
        config,
        device='cpu',
    ):
        super(Transformer, self).__init__()
        self.save_hyperparameters()
        
        self.input_dim = config['model']['input_dim']
        self.embed_dim = config['model']['embed_dim']
        self.hidden_dim = config['model']['hidden_dim']
        self.output_dim = config['model']['output_dim']
        self.num_heads = config['model']['num_heads']
        self.num_layers = config['model']['num_layers']
        self.ff_expantion = config['model']['ff_expantion']
        self.enc_dropout = config['model']['encoder_dropout']
        self.dec_dropout = config['model']['decoder_dropout']
        self.max_sent_size = config['model']['max_length']
        self.vocab = vocab
        self.my_device = device
        
        self.encoder = Encoder(self.input_dim, self.embed_dim, self.enc_dropout)
        self.decoder = Decoder(
            self.output_dim,
            self.hidden_dim,
            self.num_layers,
            self.num_heads,
            self.ff_expantion,
            self.dec_dropout,
            self.my_device,
            self.max_sent_size,
            self.vocab
        )
        
        self.all_validation = []
        self.LR = config['train']['learning_rate']
        self.max_epochs = config['train']['max_epochs']
        self.min_lr = config['train'].get('min_learning_rate', None)
        self.label_smoothing = config['train'].get('label_smoothing', 0)
        self.weight_decay = config['train'].get('weight_decay', 0)

        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=vocab['<pad>'],
            label_smoothing=self.label_smoothing,
        )

    def mask_trg(self, x):
        # x.shape: [batch_size, max_sent_size]
        trg_pad_mask = (x != self.decoder.padding_idx).unsqueeze(1).unsqueeze(2)
        max_size = x.shape[1]
        mask = torch.tril(torch.ones((max_size, max_size), device=self.my_device)).bool()
        return trg_pad_mask & mask
        
    def forward(self, img, captions):
        captions = captions[:, :-1]
        # image.shape: [batch_size, C, H, W]
        # captions.shape: [batch_size, max_sent_size-1]
        captions_mask = self.mask_trg(captions)
        
        img_feat = self.encoder(img)
        # enc_outputs.shape: [batch_size, embed_dim]
        
        outputs = self.decoder(captions, img_feat, trg_mask=captions_mask)
        return outputs

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
        
        # predict the caption for these images using greed search
        self.all_validation.append((self.sample(img, self.max_sent_size), all_captions))
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def on_validation_epoch_end(self):
        candidate_corpus = []
        references_corpus = []
        for preds, all_captions in self.all_validation:
            preds = preds.cpu().numpy()
            for i in range(len(preds)):
                pred_token = self.vocab.lookup_tokens(preds[i])
                pred_token = remove_specials(pred_token)
                candidate_corpus.append(pred_token)
                                
                all_caption = [remove_specials(cap) for cap in all_captions[i]]
                references_corpus.append(all_caption)
        
        bleu = corpus_bleu(references_corpus, candidate_corpus)
        self.log("bleu_score@4", bleu)
        self.all_validation.clear()
    
    def sample(self, img, max_size):
        img_feat = self.encoder(img)
        # img_feat.shape: [batch_size, 1, hidden_dim]
        
        batch_size = img_feat.shape[0]
        res = torch.tensor([self.vocab['<sos>']] * batch_size, device=self.my_device).unsqueeze(1)
        # res.shape: [batch_size, 1]
        
        for i in range(max_size):
            preds = self.decoder(res, img_feat)
            # preds.shape: [batch_size, max_sent_size, output_dim]
            
            preds = preds.argmax(-1)[:, -1:]
            # preds.shape: [batch_size, 1] (predictions for each sample)
            
            res = torch.cat((res, preds), dim=1)
        
        return res
    
    def configure_optimizers(self):
        if self.weight_decay > 0:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.LR, weight_decay=self.weight_decay)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.LR, weight_decay=self.weight_decay)
        
        if self.min_lr is None:
            return optimizer
        
        train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.max_epochs, self.min_lr)
        return (
            {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": train_scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                    "strict": True,
                },
            },
        )

        
        
