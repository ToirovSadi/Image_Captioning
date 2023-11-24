from torchvision import models
import torch.nn as nn
import torch
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


class Decoder(nn.Module):
    def __init__(
        self,
        output_dim,
        embed_dim,
        hidden_dim,
        num_layers,
        dropout,
        padding_idx,
    ):
        super(Decoder, self).__init__()
        
        self.embedding = nn.Embedding(output_dim, embed_dim, padding_idx)
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        self.rnn = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, img_features, captions):
        # img_features.shape: [batch_size, embed_size]
        # captions.shape: [batch_size, max_length-1]
        captions = captions[:, :-1] # remove <eos>
        
        emb = self.dropout(self.embedding(captions))
        # x.shape: [batch_size, max_length-1, embed_dim]
        
        x = torch.cat((img_features.unsqueeze(1), emb), dim=1)
        # x.shape: [batch_size, max_length, embed_dim]
        
        outputs, _ = self.rnn(x)
        # outputs.shape: [batch_size, max_length, hidden_dim]
        x = self.fc_out(outputs)
        
        # output.shape: [batch_size, max_length, output_dim]
        return x
    
    def sample(self, img_feat, max_sent_size, states=None):
        res = []
        dec_in = img_feat.unsqueeze(1)
        
        for i in range(max_sent_size):
            hidden, states = self.rnn(dec_in, states)
            outputs = self.fc_out(hidden.squeeze(1))
            # outputs.shape: [batch_size, output_dim]

            preds = outputs.argmax(-1)
            res.append(preds)

            dec_in = self.embedding(preds)
            dec_in = dec_in.unsqueeze(1)
        
        return torch.stack(res, dim=1)
        
class Seq2Seq(L.LightningModule):
    def __init__(
        self,
        input_dim,
        embed_dim,
        hidden_dim,
        output_dim,
        num_layers,
        vocab,
        dropout,
        max_sent_size,
    ):
        super(Seq2Seq, self).__init__()
        self.save_hyperparameters()
        
        self.encoder = Encoder(input_dim, embed_dim, dropout)
        self.decoder = Decoder(output_dim, embed_dim, hidden_dim, num_layers, dropout, vocab['<pad>'])
        
        self.max_sent_size = max_sent_size
        self.vocab = vocab
        self.all_validation = []
        
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=vocab['<pad>'],
            label_smoothing=config.LABEL_SMOOTHING,
        )

        
    def forward(self, img, captions):
        # image.shape: [batch_size, C, H, W]
        # captions.shape: [batch_size, max_sent_size]
        
        img_feat = self.encoder(img)
        # enc_outputs.shape: [batch_size, embed_dim]
        
        outputs = self.decoder(img_feat, captions)
        
        return outputs

    def training_step(self, batch, batch_idx):
        img, caption = batch
        
        img_feat = self.encoder(img)
        preds = self.decoder(img_feat, caption)
        
        output_dim = preds.size(-1)
        loss = self.loss_fn(
            preds.view(-1, output_dim),
            caption.reshape(-1),
        )
        self.log("train_loss", loss, prog_bar=True, on_step=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        img, caption, all_captions = batch
        
        img_feat = self.encoder(img)
        preds = self.decoder(img_feat, caption)
        
        output_dim = preds.size(-1)
        loss = self.loss_fn(
            preds.view(-1, output_dim),
            caption.reshape(-1),
        )
        
        # predict the caption for these images using greed search
        self.all_validation.append((self.decoder.sample(img_feat, self.max_sent_size), all_captions))
        self.log("val_loss", loss)
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