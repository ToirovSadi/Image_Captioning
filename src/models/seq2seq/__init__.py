import torch.nn as nn
import torch
from random import random

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, max_sent_size, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.max_sent_size = max_sent_size
        self.device = device
        
    def forward(self, image_features, captions, teacher_force_ratio=0.5):
        # image_features.shape: [batch_size, encoder.hidden_dim]
        # captions.shape: [batch_size, max_sent_size-1]
        batch_size, max_sent_size = captions.shape
        enc_outputs = self.encoder(image_features)
        # enc_outputs.shape: [batch_size, hidden_dim]
        vocab = self.decoder.vocab
        output_dim = self.decoder.output_dim
        
        output = torch.zeros(batch_size, max_sent_size, output_dim, device=self.device)
        hidden = enc_outputs.unsqueeze(0)
        dec_input = torch.ones(batch_size, device=self.device).long() * vocab['<sos>']
        for t in range(1, self.max_sent_size):
            preds, hidden = self.decoder(dec_input.unsqueeze(1), hidden)
            # preds.shape: [batch_size, output_dim]
            
            output[:, t-1, :] = preds

            top1 = preds.argmax(1)
            
            teacher_force = random() < teacher_force_ratio
            
            dec_input = captions[:, t-1] if teacher_force else top1
        
        return output
    
    
    def predict(self, image, transform_image):
        self.eval()
        image = transform_image(image).to(self.device).unsqueeze(0)
        with torch.no_grad():
            enc_outputs = self.encoder(image)
        vocab = self.decoder.vocab
        res = [vocab['<sos>']]
        hidden = enc_outputs.unsqueeze(0)
        eos_idx = vocab['<eos>']
        with torch.no_grad():
            for t in range(self.max_sent_size):
                dec_input = torch.LongTensor([res[-1]]).to(self.device).unsqueeze(0)

                preds, hidden = self.decoder(dec_input, hidden)

                top1 = preds.argmax(-1)

                if top1 == eos_idx:
                    break
                res.append(top1.cpu().detach().numpy()[0])
        
        res = vocab.lookup_tokens(res[1:])
        
        return res

# model information
model = {
    'name': 'seq2seq',
    'desc': 'Seq2Seq uses Encoder and Decoder Architerture',
    'notebook_path': 'notebooks/seq2seq.ipynb',
    'pre-trained_model': 'models/seq2seq.pt', # find the link in `models/README.md`
    'model_params': {
        'encoder': {
            'input_channels': 3,
            'hidden_dim': 512,
            'dropout': 0.3,
        },
        'decoder': {
            'output_dim': 9038, # size of vocab
            'embed_dim': 256,
            'hidden_dim': 512,
            'vocab': None, # will be loaded with the model
            'dropout': 0.3,
            'padding_idx': 1,
        },
        'seq2seq': {
            # 'encoder': None,
            # 'decoder': None,
            'max_sent_size': 20,
            'device': 'cpu', # run on cpu
        }
    }
}

def get_instance(load=True):
    from seq2seq.encoder import Encoder
    from seq2seq.decoder import Decoder
    from seq2seq import Seq2Seq
    
    params = model['model_params']
    
    encoder = Encoder(**params['encoder'])
    decoder = Decoder(**params['decoder'])
    seq2seq = Seq2Seq(encoder, decoder, **params['seq2seq'])
    
    if load:
        seq2seq = torch.load(model['pre-trained_model'], map_location=torch.device('cpu'))
    
    return seq2seq