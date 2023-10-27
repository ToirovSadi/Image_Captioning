import torch.nn.functional as F
import torch.nn as nn
import torch

class Attention(nn.Module):
    def __init__(self, encoder, decoder, max_sent_size, device):
        super(Attention, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.max_sent_size = max_sent_size
        self.device = device
        
    def forward(self, image_features, captions, teacher_force_ratio=0.5):
        batch_size, max_sent_size = captions.shape
        enc_outputs = self.encoder(image_features)
        vocab = self.decoder.vocab
        output_dim = self.decoder.output_dim
        out = self.decoder(captions, enc_outputs)
        return out
    
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
                dec_input = torch.LongTensor(res).to(self.device).unsqueeze(0)
                preds = self.decoder(dec_input, enc_outputs) # (1, t, vocab_size)
                logits = preds[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                res.append(idx_next[0].cpu().detach().numpy()[0])
        
        res = vocab.lookup_tokens(res[1:])
        
        return res