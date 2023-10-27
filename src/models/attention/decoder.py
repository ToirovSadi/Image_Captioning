import torch
import torch.nn as nn
import torch.nn.functional as F

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

        
class Attention(nn.Module):
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
        self.self_blocks = nn.Sequential(*[Attention(embed_dim, n_head, batch_size, mask=True) for _ in range(n_layer)])
        self.cros_blocks = nn.Sequential(*[Attention(embed_dim, n_head, batch_size) for _ in range(n_layer)])
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