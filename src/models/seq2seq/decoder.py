import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, vocab, dropout, padding_idx):
        super(Decoder, self).__init__()
        
        self.embedding = nn.Embedding(output_dim, embed_dim, padding_idx)
        self.output_dim = output_dim
        self.vocab = vocab
        self.hidden_dim = hidden_dim
        
        self.rnn = nn.GRU(
            embed_dim,
            hidden_dim,
            batch_first=True,
        )
        
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hidden):
        # in our case batch_size=1
        # x.shape: [batch_size, max_sent_size]
        # hidden.shape: [num_layers, batch_size, hidden_dim]
        x = self.dropout(self.embedding(x))
        # x.shape: [batch_size, max_sent_size, embed_dim]
        
        outputs, hidden = self.rnn(x, hidden)
        # outputs.shape: [batch_size, max_sent_size, hidden_dim]
        # outputs.shape: [1, 1, hidden_dim]
        x = self.fc_out(outputs.squeeze(1))
        
        return x, hidden