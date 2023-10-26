from torchvision import models
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_channels, hidden_dim, dropout=0.3):
        super(Encoder, self).__init__()
        
        self.model = models.resnet50(weights='DEFAULT', progress=False)
        
        self.model.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            
            nn.LazyLinear(2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(dropout),
            
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            
            nn.Linear(512, hidden_dim),
        )
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