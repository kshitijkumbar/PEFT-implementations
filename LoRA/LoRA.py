import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        
        # He initialzation for A (using only input dims and std dev for weight init)
        std_dev = 1/torch.sqrt(torch.tensor(rank).float())
        self.A = nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        
        # Zero matrix as per paper implementation
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        
        # Tunable hyperparameter
        self.alpha = alpha
        
    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x
    
            
        