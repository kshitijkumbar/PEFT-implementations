import torch
import torch.nn as nn
import torch.functional as F
from LoRA import LoRALayer
class LinearWithDoRAMerged(nn.Module):
    
    def __init__(self, linear, rank, alpha) -> None:
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features,
            linear.out_features,
            rank,
            alpha
        )
        
        self.m = nn.Parameter(
            self.linear.weight.norm(p=2, dim=0,keepdim=True)
        )
        
    
    def forward(self, x):
        
        lora = self.lora.A @ self.lora.B
        combined_weight = self.linear.weight + self.lora.alpha * lora.T
        column_norm = combined_weight.norm(p=2, dim=0, keedim=True)
        V = combined_weight / column_norm
        
        new_weight = self.m * V
        return F.linear(x, new_weight, self.linear.bias)
    