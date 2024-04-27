import torch
import torch.nn as nn
import torch.functional as F
from LoRA import LoRALayer

# DoRA for linear layer
class LinearWithDoRAMerged(nn.Module):
    
    def __init__(self, linear, rank, alpha) -> None:
        super().__init__()
        
        # Normal Linear Layer
        self.linear = linear
        
        # LoRA Layer for directional component
        self.lora_layer = LoRALayer(
            linear.in_features,
            linear.out_features,
            rank,
            alpha
        )
        
        # Learnable Magnitude componente
        self.m = nn.Parameter(
            self.linear.weight.norm(p=2, dim=0,keepdim=True)
        )
        
    
    def forward(self, x):
        
        # Get lora product for low rank matrices for directional componet
        lora_prod = self.lora_layer.A @ self.lora_layer.B
        
        # Get the normalized directional matrix
        V_deltaV = self.linear.weight + self.lora_layer.alpha * lora_prod.T
        column_norm = V_deltaV.norm(p=2, dim=0, keedim=True)
        dir_mat = V_deltaV / column_norm
        
        # return DoRA updated weights
        new_weight = self.m * dir_mat
        return F.linear(x, new_weight, self.linear.bias)
    