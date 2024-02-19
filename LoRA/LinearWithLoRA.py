import torch
import torch.nn as nn
from LoRA import LoRALayer
class LinearWithLoRA(nn.Module):
    
    def __init__(self, linear, rank, alpha) -> None:
        super().__init__()
        self.linear = linear
        
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )
        
    def forward(self, x):
        return self.linear(x) + self.lora(x)
