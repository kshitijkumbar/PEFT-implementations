import torch
import torch.nn as nn
from LinearWithLoRA import LinearWithLoRA

class MultiLayerPerceptron(nn.Module):
    
    def __init__(self, num_feats, num_hidden_1, num_hidden_2, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_feats, num_hidden_1),
            nn.ReLU(),
            nn.Linear(num_hidden_1, num_hidden_2),
            nn.ReLU(),
            nn.Linear(num_hidden_2, num_classes)
        )
    
    def forward(self, x):
        x = self.layers(x)
        
        return x 


model = MultiLayerPerceptron(
    num_feats= 784,
    num_hidden_1= 128,
    num_hidden_2= 256,
    num_classes= 10
)       

model.layers[0] = LinearWithLoRA(model.layers[0], rank=4, alpha=8)
model.layers[2] = LinearWithLoRA(model.layers[2], rank=4, alpha=8)
model.layers[4] = LinearWithLoRA(model.layers[4], rank=4, alpha=8)

print(model)