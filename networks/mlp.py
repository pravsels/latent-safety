import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm
import numpy as np
import math
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, hidden_layer=2):
        super(MLP, self).__init__()
        self.fcs = []
        # if hidden_layer = 0, go input_dim to output_dim
        
        self.fcs.append(nn.Linear(input_dim, hidden_dim))
        if hidden_layer > 1:
            for _ in range(hidden_layer-2):
                self.fcs.append(nn.Linear(hidden_dim, hidden_dim))
        self.fcs.append(nn.Linear(hidden_dim, output_dim))
        self.fcs = nn.ModuleList(self.fcs)  # Add this line
        
    def forward(self, x):
        for i in range(len(self.fcs)-1):
            fc = self.fcs[i]
            x = F.relu(fc(x))
        return self.fcs[-1](x)

class SpectralMLP(nn.Module):
    def __init__(self, input_dim, output_dim,  hidden_dim, gamma=1):
        super(SpectralMLP, self).__init__()
        self.fc1 = spectral_norm(nn.Linear(input_dim, hidden_dim))
        self.fc2 = spectral_norm(nn.Linear(hidden_dim, hidden_dim))
        self.fc3 = spectral_norm(nn.Linear(hidden_dim, output_dim))
        self.num_L = 3
        self.gamma=gamma**(1/self.num_L)
        
    def forward(self, x):
        x = F.relu(self.fc1(x)*self.gamma)
        x = F.relu(self.fc2(x)*self.gamma)
        x = self.fc3(x)*self.gamma
        return x



