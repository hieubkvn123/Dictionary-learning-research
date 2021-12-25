import numpy as np

# Torch dependencies
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as VF

import torchvision
from torchvision import datasets, transforms

# Custom transform class to en-noise image data
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        noisy = tensor + torch.normal(mean=self.mean, std=self.std, size=tensor.size())
        noisy = torch.clamp(noisy, 0.0, 0.5)
        return (noisy, tensor)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class ISTA_Module(nn.Module):
    def __init__(self, input_dim):
        super(ISTA_Module, self).__init__()
        self._lambda = nn.Parameter(data=torch.normal(size=(1,), std=1.0, mean=0.0))
        self.linear1 = nn.Linear(in_features=input_dim, out_features=input_dim)
        self.linear2 = nn.Linear(in_features=input_dim, out_features=input_dim)
        
    def soft_thresholding(self, x):
        return torch.sign(x) * F.relu(torch.abs(x) - self._lambda)
    
    def forward(self, x, y):
        outputs = self.soft_thresholding(self.linear1(x) + self.linear2(y))
        
        return outputs
    
class LISTA(nn.Module):
    def __init__(self, input_dim, L=10):
        super(LISTA, self).__init__()
        self.input_dim = input_dim
        self.L = L
    
        self.ista_modules = nn.ModuleList()
        
        for i in range(self.L):
            self.ista_modules.append(ISTA_Module(input_dim=self.input_dim))
        
    def forward(self, y):
        x = torch.zeros_like(y)
        
        for layer in self.ista_modules:
            x = layer(x, y)
            
        return x
    
class DenoiseNetwork(nn.Module):
    def __init__(self, input_dim, num_layers=3):
        super(DenoiseNetwork, self).__init__()
        self.num_layers = num_layers
        self.batchnorms = nn.ModuleList()
        self.layers = nn.ModuleList()
        
        for i in range(self.num_layers):
            self.layers.append(nn.Linear(in_features=input_dim, out_features=input_dim))
            self.layers.append(nn.BatchNorm1d(input_dim))
            
    def forward(self, x):
        outputs = x
        
        for norm, layer in zip(self.batchnorms, self.layers):
            x = layer(x)
            x = F.relu(x)
            x = norm(x)
            
        outputs = torch.sigmoid(x)
        return outputs
