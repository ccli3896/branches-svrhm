'''
Classes for building branched architectures.
'''

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader

import numpy as np

class ConvBranch(nn.Module):
    '''
    The base convolutional branch. 
    '''
    def __init__(self,
                channels = 3,
                dense_layers = 2, # Must be at least 2
                dense_size = 120,
                output_size = 84,
                ):
        super(ConvBranch, self).__init__()
        self.c = channels 
        
        # Make conv layers
        self.conv1 = nn.Conv2d(self.c, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Make a list of dense layers at the end
        self.denses = nn.ModuleList([nn.Linear(32*8*8, dense_size), nn.Linear(dense_size, output_size)])
        if dense_layers < 2:
            raise ValueError('Dense_layers must be at least 2')
        for _ in range(dense_layers-2):
            self.denses.insert(-1, nn.Linear(dense_size, dense_size))
            
            
    def forward(self, x):
        # Send through conv layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Send through dense layers
        x = x.view(-1, 32*8*8)
        for i,l in enumerate(self.denses):
            x = F.relu(l(x))
        return x
    

class WideBranch(nn.Module):
    '''
    The base wide branch.
    '''
    def __init__(self,
                input_size = 32*32*3,
                layers = 2,
                size = 86, # This was to approximately match the ConvBranch() defaults.
                output_size = 10,
                ):
        super(WideBranch, self).__init__()
        
        self.input_size = input_size
        
        # Make a list of dense layers 
        self.denses = nn.ModuleList([nn.Linear(input_size, size), nn.Linear(size, output_size)])
        if layers < 2:
            raise ValueError('Layers must be at least 2')
        for _ in range(layers-2):
            self.denses.insert(-1, nn.Linear(size, size))
            
    def forward(self, x):
        # Reshape image if necesary
        if len(x.shape) > 2:
            x = x.view(-1, self.input_size)
        
        # Run inputs through the dense layers
        for i,l in enumerate(self.denses):
            x = F.relu(l(x))
        return x
        

class BranchNet(nn.Module):
    '''
    Takes a list of modules that all take the same input size.
    Also needs the sum of output sizes from all branches.
    E.g. 
        BranchNet([ConvBranch(),WideBranch()], 84+10, [2])
    Makes a net that connects them all in parallel and adds a linear transform at the end.
    '''
    def __init__(self,
                 branches,
                 branch_outputs,
                 output_size):
        super(BranchNet, self).__init__()
        
        self.branches = nn.ModuleList(branches)
        self.fc = nn.Linear(branch_outputs, output_size)
        
    def forward(self, x, lesions=None):
        # Make a list of each branch's outputs.
        # lesions is a list of indices for branches to zero out.
        
        # Run through all branches
        branch_outs = []
        for b in self.branches:
            branch_outs.append(b(x))
            
        # Apply lesion if told to
        if lesions is not None:
            for l in lesions:
                branch_outs[l] = torch.zeros_like(branch_outs[l])
                
        # Connect branches
        outs = self.fc(torch.cat(branch_outs, dim=1))
        return outs

    def save(self, path):
        torch.save(self.state_dict(), path)
        print('Model saved')

    def load(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
        print('Model loaded')