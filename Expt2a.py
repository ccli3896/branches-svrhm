'''
Runs angle, size task on identical two conv net branches.
Run Expt2b.py after this to process models.
'''

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader

import numpy as np
import sys 
import pathlib

import DataLoaders as dl
import BranchArchs as ba

PATH = './data/Figure 3/Models/'
SEEDS = 10
alpha_increment = .25 
save_every_n_epochs = 5 # How often to save
epochs = 50

tasks = np.arange(50)

def mse_loss(preds, labels, alpha=.5):
    return alpha*torch.mean((preds[:,0]-labels[:,0])**2) + (1-alpha)*torch.mean((preds[:,1]-labels[:,1])**2) + 1e-6

trainloader = dl.load_GaborTrainer(path='./gabors/', gabor_type=2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#############
# Run
#############

if __name__=='__main__':
  path = pathlib.Path(PATH)
  path.mkdir(parents=True, exist_ok=True)

  for task_id in tasks:

    seed = task_id%SEEDS
    alpha = (task_id//SEEDS)*alpha_increment

    criterion = mse_loss

    # Train model and save
    net = ba.BranchNet([ba.ConvBranch(channels=1), ba.ConvBranch(channels=1)],84+84,2).to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    mname = f'{PATH}alpha{int(alpha*100)}_seed{seed}_start.pt'
    net.save(mname)

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.type(torch.float).to(device)
            labels = labels.type(torch.float).to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels, alpha)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print('[%d] loss: %.8f' %(epoch + 1, running_loss / (i+1)))
        running_loss = 0.0

        # Save model
        if ((epoch+1)%save_every_n_epochs==0) or epoch==0:
            mname = f'{PATH}alpha{int(alpha*100)}_seed{seed}_epoch{epoch}.pt'
            net.save(mname)
