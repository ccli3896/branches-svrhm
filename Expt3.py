'''
This experiment alternates tasks every alpha_alt epochs and applies elastic weight consolidation for the task
that currently isn't being trained. 
Saves numpy arrays for each lesioned combination on each task (4x2).

This script is designed to vary seeds, starting alpha, alpha_alt. 
Inputs [0,99].
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

import DataLoaders as dl
import BranchArchs as ba
from utils import *


PATH = './data/Figure 4/Models/Alt'
SEEDS = 10
alpha_alts = [1,5,10,20,50]
save_every_n_epochs = 5 # How often to save
epochs = 50
importance = 1e4
i_factor = 1e3

task_id = int(sys.argv[1]) ############### CHANGE THIS

trainloader = dl.load_GaborTrainer(path='./gabors/', gabor_type=2)
testloader = dl.load_GaborTester(path='./gabors/', gabor_type=2)

#############
# Run
#############

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__=='__main__':

  seed = task_id%SEEDS
  alpha_alt = alpha_alts[(task_id%(SEEDS*len(alpha_alts)))//SEEDS]
  start_alpha = float(task_id//(SEEDS*len(alpha_alts)))
  alpha = start_alpha

  # Weight the importances differently depending on task to start
  if alpha == 0: imp_factor = 1
  else: imp_factor = i_factor


  criterion = mse_loss

  # Train model and save
  net = ba.BranchNet([ba.BigConvBranch(channels=1, dropout=0.5), 
                      ba.BigConvBranch(channels=1, dropout=0.5)],84+84,2).to(device)
  optimizer = optim.Adam(net.parameters(), lr=0.001)

  mname = f'{PATH}{int(alpha_alt)}_start{int(start_alpha)}_seed{seed}_start.npy'
  test_lesioned_net(net, testloader, fname=mname, printit=False)

  # Initializing the EWC
  ewc = EWC(net, trainloader, alpha)

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
      loss = criterion(outputs, labels, alpha) + importance*ewc.penalty(net)*imp_factor
      #print('Total loss',loss.item())
      loss.backward()
      optimizer.step()
      running_loss += loss.item()

    #print('[%d] loss: %.8f' %(epoch + 1, running_loss / (i+1)))
    running_loss = 0.0

    # Save model
    if ((epoch+1)%save_every_n_epochs==0) or epoch==0:
      mname = f'{PATH}{int(alpha_alt)}_start{int(start_alpha)}_seed{seed}_epoch{epoch}.npy'
      test_lesioned_net(net, testloader, fname=mname, printit=False)

    # Change task
    if (epoch+1) % alpha_alt == 0:
      ewc = EWC(net, trainloader, alpha)
      alpha = float(not alpha)

      # Weight the importances differently depending on task to start
      if alpha == 0: imp_factor = 1
      else: imp_factor = i_factor
