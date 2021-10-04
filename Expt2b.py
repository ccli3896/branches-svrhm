'''
Tests models saved from Expt1a.py.
Scores intact and lesioned models.
Output files are [intact score, branch 1 lesioned score, branch 2 lesioned score, both branches lesioned score]. Saves scores on task 1 and task 2. That is, output format is

[[task 1, intact    task 1, branch 1 lesioned task 1, branch 2 lesioned task 1, both lesioned],
 [task 2, intact    task 2, branch 1 lesioned task 2, branch 2 lesioned task 2, both lesioned]]

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

def lesion_score(net, testloader, lesions=None):
  # Takes a trained net with a lesion option in the forward() method.
  # Returns the MSE for two regression tasks assuming two outputs.

  def mse_score(preds, labels):
    return torch.mean((preds[:,0]-labels[:,0])**2), torch.mean((preds[:,1]-labels[:,1])**2)
  
  tot0, tot1 = 0,0
  for i,(img,lbl) in enumerate(testloader):
    img = img.to(device)
    pred = net(img, lesions=lesions).detach().cpu()

    task0, task1 = mse_score(pred, lbl)
    tot0 += task0
    tot1 += task1
  tot0 /= i
  tot1 /= i
  return tot0.numpy(),tot1.numpy()

def score_all_nets(
  net,
  path,
  lesions,
  testloader,
  epochs_trained = 10, # Not including start and first epoch
  epoch_incr = 5, 
  seeds = 10,
):
  # Scores all the seeds and epochs of a set of trained networks. 
  # Will have to feed in alphas separately. 
  # 
  # net is an instance of the correct network architecture.
  # path is start of model names, e.g. './Models/alpha0'
  # lesions is a list, e.g. [None, [0], [1]]
  # testloader is a data loader object.
  #
  # Output is a list where each item is for a different lesion.
  # The items are arrays: [num_lesions, task, epoch number, seed].

  # Get filenames set up
  post = ['start','epoch0']
  for i in range(epochs_trained):
    post.append(f'epoch{(i+1)*epoch_incr-1}')

  # Loop over epochs and seeds
  outs = np.zeros((len(lesions), 2, epochs_trained+2, seeds))
  for p_i, p in enumerate(post):
    for seed in range(seeds):
      print(p_i, seed)
      # Load network state
      fname = f'{path}_seed{seed}_{p}.pt'
      net.load(fname)
      net = net.to(device)

      # Try each lesion
      for i,l in enumerate(lesions):
        scores = lesion_score(net, testloader, l)
        outs[i, :, p_i, seed] = scores

  return np.array(outs)

if __name__=='__main__':
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  net = ba.BranchNet([ba.ConvBranch(channels=1), ba.ConvBranch(channels=1)],84+84,2).to(device)
  testloader = dl.load_GaborTester(path='./gabors/', gabor_type=2)


  for alpha in np.arange(0,101,25):
    outs = score_all_nets(net, f'./data/Figure 3/Models/alpha{alpha}', [None, [0],[1],[0,1]], testloader,
                          epochs_trained=10,
                          seeds=10)
    np.save(f'./data/Figure 3/Results_alpha{alpha}.npy',outs,allow_pickle=True)
