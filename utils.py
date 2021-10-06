import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from copy import deepcopy

import numpy as np
import sys 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

def test_lesioned_net(net, testloader, saveit=True, printit=False, fname=None, lesions=[None,[0],[1],[0,1]]):
  # Takes a network and tests it in all lesioned versions. 
  # Saves the result (8 numbers: 4 lesions x 2 tasks) in fname.

  net.eval()

  with torch.no_grad():
    outs = np.zeros((len(lesions),2))
    for i,l in enumerate(lesions):
      outs[i,:] = lesion_score(net, testloader,lesions=l)

  if printit:
    print(outs,fname)

  net.train()

  if saveit:
    np.save(fname, outs)
  else:
    return outs

def mse_loss(preds, labels, alpha=.5):
    return alpha*torch.mean((preds[:,0]-labels[:,0])**2) + (1-alpha)*torch.mean((preds[:,1]-labels[:,1])**2) + 1e-6

# https://github.com/moskomule/ewc.pytorch/blob/master/utils.py
class EWC(object):
    def __init__(self, model: nn.Module, dataset, alpha: float):

        self.model = model
        self.dataset = dataset
        self.alpha = alpha

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = p.data

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_() # Zeros out the data
            precision_matrices[n] = p.data # Makes variance reciprocal matrix placeholders

        self.model.eval()
        for img,lbl in self.dataset:
            img = img.to(device)
            lbl = lbl.to(device)
            self.model.zero_grad()
            outputs = self.model(img)
            loss = mse_loss(outputs, lbl, self.alpha)
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / (len(self.dataset)/self.dataset.batch_size)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss
  