'''
For loading datasets.
Gabor and CIFAR.
'''

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader

import numpy as np


# GABORS ---------------------------------------------------------------------------------

class GaborSet(Dataset):
    def __init__(self,ims,labels):
        self.data = ims
        
        self.data = torch.tensor(self.data).type(torch.float)
        self.labels = labels
        self.labels[:,0] = (labels[:,0]-np.pi/2)/(np.pi/2)
        self.labels = torch.tensor(labels).type(torch.float)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img, label = self.data[idx], self.labels[idx]
        return img, label

def load_GaborTrainer(path='./gabors/',gabor_type=2,batch_size=32):
    if gabor_type!=2:
        gabor_type=''
    ims_train = np.load(f'{path}gabor{gabor_type}_tr_im.npy')
    lbls_train = np.load(f'{path}gabor{gabor_type}_tr_lbl.npy')
    trainloader = torch.utils.data.DataLoader(GaborSet(ims_train,lbls_train), batch_size=batch_size,
                                              shuffle=True, num_workers=0)
    return trainloader

def load_GaborTester(path='./gabors/',gabor_type=2,batch_size=32):
    if gabor_type!=2:
        gabor_type=''
    ims_test = np.load(f'{path}gabor{gabor_type}_te_im.npy')
    lbls_test = np.load(f'{path}gabor{gabor_type}_te_lbl.npy')
    testloader = torch.utils.data.DataLoader(GaborSet(ims_test,lbls_test), batch_size=batch_size,
                                              shuffle=True, num_workers=0)
    return testloader


# CIFAR ---------------------------------------------------------------------------------
    # This one makes two classification tasks: categorization and maximum average color.   
    # FOR EXPERIMENT 4

def load_CIFARTrainer(path='./data/',batch_size=32):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root=path, train=True,
                                            download=True, transform=transform)

    train_data = trainset.data.reshape(-1,3,32*32)
    train_colors = np.argmax(np.mean(train_data,axis=2), axis=1).reshape(-1,1)
    train_labels = np.concatenate([np.array(trainset.targets).reshape(-1,1), train_colors], axis=1)

    trainset.targets = train_labels
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    return trainloader

def load_CIFARTester(path='./data/',batch_size=32):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = torchvision.datasets.CIFAR10(root=path, train=False,
                                           download=True, transform=transform)

    test_data = testset.data.reshape(-1,3,32*32)
    test_colors = np.argmax(np.mean(test_data,axis=2), axis=1).reshape(-1,1)
    test_labels = np.concatenate([np.array(testset.targets).reshape(-1,1), test_colors], axis=1)

    testset.targets = test_labels
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    return testloader