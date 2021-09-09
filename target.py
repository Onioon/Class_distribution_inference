import argparse
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import warnings

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.data as data


# Define the architecture of the network
class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.fc1 = nn.Linear(104, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        #         return torch.sigmoid(self.fc4(x))
        return self.fc4(x)


# Define the accuracy on the test set
def accuracy(model, val_loader):
    model.eval()
    total_corr = 0
    for i, data in enumerate(val_loader):
        inputs, labels = data
        y_pred = model(inputs.float())
        for i in range(len(labels)):
            # <=50k is encoded to 0 and > 50k encoded to 1
            if y_pred[i].item() > 0.5:
                r = 1
            else:
                r = 0
            if r == labels[i].item():
                total_corr += 1

    return float(total_corr) / len(val_loader.dataset)
