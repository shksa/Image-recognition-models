from __future__ import print_function
import numpy as np
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt
from six.moves import range
from data_utils import get_CIFAR10_data
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import time
import copy
import os

tr = transforms.Compose([
        transforms.Scale(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

model = torchvision.models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)
f = open('soft.pkl', 'rb')
model.load_state_dict(torch.load(f))
f.close()
tstset = datasets.CIFAR10('data', train=False, transform=tr, target_transform=None, download=False)
