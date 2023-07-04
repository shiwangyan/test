import torch
from torch import nn, optim
from torchvision import datasets, transforms
import numpy as np

a = torch.rand(3,3)
print(a)
b = a.add_(3)
print(a)
print(b)