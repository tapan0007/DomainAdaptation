import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from operator import itemgetter
from collections import OrderedDict

import cv2
from torch.nn import functional as F
#from pytorch_grad_cam import GradCAM
import torch.nn.init as init

from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch import optim, nn
import torch.nn.functional as F
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torchvision.utils import make_grid

#pd.options.plotting.backend = "plotly"
#pd.set_option("plotting.backend", "plotly")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)