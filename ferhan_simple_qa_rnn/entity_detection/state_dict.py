import os
import sys
import torch
import pickle
import math
import unicodedata
import pandas as pd
import numpy as np

from args import get_args
from torchtext import data

args = get_args()
torch.manual_seed(args.seed)
torch.nn.Module.dump_patches = True

if not args.cuda:
    args.gpu = -1
if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have CUDA but not using it.")
if torch.cuda.is_available() and args.cuda:
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)

model = torch.load("model/model.pt", map_location=lambda storage,location: storage.cuda(args.gpu))
torch.save(model.state_dict(), "state_dict_model/model.pt")
