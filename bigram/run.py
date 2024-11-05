import torch
import torch.nn as nn
import pandas as pd
import re

from generate import generate
from train import train_bigram
from preproc import preproc

X,ys,stoi,itos=preproc('../data/IndianNames.txt')

model=train_bigram(X,ys)

names=generate(model,itos,10)

print(names)