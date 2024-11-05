import torch
import torch.nn as nn
import pandas as pd
import re
import onnx
from onnx2pytorch import ConvertModel

from preproc import preproc
from generate import generate
import warnings
import argparse

parser = argparse.ArgumentParser(description="Process some inputs.")

parser.add_argument("num", type=int, help="Number of names")

args = parser.parse_args()

warnings.filterwarnings("ignore", category=UserWarning)

X,ys,stoi,itos=preproc('../data/IndianNames.txt')

onnx_model = onnx.load("../models/bigram_model.onnx")

pytorch_model = ConvertModel(onnx_model)

n=10
if(args.num):
    n=args.num
names=generate(pytorch_model,itos,n)

print(names)