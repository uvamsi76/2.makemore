import torch
import torch.nn as nn
import pandas as pd
import re

def generate(model,itos,n):
    results=[]
    for i in range(n):
        out=[]
        ix=0
        while True:
            xin=nn.functional.one_hot(torch.tensor([ix]),num_classes=28).float()
            op=model(xin)
            ix=torch.multinomial(op,num_samples=1,replacement=True).item()
            out.append(itos[ix])
            if(ix==0):
                break
        name=''.join(out)
        results.append(name)
        print(name)
    return results