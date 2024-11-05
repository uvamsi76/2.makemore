import torch
import torch.nn as nn
import pandas as pd
import re
def preproc(data_path):        
    Wo=open(data_path,'r').read().splitlines()
    words=[re.sub(r'[^a-z ]', ' ', w.lower()) for w in Wo]
    
    chars=sorted(list(set(''.join(words))))
    stoi={s:i+1 for i,s in enumerate(chars)}
    stoi['.']=0
    itos={i:s for s,i in stoi.items()}
    
    xs,ys=[],[]
    for w in words:
        name=['.']+list(w)+['.']
        for x,y in zip(name,name[1:]):
            xs.append(stoi[x])
            ys.append(stoi[y])
    
    xs=torch.tensor(xs)
    ys=torch.tensor(ys)
    X=torch.nn.functional.one_hot(xs,num_classes=28).float()

    return X,ys,stoi,itos