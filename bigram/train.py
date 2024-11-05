import torch
from model import model,optimiser,loss_fn
from train import train_bigram
from preproc import preproc

def train_bigram(X,ys):
    num=ys.nelement()
    for _ in range(500):
        optimiser.zero_grad()
        # forward pass
        op=model(X)
        loss=loss_fn(op,ys,num)
        # backward pass
        loss.backward()
        optimiser.step()
        print(loss.data)
    dummy_input=torch.randn(1,28)
    torch.onnx.export(model, dummy_input, '../models/bigram_model.onnx', export_params=True, opset_version=11, do_constant_folding=True)
    return model

if __name__ == "__main__":
    
    X,ys,stoi,itos=preproc('../data/IndianNames.txt')

    model=train_bigram(X,ys)
