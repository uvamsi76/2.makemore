import torch
import torch.nn as nn
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, output, target, num):
        loss=-output[torch.arange(num),target].log().mean()
        return loss

class ExpNormalizeLayer(nn.Module):
    def forward(self, x):
        x = x.exp()  # Apply exponential
        x = x / x.sum(1, keepdim=True)  # Normalize to make sum equal to 1
        return x
    
model=torch.nn.Sequential(
    torch.nn.Linear(28,28),
    ExpNormalizeLayer()
)
loss_fn=CustomLoss()
optimiser=torch.optim.SGD(model.parameters(),lr=10)