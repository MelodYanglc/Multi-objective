import torch
from torch import nn

class Total_Loss(nn.Module):
    def __init__(self,lossType='mse'):
        super(Total_Loss, self).__init__()
        if lossType == 'mse':
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.CrossEntropyLoss()

    def forward(self,model,pred,true):
        L1_reg = 0
        loss = self.loss(pred,true)
        for param in model.parameters():
            L1_reg = L1_reg + torch.sum(torch.abs(param))
        totalLoss = loss + 0.001 * L1_reg  # lambda=0.001
        return totalLoss

class Total_Loss2(nn.Module):
    def __init__(self,lossType='mse'):
        super(Total_Loss2, self).__init__()
        if lossType == 'mse':
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.CrossEntropyLoss()

    def forward(self,pred,true,mu,logvar):
        loss = self.loss(pred,true)
        KLD_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return loss + KLD_loss

class Total_LossL2(nn.Module):
    def __init__(self,lossType='mse'):
        super(Total_LossL2, self).__init__()

    def forward(self,pred,true):
        output = torch.sum((pred - true)**2)/pred.shape[0]
        return output