import torch
import torch.nn as nn
import torch.nn.functional as F

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_grl):
        ctx.lambda_grl = lambda_grl
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_grl * grad_output, None

class GRL(nn.Module):
    def __init__(self, lambda_grl=1.0):
        super().__init__()
        self.lambda_grl = lambda_grl
    
    def forward(self, x):
        return GradientReversalLayer.apply(x, self.lambda_grl)

class Encoder(nn.Module):
    def __init__(self, in_dim, hid=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hid), 
            nn.LeakyReLU(0.1)
        )
        self.priv = nn.Sequential(
            nn.Linear(in_dim, hid), 
            nn.LeakyReLU(0.1)
        )
    
    def forward(self, x):
        return self.shared(x), self.priv(x)

class AttnFusion(nn.Module):
    def __init__(self, hid=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3*hid, 64), 
            nn.Tanh(), 
            nn.Linear(64, 3)
        )
    
    def forward(self, s_mir, s_meth, s_mrna):
        s_cat = torch.cat([s_mir, s_meth, s_mrna], dim=1)
        alpha = torch.softmax(self.net(s_cat), dim=1)
        h = alpha[:,0:1]*s_mir + alpha[:,1:2]*s_meth + alpha[:,2:3]*s_mrna
        return h, alpha