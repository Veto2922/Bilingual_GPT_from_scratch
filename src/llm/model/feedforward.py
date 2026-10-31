

import torch 
import torch.nn as nn



class FeedForward(nn.Module):

    def __init__(self, config, dropout=0.1):
        super().__init__()
        d_model = config.d_model
        d_ff = config.d_ff
        self.fc1 = nn.Linear(d_model,d_ff)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(d_ff,d_model)


    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        return self.fc2(x)