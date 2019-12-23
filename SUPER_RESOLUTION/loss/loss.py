import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np

class Landmark_Loss(nn.Module):
    def __init__(self):
        super(Landmark_Loss, self).__init__()

    def forward(self, input, target):

        input = torch.sum(input, dim=1)
        # pdb.set_trace()
        # loss = torch.sum(torch.pow(((input.float() - target.float()).mul(weight)) ,2))/97.0
        loss = torch.pow(input.float() - target.float())
        return loss


class CrossEntropyLoss2d(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss2d, self).__init__()
        self.loss = nn.NLLLoss()

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs, 1), torch.squeeze(targets))
