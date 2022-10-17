import torch
from torch import nn


class TuDui(nn.Module):
    def __init__(self):
        super(TuDui, self).__init__()

    def forward(self, input):
        output = input + 1
        return output


tudui = TuDui()
x = torch.tensor(1.0)
output = tudui(x)
print(output)
