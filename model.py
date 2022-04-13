import torch

class TinyModel(torch.nn.Module):
    def __init__(self):
        super(TinyModel, self).__init__()

    def forward(self, x):
        x += 0.1
        x = torch.clamp(x, 0, 1)
        return x
