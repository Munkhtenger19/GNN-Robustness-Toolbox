import torch.nn as nn

class BaseGNNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, **kwargs):
        super().__init__()
        # Placeholder - initialize layers in subclass
       
    def forward(self, x, edge_index):
        # Placeholder - define GNN logic in subclass
        raise NotImplementedError


