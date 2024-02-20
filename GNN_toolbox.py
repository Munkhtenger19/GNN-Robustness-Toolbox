import os
import torch
import torch.nn.functional as F
from torch.optim import Adam

from torch_geometric.nn import GCN, GAT, MLP
from torch_geometric.contrib.nn import GRBCDAttack, PRBCDAttack
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

data_dir = "./data"
os.makedirs(data_dir, exist_ok=True)

class GNNRobustness():
    def __init__(self, args):
        self.args = args
        self.available_gnn_architectures = {
            "GCN": GCN,
            "GAT": GAT,
            "MLP": MLP,
        }
        self.available_attack_strategies = {
            "GRBCDAttack": GRBCDAttack, 
            "PRBCDAttack": PRBCDAttack, 
        }
        
    def choose_available_gnn_architectures(self):
        print("Available GNN Architectures:")
        for name, architecture in self.available_gnn_architectures.items():
            print(f"{name}")
        self.model = self.available_gnn_architectures[self.args.architecture](in_channels = self.args.in_channels, hidden_channels = self.args.hidden_channels, num_layers = self.args.num_layers)
        print('sda', self.model)
        print('parameters', self.model)
        return self.model

    def choose_available_attack_strategies(self):
        print("Available Attack Strategies:")
        for name, strategy in self.available_attack_strategies.items():
            print(f"{name}: {strategy.__doc__}")
        attack_strategy_name = input("Enter the name of the attack strategy you want to use: ")
        self.attack_strategy = self.available_attack_strategies[attack_strategy_name]
        return self.attack_strategy
    
    def train_gnn(self, model, dataset, epochs, optimizer):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            output = model(dataset.x, dataset.edge_index)
            loss = F.nll_loss(output, dataset.y)
            loss.backward()
            optimizer.step()


