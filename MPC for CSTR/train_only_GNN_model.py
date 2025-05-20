from pyomo.environ import *
from pyomo.dae import *
import math
import matplotlib.pyplot as plt
import numpy as np
import deepxde as dde
import sys
import os


N_data = int(sys.argv[1])
N_layer = int(sys.argv[2])
lr = float(sys.argv[3])
act = sys.argv[4]
batch_size = int(sys.argv[5])
N_hid = int(sys.argv[6])


import random
from pyomo.environ import *
import networkx as nx
from networkx.algorithms import bipartite

import matplotlib.pyplot as plt

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


# ----------------------------------------
#            Load data - with the appropriate number of data points
# ----------------------------------------
data_list = torch.load("dataset_list.pt")


# model
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        # torch.manual_seed(12345)
        self.conv1 = GCNConv(5, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.conv5 = GCNConv(hidden_channels, hidden_channels)
        self.conv6 = GCNConv(hidden_channels, hidden_channels)
        self.conv7 = GCNConv(hidden_channels, hidden_channels)
        self.conv8 = GCNConv(hidden_channels, hidden_channels)
        self.conv9 = GCNConv(hidden_channels, hidden_channels)
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch=None):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.tanh()
        x = self.conv2(x, edge_index)
        x = x.tanh()
        x = self.conv3(x, edge_index)
        x = x.tanh()
        x = self.conv4(x, edge_index)
        x = x.tanh()
        x = self.conv5(x, edge_index)
        x = x.tanh()
        x = self.conv6(x, edge_index)
        x = x.tanh()
        x = self.conv7(x, edge_index)
        x = x.tanh()
        x = self.conv8(x, edge_index)
        x = x.tanh()
        x = self.conv9(x, edge_index)
        x = x.tanh()
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        # x = F.dropout(x, p=0.4, training=self.training)
        x = self.lin1(x)
        x = x.tanh()
        x = self.lin2(x)
        x = x.tanh()
        return x


dataset = data_list.copy()
torch.manual_seed(14523)
# dataset = dataset.shuffle()
import random

random.seed(14523)
random.shuffle(dataset)
train_dataset = dataset[:1000]
test_dataset = dataset[1000:]
print(f"Number of training graphs: {len(train_dataset)}")
print(f"Number of test graphs: {len(test_dataset)}")


from torch_geometric.loader import DataLoader

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=200, shuffle=True)

import torch
import torch.optim as optim
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F

# Define the model (hidden_channels can be any value)
model = GCN(hidden_channels=N_hid)
# Set the model to training mode
model.train()

criterion = torch.nn.MSELoss()  # MSE loss for regression
# criterion = torch.nn.L1Loss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=lr)  # Learning rate can be adjusted

num_epochs = 5000  # Set the number of epochs you want to train the model for

loss_iter = []
for epoch in range(num_epochs):
    model.train()  # Ensure the model is in training mode
    total_loss = 0

    for data in train_loader:
        # Forward pass
        output = model(data.x, data.edge_index, data.batch)
        # Compute the loss (MSE)
        loss = criterion(output, data.y)
        total_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_iter.append(total_loss / len(train_loader))
    # Print the loss for every epoch
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")

model.eval()  # Set the model to evaluation mode
total_mse = 0
total_mae = 0
y_pred = []
y_test = []
with torch.no_grad():  # No need to compute gradients during evaluation
    for data in test_loader:
        # Forward pass
        output = model(data.x, data.edge_index, data.batch)
        y_pred = output
        y_test = data.y
        # Compute MSE and MAE
        mse = criterion(output, data.y)
        mae = torch.abs(output - data.y).mean()

        total_mse += mse.item()
        total_mae += mae.item()

# Compute average MSE and MAE for the test set
avg_mse = total_mse / len(test_loader)
avg_mae = total_mae / len(test_loader)

print(f"Test MSE: {avg_mse}")
print(f"Test MAE: {avg_mae}")


torch.save(
    model.state_dict(),
    "Model_deep_Nd_{}_Nl_{}_lr_{}_act_{}_BS_{}_hid_{}_MSE_{}_MAS_{}.pth".format(
        N_data,
        N_layer,
        lr,
        act,
        batch_size,
        N_hid,
        round(avg_mse, 2),
        round(avg_mae, 2),
    ),
)
