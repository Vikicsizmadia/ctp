import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Linear

import torch_geometric.transforms as T
from simple import DataParser
# from torch_geometric.datasets import MovieLens
from torch_geometric.nn import SAGEConv, to_hetero

from os.path import join, dirname, abspath

use_weighted_loss = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# getting the dataset

train_path = join(dirname(dirname(abspath(__file__))),'data', 'clutrr-emnlp', 'data_db9b8f04', '1.2,1.3,1.4_train.csv')
test_path1 = join(dirname(dirname(abspath(__file__))),'data', 'clutrr-emnlp', 'data_db9b8f04', '1.10_test.csv')
test_path2 = join(dirname(dirname(abspath(__file__))), 'data', 'clutrr-emnlp', 'data_db9b8f04', '1.2_test.csv')
test_path3 = join(dirname(dirname(abspath(__file__))), 'data', 'clutrr-emnlp', 'data_db9b8f04', '1.3_test.csv')
test_path4 = join(dirname(dirname(abspath(__file__))), 'data', 'clutrr-emnlp', 'data_db9b8f04', '1.4_test.csv')
test_path5 = join(dirname(dirname(abspath(__file__))), 'data', 'clutrr-emnlp', 'data_db9b8f04', '1.5_test.csv')
test_path6 = join(dirname(dirname(abspath(__file__))), 'data', 'clutrr-emnlp', 'data_db9b8f04', '1.6_test.csv')
test_path7 = join(dirname(dirname(abspath(__file__))), 'data', 'clutrr-emnlp', 'data_db9b8f04', '1.7_test.csv')
test_path8 = join(dirname(dirname(abspath(__file__))), 'data', 'clutrr-emnlp', 'data_db9b8f04', '1.8_test.csv')
test_path9 = join(dirname(dirname(abspath(__file__))), 'data', 'clutrr-emnlp', 'data_db9b8f04', '1.9_test.csv')
test_paths = [test_path1, test_path2, test_path3, test_path4, test_path5, test_path6, test_path7, test_path8, test_path9]

dataset = DataParser(train_path=train_path, test_paths=test_paths)
train_data = dataset.train_graph
test_datas = dataset.test_graphs

#path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/MovieLens')
#dataset = MovieLens(path, model_name='all-MiniLM-L6-v2')
#data = dataset[0].to(device)

# Add user node features for message passing:
# print(len(dataset.train_nodes_ids))
# train_data['entity'].x = torch.eye(len(dataset.train_nodes_ids), device=device)

#for test_path in test_paths:
#    test_datas[test_path]['entity'].x = torch.eye(len(dataset.test_nodes_ids[test_path]), device=device)

# Add a reverse ('movie', 'rev_rates', 'user') relation for message passing:

# train_data = T.ToUndirected()(train_data)
# data = T.ToUndirected()(data)
# del train_data['entity', 'rel', 'entity'].edge_label  # Remove "reverse" label.

# Perform a link-level split into training, validation, and test edges:
#train_data, val_data, test_data = T.RandomLinkSplit(
#    num_val=0.1,
#    num_test=0.1,
#    neg_sampling_ratio=0.0,
#    edge_types=[('user', 'rates', 'movie')],
#    rev_edge_types=[('movie', 'rev_rates', 'user')],
#)(data)

# We have an unbalanced dataset with many labels for rating 3 and 4, and very
# few for 0 and 1. Therefore we use a weighted MSE loss.
if use_weighted_loss:
    weight = torch.bincount(train_data['entity', 'entity'].edge_label.int())
    weight = weight.max() / weight
else:
    weight = None


def weighted_mse_loss(pred, target, weight=None):
    weight = 1. if weight is None else weight[target].to(pred.dtype)
    return (weight * (pred - target.to(pred.dtype)).pow(2)).mean()


class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        print("1")
        x = self.conv1(x, edge_index).relu()
        print("2")
        x = self.conv2(x, edge_index)
        print("3")
        return x


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict['entity'][row], z_dict['entity'][col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, train_data.metadata(), aggr='sum')
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)


model = Model(hidden_channels=32).to(device)

# Due to lazy initialization, we need to run one model step so the number
# of parameters can be inferred:
with torch.no_grad():
    model.encoder(train_data.x_dict, train_data.edge_index_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()
    optimizer.zero_grad()
    pred = model(train_data.x_dict, train_data.edge_index_dict, train_data['entity', 'entity'].edge_index)
    target = train_data['entity', 'entity'].edge_label
    loss = weighted_mse_loss(pred, target, weight)
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(data):
    model.eval()
    pred = model(data.x_dict, data.edge_index_dict, data['entity', 'entity'].edge_index)
    pred = pred.clamp(min=0, max=5)
    target = data['entity', 'entity'].edge_label.float()
    rmse = F.mse_loss(pred, target).sqrt()
    return float(rmse)


for epoch in range(1, 301):
    loss = train()
    train_rmse = test(train_data)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_rmse:.4f}')
    for test_path in test_paths:
        test_rmse = test(test_datas[test_path])
        print(f'Test: {test_rmse:.4f}')