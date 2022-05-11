import torch
from torch.nn import Linear
from torch import nn

import os
from os.path import join, dirname, abspath
import sys

import argparse

import multiprocessing
import numpy as np

from simple import DataParser, accuracyGNN
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.data import HeteroData

from os.path import join, dirname, abspath

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))
np.set_printoptions(linewidth=256, precision=4, suppress=True, threshold=sys.maxsize)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# getting the dataset
train_path = join(dirname(dirname(abspath(__file__))),'data', 'clutrr-emnlp', 'data_test', '64.csv')
test_path1 = join(dirname(dirname(abspath(__file__))),'data', 'clutrr-emnlp', 'data_db9b8f04', '1.10_test.csv')
#test_path2 = join(dirname(dirname(abspath(__file__))), 'data', 'clutrr-emnlp', 'data_db9b8f04', '1.2_test.csv')
test_path3 = join(dirname(dirname(abspath(__file__))), 'data', 'clutrr-emnlp', 'data_db9b8f04', '1.3_test.csv')
test_path4 = join(dirname(dirname(abspath(__file__))), 'data', 'clutrr-emnlp', 'data_db9b8f04', '1.4_test.csv')
test_path5 = join(dirname(dirname(abspath(__file__))), 'data', 'clutrr-emnlp', 'data_db9b8f04', '1.5_test.csv')
test_path6 = join(dirname(dirname(abspath(__file__))), 'data', 'clutrr-emnlp', 'data_db9b8f04', '1.6_test.csv')
test_path7 = join(dirname(dirname(abspath(__file__))), 'data', 'clutrr-emnlp', 'data_db9b8f04', '1.7_test.csv')
test_path8 = join(dirname(dirname(abspath(__file__))), 'data', 'clutrr-emnlp', 'data_db9b8f04', '1.8_test.csv')
test_path9 = join(dirname(dirname(abspath(__file__))), 'data', 'clutrr-emnlp', 'data_db9b8f04', '1.9_test.csv')
test_paths = [test_path1, test_path3, test_path4, test_path5, test_path6, test_path7, test_path8, test_path9]
# test_path2,
dataset = DataParser(train_path=train_path, test_paths=test_paths)
train_data = dataset.train_graph
test_datas = dataset.test_graphs

relation_lst = dataset.relation_lst
nb_relations = len(relation_lst)
nb_entities = len(dataset.entity_lst)
embedding_size = 20

loss_function = nn.BCELoss()

class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, nb_relations)  # instead of 1

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict['entity'][row], z_dict['entity'][col]], dim=-1)
        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.embeddings = nn.Embedding(nb_entities, embedding_size).to(device)  # sparse=True - Adam does not support it
        nn.init.uniform_(self.embeddings.weight, -1.0, 1.0)
        self.embeddings.requires_grad = False
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, train_data.metadata(), aggr='sum')
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        x_dict = {'entity': self.embeddings(x_dict['entity'])}
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)


model = Model(hidden_channels=32).to(device)

# Due to lazy initialization, we need to run one model step so the number of parameters can be inferred:
with torch.no_grad():
    x_dict = {'entity': model.embeddings(train_data.x_dict['entity'])}
    model.encoder(x_dict, train_data.edge_index_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()
    optimizer.zero_grad()
    pred = model(train_data.x_dict, train_data.edge_index_dict, train_data['entity', 'target', 'entity'].edge_index)
    pred = pred.clamp(min=0, max=1)
    print(f"pred shape: {pred.shape}")
    print(f"other shape: {train_data['entity', 'target', 'entity'].edge_index.shape}")
    target = torch.zeros(pred.shape[0], device=device)
    for i in range(len(train_data['entity', 'target', 'entity'].edge_label)):
        zero_idx = i*nb_relations
        class_num = train_data['entity', 'target', 'entity'].edge_label[i]
        target[zero_idx+class_num] = 1
    # TODO: one_hot-tal megcsinalni a targetet
    loss = loss_function(pred, target)
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(data):
    model.eval()
    pred = model(data.x_dict, data.edge_index_dict, data['entity', 'target', 'entity'].edge_index)
    pred = pred.clamp(min=0, max=1)
    target = torch.zeros(pred.shape[0], device=device)
    for i in range(len(data['entity', 'target', 'entity'].edge_label)):
        zero_idx = i * nb_relations
        class_num = data['entity', 'target', 'entity'].edge_label[i]
        target[zero_idx + class_num] = 1
    loss = loss_function(pred, target)
    return float(loss)


def evaluate_GNN(graph_data: HeteroData,
                 path: str) -> float:
    res = accuracyGNN(gnn_model=model,
                   graph_data=graph_data,
                   relation_lst=relation_lst)
    print(f'Test Accuracy on {path}: {res:.6f}')
    return res


#for epoch in range(1, 301):
#    loss = train()
#    train_rmse = test(train_data)
#    val_rmse = test(val_data)
#    test_rmse = test(test_data)
#    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_rmse:.4f}, '
#          f'Val: {val_rmse:.4f}, Test: {test_rmse:.4f}')

for epoch in range(1, 301):
    loss = train()
    print(f'Epoch: {epoch:03d}, Train: {loss:.4f}')
    evaluate_GNN(train_data, train_path)
    for test_path in test_paths:
        test_loss = test(test_datas[test_path])
        print(f'Test: {test_loss:.4f}')
        evaluate_GNN(test_datas[test_path], test_path)
