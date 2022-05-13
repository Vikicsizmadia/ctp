import copy

from typing import List, Tuple, Dict, Optional

import torch
from torch import nn, Tensor
from torch_geometric.data import HeteroData


# gives back the graph containing only the node_ids that are connected to the given node ids
def get_neighbours(node_ids: Tensor,
                   graph_data: HeteroData,
                   device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')) -> (HeteroData, List[int]):

    res_graph = HeteroData()

    def select_edges(nodes_to_include, edge_index):
        edges_to_include = (edge_index[:, :, None] == nodes_to_include[None, None, :]).any(1).any(1)
        return edge_index[edges_to_include]

    def select_nodes(edge_index):
        return edge_index.flatten()

    nodes_to_include = node_ids  # tensor of node ids
    num_nodes = -1
    while num_nodes != nodes_to_include.shape[0]:
        num_nodes = nodes_to_include.shape[0]
        for edge in graph_data.edge_types:
            edge_index = graph_data[edge].edge_index.T
            edges_to_include = select_edges(nodes_to_include, edge_index)
            new_include = select_nodes(edges_to_include)
            nodes_to_include = torch.unique(torch.cat((nodes_to_include, new_include)))

    # print(f"new node_ids are: {nodes_to_include}")
    res_graph['entity'].x = nodes_to_include
    res_graph['entity'].x, _ = res_graph['entity'].x.sort()

    for edge in graph_data.edge_types:
        edge_index = graph_data[edge].edge_index.T
        edges_to_include = select_edges(nodes_to_include, edge_index)
        res_graph[edge].edge_index = (edges_to_include.T[:, :, None] > res_graph['entity'].x[None, None, :]).sum(-1)

    entity_list = sorted(nodes_to_include.tolist())

    return res_graph, entity_list
