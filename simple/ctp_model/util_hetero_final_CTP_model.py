import torch
from torch import Tensor
from torch_geometric.data import HeteroData


def get_neighbours_CTP(node_ids: Tensor,
                       graph_data: HeteroData) -> HeteroData:

    """Gives back the graph containing only the node_ids that are connected to the given node ids.

        Args:
            node_ids (Tensor): Node indices in graph_data to take the connected components of.
            graph_data (HeteroData): The data from which to take the given nodes.

        Returns:
            HeteroData of only the nodes that were in a connected component of at least one of the provided nodes.
    """

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

    res_graph['entity'].x = nodes_to_include
    res_graph['entity'].x, _ = res_graph['entity'].x.sort()

    for edge in graph_data.edge_types:
        edge_index = graph_data[edge].edge_index.T
        edges_to_include = select_edges(nodes_to_include, edge_index)
        res_graph[edge].edge_index = (edges_to_include.T[:, :, None] > res_graph['entity'].x[None, None, :]).sum(-1)

    return res_graph
