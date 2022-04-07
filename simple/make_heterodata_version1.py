import csv

import torch
import yaml
import json
from os.path import join, dirname, abspath

from collections import OrderedDict

from typing import List, Tuple, Any, Optional, Dict
from torch_geometric.data import HeteroData


# parses the data given from a csv specified by the path inputs
class DataParser:
    def __init__(self,
                 train_path: str = join(dirname(dirname(dirname(abspath(__file__)))), 'data', 'clutrr-emnlp',
                                        'data_089907f8', '1.2,1.3_train.csv'),
                 test_paths: Optional[List[str]] = None,
                 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

### Needed?
        with open(join(dirname(dirname(abspath(__file__))), "data", "clutrr-emnlp", "relations_store.yaml"), 'r') as f:
            rs = yaml.safe_load(f)
        self.relation_to_predicate = {r['rel']: k for k, v in rs.items()
                                      for _, r in v.items() if k != 'no-relation'}
        self.relation_lst = sorted({r for r in self.relation_to_predicate.keys()})
###
        self.device = device

        (self._train_graph, self._train_nodes_ids, self._train_edge_classes, new_idx) = \
            DataParser.parse(train_path, 0, self.device)
        entity_set = set(self.train_nodes_ids.keys())

        self._test_graphs = OrderedDict()
        self._test_nodes_ids = OrderedDict()
        self._test_edge_classes = OrderedDict()
        for test_path in (test_paths if test_paths is not None else []):
            (self._test_graphs[test_path], self._test_nodes_ids[test_path],
             self._test_edge_classes[test_path], new_idx) = DataParser.parse(test_path, new_idx, self.device)
            entity_set |= set(self.test_nodes_ids[test_path].keys())

        self.entity_lst = sorted(entity_set)


    @property
    def train_graph(self) -> HeteroData:
        return self._train_graph

    @property
    def train_nodes_ids(self) -> Dict[str, int]:
        return self._train_nodes_ids

    @property
    def train_edge_classes(self) -> Dict[str, int]:
        return self._train_edge_classes

    @property
    def test_graphs(self) -> OrderedDict[str, HeteroData]:
        return self._test_graphs

    @property
    def test_nodes_ids(self) -> OrderedDict[str, Dict[str, int]]:
        return self._test_nodes_ids

    @property
    def test_edge_classes(self) -> OrderedDict[str, int]:
        return self._test_edge_classes

    @staticmethod
    def _to_obj(s: str) -> Any:
        return json.loads(s.replace(")", "]").replace("(", "[").replace("'", "\""))

    # process the csv file at the given path
    # returning the graph (with target edges as well), the node names corresponding to their ids in the graph
    # and edge types (relations) corresponding to their class ids
    @staticmethod
    def parse(path: str, idx: int, device) -> (HeteroData, Dict[str, int], Dict[str, int]):
        data = HeteroData()
        with open(path, newline='') as f:
            reader = csv.reader(f)
            # idx: to add to names to make them unique for each mini graph
            name_to_id = {}  # dictionary to collect unique entities and their unique ids
            edges = [[], []]  # collecting edges of different types ([[subject_nodes],[object_nodes]])
            labels = []  # collecting edge labels for the edges above
            edge_types_to_class = {} # dictionary to collect relation types to corresponding class ids
            # target_edges = {}  # dictionary for collecting target edges (edge_type : [[subject_nodes],[object_nodes]])
            for row in reader:
                if len(name_to_id) > 7000:
                    break
                _id, _, _, query, _, target, _, _, _, _, _, story_edges, edge_types, _, genders, _, tmp, _ = row
                if len(_id) > 0:
                    # this many entities we have so far, so subsequent entities are numbered from there
                    add_to_id = len(name_to_id)
                    name_to_id |= {name.split(':')[0] + str(idx): (i + add_to_id) for i, name in
                                   enumerate(genders.split(','))}
                    _story, _edge, _query = DataParser._to_obj(story_edges), DataParser._to_obj(edge_types), DataParser._to_obj(query)
                    for (s_id, o_id), p in zip(_story, _edge):
                        if p not in edge_types_to_class.keys():
                            edge_types_to_class[p] = len(edge_types_to_class) # next id
                        edges[0].append(s_id + add_to_id)
                        edges[1].append(o_id + add_to_id)
                        labels.append(edge_types_to_class[p])
                    edges[0].append(name_to_id[_query[0]+str(idx)])
                    edges[1].append(name_to_id[_query[1]+str(idx)])
                    if target not in edge_types_to_class.keys():
                        edge_types_to_class[target] = len(edge_types_to_class)  # next id
                    labels.append(edge_types_to_class[target])

                idx += 1

            # TODO: embeddings instead of one hot encoding
            data['entity'].x = torch.eye(len(name_to_id), device=device)
            # data['entity'].x = torch.arange(0, len(name_to_id)).view(len(name_to_id), 1)  # just the indices
            data['entity', 'rel', 'entity'].edge_index = torch.tensor(edges, dtype=torch.long, device=device)
            data['entity', 'rel', 'entity'].edge_label = torch.tensor(labels, dtype=torch.long, device=device)

        return data, name_to_id, edge_types_to_class, idx
