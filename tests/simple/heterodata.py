import unittest

import os
from os.path import join, dirname, abspath
import sys

import argparse

import multiprocessing
import numpy as np

import torch
from torch import nn, optim, Tensor
import torch.nn.functional as F

from ctp.util import make_batches
from ctp.clutrr import Fact, Data, Instance
from simple import DataParserCTP, accuracy, BatchNeuralKB, get_neighbours, BatchHoppy, GaussianKernel

# from ctp.clutrr.models import BatchNeuralKB

from ctp.reformulators import BaseReformulator
from ctp.reformulators import StaticReformulator
from ctp.reformulators import LinearReformulator
from ctp.reformulators import AttentiveReformulator
from ctp.reformulators import MemoryReformulator
from ctp.reformulators import NTPReformulator

# from ctp.kernels import BaseKernel, GaussianKernel
from ctp.regularizers import N2, N3, Entropy

from typing import List, Tuple, Dict, Optional

from torch_geometric.data import HeteroData


class TestHeteroData(unittest.TestCase):
    train_path = join(dirname(dirname(dirname(abspath(__file__)))), 'data', 'clutrr-emnlp', 'data_test', '64.csv')
    data = DataParserCTP(train_path=train_path)
    entity_lst, relation_lst = data.entity_lst, data.relation_lst

    def test_node_number(self):
        self.assertEqual(len(self.entity_lst), 63*3)

    def test_relation_number(self):
        self.assertEqual(len(self.relation_lst), 22)

    def test_relations(self):

        test_relation_lst = ["aunt", "brother", "brother-in-law", "daughter", "daughter-in-law", "father", "father-in-law",
                             "granddaughter", "grandfather", "grandmother", "grandson", "husband", "mother", "mother-in-law", "nephew", "niece",
                             "sister", "sister-in-law", "son", "son-in-law", "uncle", "wife"]

        self.assertEqual(self.relation_lst, test_relation_lst)


if __name__ == '__main__':
    unittest.main()
