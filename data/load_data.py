import hashlib
import json
import time
from enum import Enum
from pathlib import Path

import dgl
import networkx as nx
import numpy as np
import torch
from dgl.data import TUDataset
from scipy import sparse as sp

DATASETS_DIR = Path("datasets")
DATA_SPLITS_DIR = Path("data_splits")


class DatasetName(Enum):
    ZINC = "ZINC_full"
    DD = "DD"
    NCI1 = "NCI1"
    PROTEINS = "PROTEINS_full"
    ENZYMES = "ENZYMES"
    IMDB_BINARY = "IMDB-BINARY"
    IMDB_MULTI = "IMDB-MULTI"
    REDDIT_BINARY = "REDDIT-BINARY"
    REDDIT_MULTI = "REDDIT-MULTI-5K"
    COLLAB = "COLLAB"


def load_indexes(dataset_name: DatasetName):
    path = f"{DATA_SPLITS_DIR}/{dataset_name.value}.json"
    with open(path, "r") as f:
        indexes = json.load(f)
    return indexes


def self_loop(g):
    """
    Utility function only, to be used only when necessary as per user self_loop flag
    : Overwriting the function dgl.transform.add_self_loop() to not miss ndata['feat'] and edata['feat']


    This function is called inside a function in MoleculeDataset class.
    """
    new_g = dgl.DGLGraph()
    new_g.add_nodes(g.number_of_nodes())
    new_g.ndata["feat"] = g.ndata["feat"]

    src, dst = g.all_edges(order="eid")
    src = dgl.backend.zerocopy_to_numpy(src)
    dst = dgl.backend.zerocopy_to_numpy(dst)
    non_self_edges_idx = src != dst
    nodes = np.arange(g.number_of_nodes())
    new_g.add_edges(src[non_self_edges_idx], dst[non_self_edges_idx])
    new_g.add_edges(nodes, nodes)

    # This new edata is not used since this function gets called only for GCN, GAT
    # However, we need this for the generic requirement of ndata and edata
    new_g.edata["feat"] = torch.zeros(new_g.number_of_edges())
    return new_g


def make_full_graph(g):
    """
    Converting the given graph to fully connected
    This function just makes full connections
    removes available edge features
    """

    full_g = dgl.from_networkx(nx.complete_graph(g.number_of_nodes()))
    full_g.ndata["feat"] = g.ndata["feat"]
    full_g.edata["feat"] = torch.zeros(full_g.number_of_edges()).long()

    try:
        full_g.ndata["lap_pos_enc"] = g.ndata["lap_pos_enc"]
    except:
        pass

    try:
        full_g.ndata["wl_pos_enc"] = g.ndata["wl_pos_enc"]
    except:
        pass

    return full_g


def laplacian_positional_encoding(g, pos_enc_dim):
    """
    Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    g.ndata["lap_pos_enc"] = torch.from_numpy(EigVec[:, 1 : pos_enc_dim + 1]).float()

    return g


def wl_positional_encoding(g):
    """
    WL-based absolute positional embedding
    adapted from

    "Graph-Bert: Only Attention is Needed for Learning Graph Representations"
    Zhang, Jiawei and Zhang, Haopeng and Xia, Congying and Sun, Li, 2020
    https://github.com/jwzhanggy/Graph-Bert
    """
    max_iter = 2
    node_color_dict = {}
    node_neighbor_dict = {}

    edge_list = torch.nonzero(g.adj().to_dense() != 0, as_tuple=False).numpy()
    node_list = g.nodes().numpy()

    # setting init
    for node in node_list:
        node_color_dict[node] = 1
        node_neighbor_dict[node] = {}

    for pair in edge_list:
        u1, u2 = pair
        if u1 not in node_neighbor_dict:
            node_neighbor_dict[u1] = {}
        if u2 not in node_neighbor_dict:
            node_neighbor_dict[u2] = {}
        node_neighbor_dict[u1][u2] = 1
        node_neighbor_dict[u2][u1] = 1

    # WL recursion
    iteration_count = 1
    exit_flag = False
    while not exit_flag:
        new_color_dict = {}
        for node in node_list:
            neighbors = node_neighbor_dict[node]
            neighbor_color_list = [node_color_dict[neb] for neb in neighbors]
            color_string_list = [str(node_color_dict[node])] + sorted(
                [str(color) for color in neighbor_color_list]
            )
            color_string = "_".join(color_string_list)
            hash_object = hashlib.md5(color_string.encode())
            hashing = hash_object.hexdigest()
            new_color_dict[node] = hashing
        color_index_dict = {
            k: v + 1 for v, k in enumerate(sorted(set(new_color_dict.values())))
        }
        for node in new_color_dict:
            new_color_dict[node] = color_index_dict[new_color_dict[node]]
        if node_color_dict == new_color_dict or iteration_count == max_iter:
            exit_flag = True
        else:
            node_color_dict = new_color_dict
        iteration_count += 1

    g.ndata["wl_pos_enc"] = torch.LongTensor(list(node_color_dict.values()))
    return g


class SplitDataset(torch.utils.data.Dataset):
    def __init__(self, split, graphs, labels):
        self.split = split

        self.graph_lists = list(graphs)
        self.graph_labels = list(labels)
        self.n_samples = len(graphs)

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
        Get the idx^th sample.
        Parameters
        ---------
        idx : int
            The sample index.
        Returns
        -------
        (dgl.DGLGraph, int)
            DGLGraph with node feature stored in `feat` field
            And its label.
        """
        return self.graph_lists[idx], self.graph_labels[idx]


class GraphsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, train_idx, val_idx, test_idx):
        self.name = dataset_name.value
        start = time.time()
        print("[I] Loading dataset %s..." % (self.name))
        data_dir = f"data/{DATASETS_DIR}/{self.name}/"
        self.tu_dataset = TUDataset(self.name, raw_dir=data_dir)
        self.size = len(self.tu_dataset)
        self.max_num_node = self.tu_dataset.max_num_node
        self.num_classes = self.tu_dataset.num_labels
        self.num_edge_type = 4  # todo
        self.num_node_type = 28  # todo

        graphs, labels = self._create_dataset_from_indexes(train_idx)
        self.train = SplitDataset("train", graphs, labels)

        graphs, labels = self._create_dataset_from_indexes(val_idx)
        self.val = SplitDataset("val", graphs, labels)

        graphs, labels = self._create_dataset_from_indexes(test_idx)
        self.test = SplitDataset("test", graphs, labels)

        print(
            "train, test, val sizes :", len(self.train), len(self.test), len(self.val)
        )
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time() - start))

    def _create_dataset_from_indexes(self, indexes):
        graphs = []
        labels = []
        for idx in indexes:
            g, l = self.tu_dataset[idx]
            graphs.append(g)
            labels.append(l)
        return graphs, labels

    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels)).unsqueeze(1)
        batched_graph = dgl.batch(graphs)

        return batched_graph, labels

    def _add_self_loops(self):

        # function for adding self loops
        # this function will be called only if self_loop flag is True

        self.train.graph_lists = [self_loop(g) for g in self.train.graph_lists]
        self.val.graph_lists = [self_loop(g) for g in self.val.graph_lists]
        self.test.graph_lists = [self_loop(g) for g in self.test.graph_lists]

    def _make_full_graph(self):

        # function for converting graphs to full graphs
        # this function will be called only if full_graph flag is True
        self.train.graph_lists = [make_full_graph(g) for g in self.train.graph_lists]
        self.val.graph_lists = [make_full_graph(g) for g in self.val.graph_lists]
        self.test.graph_lists = [make_full_graph(g) for g in self.test.graph_lists]

    def _add_laplacian_positional_encodings(self, pos_enc_dim):

        # Graph positional encoding v/ Laplacian eigenvectors
        self.train.graph_lists = [
            laplacian_positional_encoding(g, pos_enc_dim)
            for g in self.train.graph_lists
        ]
        self.val.graph_lists = [
            laplacian_positional_encoding(g, pos_enc_dim) for g in self.val.graph_lists
        ]
        self.test.graph_lists = [
            laplacian_positional_encoding(g, pos_enc_dim) for g in self.test.graph_lists
        ]

    def _add_wl_positional_encodings(self):

        # WL positional encoding from Graph-Bert, Zhang et al 2020.
        self.train.graph_lists = [
            wl_positional_encoding(g) for g in self.train.graph_lists
        ]
        self.val.graph_lists = [wl_positional_encoding(g) for g in self.val.graph_lists]
        self.test.graph_lists = [
            wl_positional_encoding(g) for g in self.test.graph_lists
        ]
