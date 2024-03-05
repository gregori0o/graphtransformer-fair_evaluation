import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    Graph Transformer with edge features
    
"""
from layers.graph_transformer_edge_layer import GraphTransformerLayer
from layers.mlp_readout_layer import MLPReadout


class NodeEdgeEmbedding(nn.Module):
    def __init__(self, num_emb, out_dim):
        super(NodeEdgeEmbedding, self).__init__()

        if isinstance(num_emb, int):
            self.embedding = nn.Embedding(num_emb, out_dim)
        else:
            self.embedding = nn.ModuleList([nn.Embedding(n, out_dim) for n in num_emb])

    def forward(self, x):
        if isinstance(self.embedding, nn.Embedding):
            return self.embedding(x)
        result = 0
        for i in range(x.shape[1]):
            result += self.embedding[i](x[:, i])
        return result


class GraphTransformerNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        num_node_type = net_params["num_node_type"]
        num_edge_type = net_params["num_edge_type"]
        hidden_dim = net_params["hidden_dim"]
        num_heads = net_params["n_heads"]
        out_dim = net_params["out_dim"]
        in_feat_dropout = net_params["in_feat_dropout"]
        dropout = net_params["dropout"]
        n_layers = net_params["L"]
        self.readout = net_params["readout"]
        self.layer_norm = net_params["layer_norm"]
        self.batch_norm = net_params["batch_norm"]
        self.residual = net_params["residual"]
        self.edge_feat = net_params["edge_feat"]
        self.device = net_params["device"]
        self.lap_pos_enc = net_params["lap_pos_enc"]
        self.wl_pos_enc = net_params["wl_pos_enc"]
        max_wl_role_index = net_params["max_wl_role_index"]
        self.num_classes = net_params["num_classes"]

        if self.lap_pos_enc:
            pos_enc_dim = net_params["pos_enc_dim"]
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        if self.wl_pos_enc:
            self.embedding_wl_pos_enc = nn.Embedding(max_wl_role_index, hidden_dim)

        self.embedding_h = NodeEdgeEmbedding(num_node_type, hidden_dim)

        if self.edge_feat:
            self.embedding_e = NodeEdgeEmbedding(num_edge_type, hidden_dim)
        else:
            self.embedding_e = nn.Linear(1, hidden_dim)

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.layers = nn.ModuleList(
            [
                GraphTransformerLayer(
                    hidden_dim,
                    hidden_dim,
                    num_heads,
                    dropout,
                    self.layer_norm,
                    self.batch_norm,
                    self.residual,
                )
                for _ in range(n_layers - 1)
            ]
        )
        self.layers.append(
            GraphTransformerLayer(
                hidden_dim,
                out_dim,
                num_heads,
                dropout,
                self.layer_norm,
                self.batch_norm,
                self.residual,
            )
        )
        self.MLP_layer = MLPReadout(
            out_dim, self.num_classes
        )  # 1 out dim since regression problem

    def forward(self, g, h, e, h_lap_pos_enc=None, h_wl_pos_enc=None):

        # input embedding
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        if self.lap_pos_enc:
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float())
            h = h + h_lap_pos_enc
        if self.wl_pos_enc:
            h_wl_pos_enc = self.embedding_wl_pos_enc(h_wl_pos_enc)
            h = h + h_wl_pos_enc
        if not self.edge_feat:  # edge feature set to 1
            e = torch.ones(e.size(0), 1).to(self.device)
        e = self.embedding_e(e)

        # convnets
        for conv in self.layers:
            h, e = conv(g, h, e)
        g.ndata["h"] = h

        if self.readout == "sum":
            hg = dgl.sum_nodes(g, "h")
        elif self.readout == "max":
            hg = dgl.max_nodes(g, "h")
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, "h")
        else:
            hg = dgl.mean_nodes(g, "h")  # default readout is mean nodes

        return self.MLP_layer(hg)

    def loss(self, scores, targets):
        if self.num_classes == 1:
            loss = F.l1_loss(scores, targets)
            # loss = F.mse_loss(scores, targets)
        else:
            loss = F.l1_loss(
                scores, F.one_hot(targets.reshape(-1), self.num_classes).float()
            )
            # loss = F.cross_entropy(scores, targets.squeeze())
        return loss
