import torch_geometric.utils as PyG_utils
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_geometric.nn import MessagePassing
from gnn_baselines.got import Sinkhron,LogSinkhorn

class OTMaskGraph(nn.Module):
    def __init__(self, nlayers, isize, non_linearity, mlp_act, ot_config):
        super(OTMaskGraph, self).__init__()

        self.layers = nn.ModuleList()
        for _ in range(nlayers):
            self.layers.append(Diag(isize))
        self.Log_sinkhorn = LogSinkhorn(eps=0.1, thresh=0.1, max_iter=100, reduction=None)
        self.sinkhorn = Sinkhron(bs=None, beta=0.5, got_beta=0.1, is_uniform=True, got_iteration=20,
                                 got_wd_iteration=20, wd_iteration=10, args=ot_config)

        self.non_linearity = non_linearity
        self.mlp_act = mlp_act
        self.ot_config = ot_config

    def internal_forward(self, h):
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != (len(self.layers) - 1):
                if self.mlp_act == "relu":
                    h = F.relu(h)
                elif self.mlp_act == "tanh":
                    h = F.tanh(h)
        return h

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        b_nodes_fea, mask = PyG_utils.to_dense_batch(x=x, batch=batch)

        b_nodes_fea = self.internal_forward(b_nodes_fea)

        cost,similarity= self.Log_sinkhorn.cost_matrix_batch_torch(b_nodes_fea.transpose(2, 1), b_nodes_fea.transpose(2, 1),
                                                         mask=mask, add_eye_diag=True)
        wd, P, cost = self.sinkhorn(C=cost, bs=None, N_s=cost.shape[-1], N_t=cost.shape[-2], mask=mask)
        P = P.detach()
        P = (mask.sum(-1).unsqueeze(-1).unsqueeze(-1) * P)

        A = (P > 0.1).float()
        similarity = A*similarity
        # embeddings = self.internal_forward(features)
        # embeddings = F.normalize(embeddings, dim=1, p=2)
        # similarities = cal_similarity_graph(embeddings)
        # cossim = 1 - similarities + torch.eye(features.shape[0]).to(features.device)
        # trans = NeuralSinkhorn(cossim, beta=self.ot_beta, outer_iter=self.ot_iter)
        # return trans
        # embeddings = F.normalize(embeddings, dim=1, p=2)
        # similarities = cal_similarity_graph(embeddings)
        # similarities = ot_mask(similarities, self.ot_config)
        similarities = apply_non_linearity(similarity, self.non_linearity, 5)


        return similarities

class MLP_Diag(nn.Module):
    def __init__(self, nlayers, isize, k, knn_metric, non_linearity, sparse, mlp_act):
        super(MLP_Diag, self).__init__()

        self.layers = nn.ModuleList()
        for _ in range(nlayers):
            self.layers.append(Diag(isize))
        self.k = k
        self.knn_metric = knn_metric
        self.non_linearity = non_linearity
        self.sparse = sparse
        self.mlp_act = mlp_act

    def internal_forward(self, h):
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != (len(self.layers) - 1):
                if self.mlp_act == "relu":
                    h = F.relu(h)
                elif self.mlp_act == "tanh":
                    h = F.tanh(h)
        return h

    def forward(self, features):
        if self.sparse:
            embeddings = self.internal_forward(features)
            # rows, cols, values = knn_fast(embeddings, self.k, 1000)
            # rows_ = torch.cat((rows, cols))
            # cols_ = torch.cat((cols, rows))
            # values_ = torch.cat((values, values))
            # values_ = apply_non_linearity(values_, self.non_linearity, 5)
            # adj = dgl.graph((rows_, cols_), num_nodes=features.shape[0], device='cuda')
            # adj.edata['w'] = values_
            # return adj
        else:
            embeddings = self.internal_forward(features)
            embeddings = F.normalize(embeddings, dim=1, p=2)
            similarities = cal_similarity_graph(embeddings)
            similarities = top_k(similarities, self.k + 1)
            similarities = apply_non_linearity(similarities, self.non_linearity, 5)
            return similarities

class Diag(nn.Module):
    def __init__(self, input_size):
        super(Diag, self).__init__()
        self.W = nn.Parameter(torch.ones(input_size))
        self.input_size = input_size

    def forward(self, input):
        hidden = input @ torch.diag(self.W)
        return hidden

def top_k(raw_graph, K):
    values, indices = raw_graph.topk(k=int(K), dim=-1)
    assert torch.max(indices) < raw_graph.shape[-1]
    mask = torch.zeros(raw_graph.shape).to(raw_graph.device)
    mask[torch.arange(raw_graph.shape[0]).view(-1, 1), indices] = 1.

    mask.requires_grad = False
    sparse_graph = raw_graph * mask
    return sparse_graph

def apply_non_linearity(tensor, non_linearity, i):
    if non_linearity == 'elu':
        return F.elu(tensor * i - i) + 1
    elif non_linearity == 'relu':
        return F.relu(tensor)
    elif non_linearity == 'tanh':
        return F.tanh(tensor)
    else:
        return tensor


