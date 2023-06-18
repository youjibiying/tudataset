import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
import torch_geometric.utils as PyG_utils
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
import torch.nn as nn

num_atom_type = 120  # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6  # including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3

def get_sym(A):
    A = (A + A.transpose(2,1)) / 2
    A += torch.eye(A.shape[-1]).repeat(A.shape[0],1,1).to(A.device)
    D = A.sum(-1)
        # .squeeze(0)
    # D = torch.squeeze(D, 0)
    D_inv = D.pow(-1 / 2)
    D_inv[torch.isinf(D_inv)] = 0
    D_inv = D_inv.unsqueeze(-1)
    # D_inv = torch.diag(D_inv)
    # D = torch.diag(1 / torch.sqrt(D))
    A_norm = D_inv * A * D_inv.transpose(1, 2)
    return A_norm
    # return D @ A @ D  # A@D@A#

class GCNA(nn.Module):
    def __init__(self, in_size, hid_size, out_size, layers=2, sym=True, dropout=0.5,):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_size, hid_size))
        for i in range(1,layers-1):
            self.layers.append(nn.Linear(hid_size, hid_size))
        self.layers.append(nn.Linear(hid_size, out_size))
        self.sym = sym
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, A, features, mask=None):
        if self.sym:
            A = get_sym(A)
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(h)
            h = A@h
            # if i != len(self.layers)-1:
            h = self.dropout(F.relu(h))
        h = h[mask]
        # if self.dropout:
        #     h = self.dropout(h)




        # h = self.linear2(h)
        return h

    def concat_forward(self, A, features):
        if self.sym:
            A = get_sym(A)
        h = features
        h1 = A@h
        # h1 = self.linear1(h1)
        h2 = A@h1
        h = torch.concat([h,h1,h2],1)
        if self.dropout:
            h = self.dropout(h)
        h = F.relu(h)
        h = self.linear1(h)
        h = F.relu(h)
        h = self.linear2(h)
        return h

class GOTConv(MessagePassing):
    '''
    for pyg 1.6
    aggregate to node
    '''

    def __init__(self, emb_dim, out_dim=None, heads=2, negative_slope=0.2, aggr="add", args=None):
        super(GOTConv, self).__init__(aggr='add',
                                        node_dim=0)  # node_dim=0 indicate the aggregate dim of self,meassage output
        '''node_dim (int, optional): The axis along which to propagate.
            (default: :obj:`-2`)node_dim (int, optional): The axis along which to propagate.
            (default: :obj:`-2`)'''
        self.aggr = aggr

        self.emb_dim = emb_dim
        self.out_dim = emb_dim if out_dim is None else out_dim
        self.heads = heads
        self.negative_slope = negative_slope


        self.weight_linear = torch.nn.Linear(emb_dim, heads * self.out_dim)

        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * self.emb_dim))

        self.bias = torch.nn.Parameter(torch.Tensor(self.out_dim))
        self.graph_generator = OTMaskGraph(1, emb_dim, 'relu',
                                           'tanh', ot_config=args)

        # self.edge_embedding1 = torch.nn.Embedding(num_bond_type, heads * emb_dim)
        # self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, heads * emb_dim)

        # torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        # torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        if edge_attr is not None:
            self_loop_attr = self_loop_attr.type_as(edge_attr)
            edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

            edge_attr = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        # x = self.weight_linear(x)  # x中有一半300为head1, 300为head2
        # try:  # PyG 1.6.
        return self.propagate(edge_index[0], x=x, edge_attr=edge_attr, batch=batch)[-1]


    def message(self, edge_index, edge_index_i, x, x_i, x_j, edge_attr, batch):
        '''learnable distance  by attention'''
        # edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        # x_j = x_j.view(-1, self.heads, self.out_dim)
        # x_i = x_i.view(-1, self.heads, self.out_dim)
        # x_j = x_j + edge_attr

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        # similarity = F.leaky_relu(alpha, self.negative_slope)
        # alpha = softmax(similarity, edge_index_i)
        # dissimilarity = torch.exp(-similarity)
        self.ot_node_embs = self.get_transport_map(x, edge_index, batch)
        out = x_j * alpha.view(-1, self.heads, 1)
        # todo: cost = exp(1-\beta^{l} cosine(x_i,x_j))

        return out # .transpose(0,1)



    def update(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1)
        aggr_out = aggr_out + self.bias

        return aggr_out, self.ot_node_embs


    def get_transport_map(self, x, edge_index, batch, alpha=None):
        b_nodes_fea, mask = PyG_utils.to_dense_batch(x=x, batch=batch)
        # ------------ adj_matrix exists
        # cost = PyG_utils.to_dense_adj(edge_index, batch=batch, edge_attr=alpha)
        # cost = cost.squeeze(-1)# cost = cost.squeeze(-1)
        # A = PyG_utils.to_dense_adj(edge_index, batch=batch)
        # wd, P, C = self.Log_sinkhorn(x=b_nodes_fea, y=b_nodes_fea, A=A, C=cost, mask=mask)
        # ------------

        # cost,similarity= self.Log_sinkhorn.cost_matrix_batch_torch(b_nodes_fea.transpose(2, 1), b_nodes_fea.transpose(2, 1),
        #                                                  mask=mask, add_eye_diag=True)
        # cost = cost + torch.eye(features.shape[0]).to(features.device)
        # wd, P, cost = self.sinkhorn(C=cost, bs=None, N_s=cost.shape[-1], N_t=cost.shape[-2], mask=mask)
        # P=P.detach()
        # P = (mask.sum(-1).unsqueeze(-1).unsqueeze(-1) * P)
        #
        # A = (P>0.1).float()
        # A = get_sym(A)*similarity
        # todo: (1-lambda^(l)) * attn + lambda^{l}*alpha or (1-\lambda^(l)) * attn + \lambda^{l}*A
        # todo: similar to Block Modeling-Guided Graph Convolutional Neural Networks
        similarities=self.graph_generator(b_nodes_fea,mask)
        A= get_sym(similarities)
        b_node_embs = A.bmm(b_nodes_fea)
        b_node_embs = b_node_embs[mask]
        # edge_attr = P[edge_index]
        return b_node_embs

    def message_attn(self, edge_index, edge_index_i, x, x_i, x_j, edge_attr, batch):
        # edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j = x_j.view(-1, self.heads, self.out_dim)
        x_i = x_i.view(-1, self.heads, self.out_dim)
        # x_j = x_j + edge_attr

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        similarity = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(similarity, edge_index_i)
        dissimilarity = torch.exp(-similarity)
        self.ot_node_embs = self.get_transport_map(x, edge_index, batch, dissimilarity / dissimilarity.max())
        out = x_j * alpha.view(-1, self.heads, 1)
        # todo: cost = exp(1-\beta^{l} cosine(x_i,x_j))

        return out  # .transpose(0,1)

    def update_attn(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1)
        aggr_out = aggr_out + self.bias

        return aggr_out, self.ot_node_embs
        # return node_embs,






class KNNGCNConv(MessagePassing):

    def __init__(self, emb_dim, aggr="add",out_dim=None,k=4):
        super(KNNGCNConv, self).__init__()

        self.emb_dim = emb_dim

        self.linear = torch.nn.Linear(emb_dim, emb_dim)


        self.aggr = aggr

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                 device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        # edge_index, edge_weight = add_self_loops(edge_index, num_nodes=x.size(0)) pyg 1.6
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        # self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        # edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        # edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        try:
            norm = self.norm(edge_index, x.size(0), x.dtype)
        except:
            norm = self.norm(edge_index[0], x.size(0), x.dtype)

        x = self.linear(x)
        try:  # PyG 1.6.
            return self.propagate(edge_index[0], x=x, edge_attr=None, norm=norm)
        except:  # PyG 1.0.3
            return self.propagate(self.aggr, edge_index, x=x, edge_attr=None, norm=None)
        # return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings, norm=norm)
        # return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings, norm = norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)



class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not.


    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, emb_dim, aggr="add"):
        super(GINConv, self).__init__()
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(2 * emb_dim, emb_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        try:  # PyG 1.6.
            return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)
        except:  # PyG 1.0.3
            return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GCNConv(MessagePassing):

    def __init__(self, emb_dim, out_dim=None, aggr="add"):
        super(GCNConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, out_dim)
        # self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        # self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)
        #
        # torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        # torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                 device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        # edge_index, edge_weight = add_self_loops(edge_index, num_nodes=x.size(0)) pyg 1.6
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        # self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        # edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        # edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        edge_embeddings=None
        try:
            norm = self.norm(edge_index, x.size(0), x.dtype)
        except:
            norm = self.norm(edge_index[0], x.size(0), x.dtype)

        x = self.linear(x)
        try:  # PyG 1.6.
            return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings, norm=norm)
        except:  # PyG 1.0.3
            return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings, norm=norm)
        # return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings, norm=norm)
        # return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings, norm = norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j)


class GATConv(MessagePassing):
    '''
    for pyg 1.6
    aggregate to node
    '''

    def __init__(self, emb_dim, heads=2, negative_slope=0.2, aggr="add"):
        super(GATConv, self).__init__(aggr='add',
                                      node_dim=0)  # node_dim=0 indicate the aggregate dim of self,meassage output
        '''node_dim (int, optional): The axis along which to propagate.
            (default: :obj:`-2`)node_dim (int, optional): The axis along which to propagate.
            (default: :obj:`-2`)'''
        self.aggr = aggr

        self.emb_dim = emb_dim
        self.heads = heads
        self.negative_slope = negative_slope

        self.weight_linear = torch.nn.Linear(emb_dim, heads * emb_dim)
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))

        self.bias = torch.nn.Parameter(torch.Tensor(emb_dim))

        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, heads * emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, heads * emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.type_as(edge_attr)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        x = self.weight_linear(x)  # x中有一半300为head1, 300为head2
        # try:  # PyG 1.6.
        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, edge_index_i, x_i, x_j, edge_attr):
        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j = x_j.view(-1, self.heads, self.emb_dim)
        x_i = x_i.view(-1, self.heads, self.emb_dim)
        x_j = x_j + edge_attr

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i)
        out = x_j * alpha.view(-1, self.heads, 1)
        return out  # .transpose(0,1)

    def update(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1)
        aggr_out = aggr_out + self.bias

        return aggr_out


class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim, aggr="mean"):
        super(GraphSAGEConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        x = self.linear(x)

        try:  # PyG 1.6.
            return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)
        except:  # PyG 1.0.3
            return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p=2, dim=-1)







if __name__ == "__main__":
    pass
