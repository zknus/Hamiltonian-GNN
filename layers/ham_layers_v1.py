"""Hyperbolic layers."""
import math

import dgl.graph_index
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module

from layers.att_layers import DenseAtt,SpGraphAttentionLayer,GraphAttentionLayer
import torch.nn as nn
# from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint as odeint
import geotorch
from torch_geometric.utils import get_laplacian
from pytorch_block_sparse import BlockSparseLinear
import sparselinear as sl
# from layers.ode_map import Connection_g,ODEfunc, Connectionnew,Connection,Connection_gnew,Connection_ricci,Connection_riccilearn,Connectionxlearn,Connection_v5,Connection_v5new
from layers.ode_map import *
from layers.H_1 import Hamilton,Hamilton_learn
from layers.H_2 import Hamilton_V2
from layers.H_3 import Hamilton_V3
from layers.H_4 import Hamilton_V4
from layers.H_5 import Hamilton_V5
from layers.H_6 import Hamilton_V6
from layers.H_7 import Hamilton_V7
from layers.H_8 import Hamilton_V8
from layers.H_g_1 import Connection_geognn
from layers.ode_map import Connection_v5extend

from torch_geometric.nn import GATConv
from torch_geometric.utils import from_scipy_sparse_matrix
from dgl.nn.pytorch.conv import GATConv as GAT_DGL
from torch_geometric.nn import GCNConv

import scipy.sparse as sp

import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch_sparse
from torch_geometric.utils import softmax
import geotorch
class HAM_GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        # self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        # self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # # Step 2: Linearly transform node feature matrix.
        # x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)

        # # Step 6: Apply a final bias vector.
        # out += self.bias

        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j

def torch_sparse_to_coo(adj):
    m_index = adj._indices().cpu().numpy()
    row = m_index[0]
    col = m_index[1]
    data = adj._values().cpu().numpy()
    sp_matrix = sp.coo_matrix((data, (row, col)), shape=(adj.size()[0], adj.size()[1]))
    return sp_matrix

class ODEBlock(nn.Module):

    def __init__(self, odefunc,time,odemethod,step_size):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor(time).float()
        self.method = odemethod
        self.step_size = step_size

    def forward(self, x,):
        self.integration_time = self.integration_time.type_as(x)
        # out = odeint(self.odefunc, x, self.integration_time, rtol=1e-3, atol=1e-3,method='rk4',options={'step_size':0.1})
        out = odeint(self.odefunc, x, self.integration_time, rtol=1e-5, atol=1e-3, method=self.method,
                     options={'step_size': self.step_size})
        # out = odeint(self.odefunc, x, self.integration_time, rtol=1e-3, atol=1e-3, method='implicit_adams',
        #              options={'step_size': 0.5})
        # out = odeint(self.odefunc, x, self.integration_time, rtol=1e-3, atol=1e-3, method='dopri5',)
        # out = odeint(self.odefunc, x, self.integration_time, rtol=1e-3, atol=1e-3, method='dopri8',)

        return out[1]



class HamGraphConvolution(nn.Module):
    """
    Hamiltonian graph convolution layer.
    """

    def __init__(self,  out_features, args):
        super(HamGraphConvolution, self).__init__()
        self.agg = HamAgg(out_features, args.dropout, args.agg, args.n_nodes, args.odemap,args)
        self.device = args.device

    def forward(self, x, edge_index, edge_weight):

        h = self.agg.forward(x, edge_index,edge_weight)
        output = h
        return output



class HamAgg(Module):
    """
    Hamiltonian aggregation layer.
    """

    def __init__(self,  in_features, dropout, agg, n_nodes,odemethod,args):
        super(HamAgg, self).__init__()


        self.in_features = in_features
        self.dropout = dropout
        self.v0 =None
        self.agg = agg
        self.args = args
        if self.agg == 'GAT':

            # self.aggregator = GATConv(in_features, in_features, heads=args.n_heads,concat=False, dropout=args.dropout)
            self.aggregator = HAM_GATConv(in_features, in_features, heads=args.n_heads, device=args.cuda,concat=True)
        elif self.agg == 'GCN':
            # self.aggregator = GCNConv(in_features, in_features)
            self.aggregator = HAM_GCNConv(in_features, in_features)
        else:
            raise NotImplementedError

        self.odemethod = odemethod








        if self.odemethod is not None:





            if 'learn' in self.odemethod:
                self.v = nn.Parameter(torch.randn(n_nodes, in_features), requires_grad=True)

            elif self.odemethod == 'h3extend':
                self.weight_k = nn.Parameter(torch.Tensor(in_features * args.kdim, in_features))
                self.reset_parameters_weightk()

            elif self.odemethod == 'h3new':

                self.v_k = nn.Parameter(torch.randn(n_nodes, in_features * args.kdim), requires_grad=True)

            else:
                self.weight = nn.Parameter(torch.Tensor(in_features, in_features))
                self.bias = nn.Parameter(torch.Tensor(in_features))
                self.reset_parameters()

            if self.odemethod == 'v1':
                self.odefunc = Connection(int(in_features / 2))

            elif self.odemethod == 'v5learn':
                self.odefunc = Connection_v5new(in_features, self.v)
            elif self.odemethod == 'linear':
                self.odefunc = ODEfunc(in_features)
            elif self.odemethod == 'h1extend':
                self.odefunc = Hamilton(int(in_features))
            elif self.odemethod == 'h1learn':
                self.odefunc = Hamilton_learn(int(in_features), self.v)
            elif self.odemethod == 'h2extend':
                self.odefunc = Hamilton_V2(int(in_features))
            elif self.odemethod == 'h2learn':
                self.odefunc = Hamilton_V2(int(in_features))
            elif self.odemethod == 'h3extend':
                self.odefunc = Hamilton_V3(int(in_features),args.kdim)
            elif self.odemethod == 'h3new':
                self.odefunc = Hamilton_V3(int(in_features),args.kdim)
            elif self.odemethod == 'h4extend':
                self.odefunc = Hamilton_V4(int(in_features))
            elif self.odemethod == 'h4learn':
                self.odefunc = Hamilton_V4(int(in_features))
            elif self.odemethod == 'h5extend':
                self.odefunc = Hamilton_V5(int(in_features))
            elif self.odemethod == 'h5learn':
                self.odefunc = Hamilton_V5(int(in_features))
            elif self.odemethod == 'h6extend':
                self.odefunc = Hamilton_V6(int(in_features))
            elif self.odemethod == 'h6learn':
                self.odefunc = Hamilton_V6(int(in_features))
            elif self.odemethod == 'h7extend':
                self.odefunc = Hamilton_V7(int(in_features))
            elif self.odemethod == 'h7learn':
                self.odefunc = Hamilton_V7(int(in_features))
            elif self.odemethod == 'h8extend':
                self.odefunc = Hamilton_V8(int(in_features))
            elif self.odemethod == 'h8learn':
                self.odefunc = Hamilton_V8(int(in_features))
            elif self.odemethod == 'h9extend':
                self.odefunc = Connection_v5extend(int(in_features),args.sign)
            elif self.odemethod == 'geognn':
                self.odefunc = Connection_geognn(int(in_features),args.rank)



            else:
                print(" -----no ode func------- ")
            self.odeblock_exp = ODEBlock(self.odefunc, [0, 1],args.odemethod,args.step_size)


    # def reset_parameters(self):
    #     init.xavier_uniform_(self.v, gain=1)

    def forward(self, x, edge_index, edge_weight):
        #
        xt = x



        if 'learn' in self.odemethod:
            xt = torch.hstack([xt, self.v])
            out = self.odeblock_exp(xt, )
            out_x = out[..., 0:int(self.in_features)]
        elif self.odemethod == 'h3extend':
            drop_weight = self.weight_k

            vt = xt @ drop_weight.transpose(-1, -2)
            xt = torch.hstack([xt, vt])
            out = self.odeblock_exp(xt, )
            out_x = out[..., 0:int(self.in_features)]

        elif self.odemethod == 'h3new':

            xt = torch.hstack([xt, self.v_k])
            out = self.odeblock_exp(xt, )
            out_x = out[..., 0:int(self.in_features)]
        elif self.odemethod == 'linear':
            out_x = self.odeblock_exp(xt)

        else:
            if self.args.vt == 'fc':
                drop_weight = self.weight
                vt = xt @ drop_weight.transpose(-1, -2)
                xt = torch.hstack([xt, vt])
                out = self.odeblock_exp(xt)
                out_x = out[..., 0:int(self.in_features)]
            elif self.args.vt == 'clone':
                vt = xt.clone().detach()
                xt = torch.hstack([xt, vt])
                out = self.odeblock_exp(xt)
                out_x = out[..., 0:int(self.in_features)]






        x_tangent = out_x

        if self.agg == 'GCN':
            # print("edge_index: ", edge_index.shape)
            # print("edge_weight: ", edge_weight)
            # print("x_tangent: ", x_tangent.shape)
            # support_t = self.aggregator(x_tangent, edge_index).to(self.args.device)
            support_t = torch_sparse.spmm(edge_index, edge_weight, x_tangent.shape[0], x_tangent.shape[0], x_tangent)
        elif self.agg == 'GAT':
            attention = self.aggregator(x_tangent, edge_index).to(self.args.device)
            support_t = torch.mean(torch.stack(
                [torch_sparse.spmm(edge_index, attention[:, idx], x_tangent.shape[0], x_tangent.shape[0], x_tangent) for idx in
                 range(self.args.n_heads)], dim=0),
                dim=0)
        output = support_t
        # output_p = out[..., int(self.in_features):]
        # print("output shape: ", output.shape)
        return output


    def reset_parameters(self):
        # init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.eye_(self.weight)
        init.constant_(self.bias, 0)

    def reset_parameters_weightk(self):

        init.eye_(self.weight_k)







class HAM_GATConv(nn.Module):
  """
  Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
  """

  def __init__(self, in_features, out_features,heads, device, concat=True):
    super(HAM_GATConv, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.alpha = 0.2
    self.concat = concat
    self.device = device
    # self.opt = opt
    self.h = heads
    self.attention_norm_idx = 1
    # try:
    #   self.attention_dim = opt['attention_dim']
    # except KeyError:
    #   self.attention_dim = out_features
    self.attention_dim = out_features
    assert self.attention_dim % heads == 0, "Number of heads must be a factor of the dimension size"
    self.d_k = self.attention_dim // heads

    # self.W = nn.Parameter(torch.zeros(size=(in_features, self.attention_dim))).to(device)
    # nn.init.xavier_normal_(self.W.data, gain=1.414)

    # create one torch identity matrix
    self.I = torch.eye(self.in_features).to(device)

    # self.Wout = nn.Parameter(torch.zeros(size=(self.attention_dim, self.in_features))).to(device)
    # nn.init.xavier_normal_(self.Wout.data, gain=1.414)
    #
    self.a = nn.Parameter(torch.zeros(size=(2 * self.d_k, 1, 1))).to(device)
    nn.init.xavier_normal_(self.a.data, gain=1.414)

    self.leakyrelu = nn.LeakyReLU(self.alpha)

  def forward(self, x, edge):
    # wx = torch.mm(x, self.W)  # h: N x out
    wx = torch.mm(x, self.I)
    h = wx.view(-1, self.h, self.d_k)
    h = h.transpose(1, 2)

    # Self-attention on the nodes - Shared attention mechanism
    edge_h = torch.cat((h[edge[0, :], :, :], h[edge[1, :], :, :]), dim=1).transpose(0, 1).to(
      self.device)  # edge: 2*D x E
    edge_e = self.leakyrelu(torch.sum(self.a * edge_h, dim=0)).to(self.device)
    attention = softmax(edge_e, edge[self.attention_norm_idx])
    return attention


