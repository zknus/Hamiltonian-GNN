
import torch
from torch import nn

import geotorch
from torch.nn.utils import spectral_norm
def batch_jacobian(func, x, create_graph=False):
    # x in shape (Batch, Length)
    def _func_sum(x):
        return func(x).sum(dim=0)

    # return torch.autograd.functional.jacobian(_func_sum, x, create_graph=create_graph).permute(1, 2, 0)
    return torch.autograd.functional.jacobian(_func_sum, x, create_graph=create_graph).permute(1, 0,2)
def proj( x, c=1):
    eps = {torch.float32: 1e-7, torch.float64: 1e-15}
    K = 1. / c
    d = x.size(-1) - 1
    y = x.narrow(-1, 1, d)
    y_sqnorm = torch.norm(y, p=2, dim=1, keepdim=True) ** 2
    mask = torch.ones_like(x)
    mask[:, 0] = 0
    vals = torch.zeros_like(x)
    vals[:, 0:1] = torch.sqrt(torch.clamp(K + y_sqnorm, min=eps[x.dtype]))
    return vals + mask * x

def proj_tan( u, x, c=1):
    eps = {torch.float32: 1e-7, torch.float64: 1e-15}
    K = 1. / c
    d = x.size(1) - 1
    ux = torch.sum(x.narrow(-1, 1, d) * u.narrow(-1, 1, d), dim=1, keepdim=True)
    mask = torch.ones_like(u)
    mask[:, 0] = 0
    vals = torch.zeros_like(u)
    vals[:, 0:1] = ux / torch.clamp(x[:, 0:1], min=eps[x.dtype])
    return vals + mask * u

class myLinear(nn.Module):  ############ a function to get metric g, it return g(x), which is batch x dim^2
    def __init__(self, size_in):
        super().__init__()
        self.dim = size_in
        self.linear1 = nn.Linear(in_features=self.dim, out_features= self.dim)

        self.linear2 = nn.Linear(in_features= self.dim, out_features=self.dim * self.dim)

        # self.linear3 = nn.Linear(in_features=self.dim * self.dim, out_features=self.dim * self.dim,bias=False)
        self.tanh = nn.Tanh()
        # self.normalization1 = nn.BatchNorm1d(self.dim)
        # self.normalization2 = nn.BatchNorm1d(self.dim * self.dim)
        # self.linear1 =spectral_norm(nn.Linear(in_features=self.dim, out_features=2 * self.dim))
        #
        # self.linear2  =spectral_norm(nn.Linear(in_features=2 * self.dim, out_features=self.dim * self.dim))

        # geotorch.orthogonal(self.linear1, "weight")
        # geotorch.orthogonal(self.linear2, "weight")

    def forward(self, input_):
        out = self.linear1(input_)

        out = self.tanh(out)  #######change to possible relu, tanh, etc.

        out = self.linear2(out).view(-1, self.dim, self.dim)

        out = out + out.permute(0, 2, 1)  ###### to make it be symmetric
        out = out.view(-1, self.dim * self.dim)

        # out = self.linear3(out)
        return out


class Connection_g(nn.Module):
    def __init__(self, size_in):
        super().__init__()
        self.dim = size_in
        self.g = myLinear(self.dim)
        self.linear3 = nn.Linear(in_features=self.dim * 2, out_features=self.dim * 2, bias=False)

    def forward(self,t, input_):
        ### input_ should be 2xdim as [x, v], where x is manifold position and v is the tangent vector
        ### If you only have v, set x as 0

        x = input_[..., 0:self.dim]
        v = input_[..., self.dim:]

        batchijacobian = batch_jacobian(lambda xx: self.g(xx), x, create_graph=True).view(-1, self.dim, self.dim,
                                                                                          self.dim)

        g = self.g(x).view(-1, self.dim, self.dim)
        # print("ode linear output: ",g)
        g_inv = torch.linalg.inv(g)

        connection = 0.5 * torch.einsum('nkl,nijl,njil,nlij->nkij', g_inv, batchijacobian, batchijacobian,
                                        -batchijacobian)
        # print("connection in ode output: ", connection)
        dx = v


        dv = torch.einsum('...i,...j->...ij', v, v)
        dv = - torch.einsum('hij,hkij->hk', dv, connection)
        # print("ode dv: ", dv)
        # print("dv in ode: ",dv)
        # print("dv in ode: ", dx)
        out = torch.hstack([dx, dv])

        # out = self.linear3(out)
        return out


class Connection_gnew(nn.Module):
    def __init__(self, size_in,v):
        super().__init__()
        self.dim = size_in
        self.g = myLinear(self.dim)
        self.linear3 = nn.Linear(in_features=self.dim * 2, out_features=self.dim , bias=False)
        self.v=v
    def forward(self,t, input_):
        ### input_ should be 2xdim as [x, v], where x is manifold position and v is the tangent vector
        ### If you only have v, set x as 0

        # x = input_[..., 0:self.dim]
        # v = input_[..., self.dim:]

        x=input_
        v = self.v

        batchijacobian = batch_jacobian(lambda xx: self.g(xx), x, create_graph=True).view(-1, self.dim, self.dim,
                                                                                          self.dim)

        g = self.g(x).view(-1, self.dim, self.dim)
        # print("ode linear output: ",g)
        g_inv = torch.linalg.inv(g)

        connection = 0.5 * torch.einsum('nkl,nijl,njil,nlij->nkij', g_inv, batchijacobian, batchijacobian,
                                        -batchijacobian)
        # print("connection in ode output: ", connection)
        dx = v


        dv = torch.einsum('...i,...j->...ij', v, v)
        dv = - torch.einsum('hij,hkij->hk', dv, connection)
        # print("ode dv: ", dv)
        # print("dv in ode: ",dv)
        # print("dv in ode: ", dx)
        out = torch.hstack([dx, dv])

        out = self.linear3(out)
        return out

class ODEfunc(nn.Module):

    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.dim =dim
        # self.linear1 = nn.Linear(in_features=self.dim, out_features=2* self.dim)
        #
        # self.linear2 = nn.Linear(in_features= 2*self.dim, out_features=self.dim * self.dim)
        self.linear1 = nn.Linear(in_features=dim, out_features=dim*2)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(in_features=dim*2, out_features=dim)
        self.tanh = nn.Tanh()

        # self.pe = positionalencoding1d(d_model=dim, length=dim).to('cuda:1')
        # self.weight = torch.nn.Parameter(torch.randn((dim, dim)), requires_grad=True)
        # self.mask = torch.ones((dim,dim))
        # self.mask = self.mask[:,:int(dim/2)] = 0

        # geotorch.orthogonal(self.linear1, "weight")
        # geotorch.orthogonal(self.linear1, "weight")
        # geotorch.symmetric(self.linear1, "weight")
        # geotorch.symmetric(self.linear1, "weight")
        # geotorch.positive_semidefinite(self.linear1, "weight")
        # geotorch.positive_semidefinite(self.linear1, "weight")

        # geotorch.low_rank(self.linear1,"weight",rank=dim)
        # geotorch.low_rank(self.linear2, "weight", rank=dim)
        # geotorch.sphere(self.linear1, "weight")
        # geotorch.sphere(self.linear1, "weight")
        # torch.nn.init.xavier_uniform(self.linear1.weight)
        # torch.nn.init.xavier_uniform(self.linear2.weight)
        # for para in self.linear1.parameters():
        #     para.requires_grad =False
        # for para in self.linear2.parameters():
        #     para.requires_grad =False

    def forward(self, t, x):
        # with torch.no_grad():

        out = self.linear1(x)
        # out = torch.matmul(x,self.weight)
        # out = out + torch.matmul(x,self.pe)
        # print(self.linear1.weight.shape)
        # print(out.shape)
        # out = self.dropout(out)
        out = self.tanh(out)
        # out = out.sin()
        out = self.linear2(out)
        # out = self.dropout(out)
        return out


class Connectionnew(nn.Module):
    def __init__(self, size_in,v):
        super().__init__()
        self.dim = size_in

        ######resize the connection coefﬁcients to (dim M)^3. that means for one point we have (dim M)^3 many functions.
        self.linear1 = nn.Linear(in_features=self.dim, out_features= self.dim)
        self.linear2 = nn.Linear(in_features= self.dim, out_features=pow(self.dim, 3))
        # self.tanh = nn.Tanh()
        self.tanh = nn.Sigmoid()
        # self.tanh = nn.ReLU()
        self.v=v
        self.linear3 = nn.Linear(in_features=self.dim*2, out_features= self.dim)
        self.dropout= nn.Dropout(p=0.2)

        # self.linear1 = spectral_norm(nn.Linear(in_features=self.dim, out_features= self.dim))
        # self.linear2 =spectral_norm( nn.Linear(in_features=self.dim, out_features=pow(self.dim, 3)))

        # self.normalization1 = nn.BatchNorm1d(self.dim)
        # self.normalization2 = nn.BatchNorm1d(pow(self.dim, 3))

        # self.linear1 = BlockSparseLinear(self.dim, int(self.dim*2),density=0.1)
        # self.linear2 = BlockSparseLinear(int(self.dim*2), pow(self.dim, 3),density=0.1)

        # self.linear1 = sl.SparseLinear(self.dim, int(self.dim ),sparsity=0.1)
        # self.linear2 = sl.SparseLinear(int(self.dim ), pow(self.dim, 3),sparsity=0.1 )

    def forward(self,t, input_):
        ### input_ should be 2xdim as [x, v], where x is manifold position and v is the tangent vector
        ### If you only have v, set x as 0

        # print("input shape: ", input_.shape)
        # connection = input_

        x = input_

        v = self.v
        # x = proj(x)
        # v = proj_tan(v, x)
        connection = self.linear1(x)
        # connection = x
        # print("connection shape: ", connection.shape)
        # print(input_[..., 0:self.dim])
        # connection = torch.sin(connection)
        # self.dropout(connection)
        connection = self.tanh(connection)
        # connection=self.normalization1(connection)

        connection = self.linear2(connection).view(-1, self.dim, self.dim, self.dim)
        # connection = self.tanh(connection)
        # connection = self.normalization2(connection)
        # self.dropout(connection)
        # print("connection shape: ", connection.shape)



        # print("v shape: ", v.shape)
        # print("x shape: ", x.shape)

        dx = v

        dv = torch.einsum('...i,...j->...ij', v, v)

        dv = - torch.einsum('hij,hijk->hk', dv, connection)
        # print("dv shape: ", dv.shape)

        out = torch.hstack([dx, dv])
        # print("out shape: ", out.shape)
        out = self.linear3(out)
        # self.dropout(out)
        return out

class Connection(nn.Module):
    def __init__(self, size_in):
        super().__init__()
        self.dim = size_in

        ######resize the connection coefﬁcients to (dim M)^3. that means for one point we have (dim M)^3 many functions.
        self.linear1 = nn.Linear(in_features=self.dim, out_features= self.dim)
        self.linear2 = nn.Linear(in_features= self.dim, out_features=pow(self.dim, 3))
        # self.tanh = nn.Tanh()
        self.tanh = nn.Sigmoid()

    def forward(self,t, input_):
        ### input_ should be 2xdim as [x, v], where x is manifold position and v is the tangent vector
        ### If you only have v, set x as 0
        x = input_[..., 0:self.dim]
        # print("input shape: ", input_.shape)
        connection = self.linear1(x)

        # print(input_[..., 0:self.dim])
        # connection = torch.sin(connection)

        connection = self.tanh(connection)

        connection = self.linear2(connection).view(-1, self.dim, self.dim, self.dim)
        connection = (connection + connection.permute(0, 1, 3, 2)) / 2

        v = input_[..., self.dim:]

        # print("v shape: ", v.shape)
        # print("x shape: ", x.shape)

        dx = v

        dv = torch.einsum('...i,...j->...ij', v, v)
        dv = - torch.einsum('hij,hijk->hk', dv, connection)

        out = torch.hstack([dx, dv])

        return out





global ricci
class Connection_ricci(nn.Module):
    def __init__(self, size_in):
        super().__init__()
        self.dim = size_in

        ######resize the connection coefﬁcients to (dim M)^3. that means for one point we have (dim M)^3 many functions.
        self.linear1 = nn.Linear(in_features=self.dim, out_features=5 * self.dim)
        self.linear2 = nn.Linear(in_features=5 * self.dim, out_features=pow(self.dim, 3))

    def forward(self,t, input_):
        ### input_ should be 2xdim as [x, v], where x is manifold position and v is the tangent vector
        ### If you only have v, set x as 0

        connection = self.linear1(input_[..., 0:self.dim])

        connection = torch.sin(connection)

        connection = self.linear2(connection).view(-1, self.dim, self.dim, self.dim)
        x = input_[..., 0:self.dim]
        v = input_[..., self.dim:]

        dx = v

        dv = torch.einsum('...i,...j->...ij', v, v)
        dv = - torch.einsum('hij,hijk->hk', dv, connection)

        out = torch.hstack([dx, dv])

        ####### scalar curvature #########
        global ricci
        Riemijkm = torch.einsum('...rjm,...iri->...jm', connection, connection) - torch.einsum('...rji,...irm->...jm',
                                                                                               connection, connection)
        ricci = torch.mean(torch.einsum('...ii', Riemijkm))

        ###########################

        return out


class Connection_riccilearn(nn.Module):
    def __init__(self, size_in,v):
        super().__init__()
        self.dim = size_in

        ######resize the connection coefﬁcients to (dim M)^3. that means for one point we have (dim M)^3 many functions.
        self.linear1 = nn.Linear(in_features=self.dim, out_features= self.dim)
        self.linear2 = nn.Linear(in_features= self.dim, out_features=pow(self.dim, 3))
        # self.tanh = nn.Tanh()
        self.tanh = nn.Sigmoid()
        self.v=v
        self.linear3 = nn.Linear(in_features=self.dim*2, out_features= self.dim)
        self.dropout= nn.Dropout(p=0.2)


    def forward(self,t, input_):
        ### input_ should be 2xdim as [x, v], where x is manifold position and v is the tangent vector
        ### If you only have v, set x as 0

        # print("input shape: ", input_.shape)
        # connection = input_
        connection = self.linear1(input_)
        # print("connection shape: ", connection.shape)
        # print(input_[..., 0:self.dim])
        # connection = torch.sin(connection)
        # self.dropout(connection)
        connection = self.tanh(connection)

        connection = self.linear2(connection).view(-1, self.dim, self.dim, self.dim)
        x = input_
        v = self.v
        dx = v

        dv = torch.einsum('...i,...j->...ij', v, v)

        dv = - torch.einsum('hij,hijk->hk', dv, connection)


        out = torch.hstack([dx, dv])

        out = self.linear3(out)

        ####### scalar curvature #########
        global ricci
        Riemijkm = torch.einsum('...rjm,...iri->...jm', connection, connection) - torch.einsum('...rji,...irm->...jm',
                                                                                               connection, connection)
        ricci = torch.mean(torch.einsum('...ii', Riemijkm))

        ###########################

        return out


class Connectionxlearn(nn.Module):
    def __init__(self, size_in,v):
        super().__init__()
        self.dim = size_in

        ######resize the connection coefﬁcients to (dim M)^3. that means for one point we have (dim M)^3 many functions.
        self.linear1 = nn.Linear(in_features=self.dim, out_features= self.dim)
        self.linear2 = nn.Linear(in_features= self.dim, out_features=pow(self.dim, 3))
        # self.tanh = nn.Tanh()
        self.tanh = nn.Sigmoid()
        self.x=v
        self.linear3 = nn.Linear(in_features=self.dim*2, out_features= self.dim)
        self.dropout= nn.Dropout(p=0.2)

        # self.linear1 = BlockSparseLinear(self.dim, int(self.dim*2),density=0.1)
        # self.linear2 = BlockSparseLinear(int(self.dim*2), pow(self.dim, 3),density=0.1)

        # self.linear1 = sl.SparseLinear(self.dim, int(self.dim ),sparsity=0.1)
        # self.linear2 = sl.SparseLinear(int(self.dim ), pow(self.dim, 3),sparsity=0.1 )

    def forward(self,t, input_):
        ### input_ should be 2xdim as [x, v], where x is manifold position and v is the tangent vector
        ### If you only have v, set x as 0

        # print("input shape: ", input_.shape)
        # connection = input_
        connection = self.linear1(self.x)
        # print("connection shape: ", connection.shape)
        # print(input_[..., 0:self.dim])
        # connection = torch.sin(connection)
        # self.dropout(connection)
        connection = self.tanh(connection)

        connection = self.linear2(connection).view(-1, self.dim, self.dim, self.dim)
        # self.dropout(connection)
        # print("connection shape: ", connection.shape)
        v = input_
        x = self.x

        # print("v shape: ", v.shape)
        # print("x shape: ", x.shape)

        dx = v

        dv = torch.einsum('...i,...j->...ij', v, v)

        dv = - torch.einsum('hij,hijk->hk', dv, connection)
        # print("dv shape: ", dv.shape)

        out = torch.hstack([dx, dv])
        # print("out shape: ", out.shape)
        out = self.linear3(out)
        # self.dropout(out)
        return out

import torch.nn.functional as F
class Linear_v5(nn.Module):  ############ a function to get metric g, it return g(x), which is a diagonal matrix
    def __init__(self, size_in,sign=4):
        super().__init__()
        self.dim = size_in
        self.linear1 = nn.Linear(in_features=self.dim, out_features=2 * self.dim)
        self.linear2 = nn.Linear(in_features=2 * self.dim, out_features=self.dim)
        self.act2 = nn.Sigmoid()  ###### this must be some function bounded below >= -K, can shift other act using + K to ensure >0
        self.constant = 0.618  ## add sth to avoid unstable when do inverse, try 0.1; 0.01 also

        # self.act1 = nn.ReLU()
        # self.act2 = nn.ReLU()
        # self.act2 = nn.Tanh()
        self.constant1 = 1.618
        self.sign = int(sign)
    def forward(self, input_):
        out = self.linear1(input_)

        # out = torch.sin(out) #######change to possible relu, tanh, etc.
        out = F.relu(out)

        out = self.linear2(out)

        out = self.act2(out) + self.constant

        # out = 0.00000001 * out + 1
        # out = self.act2(out) + self.constant1
        # out = torch.sin(out) + self.constant1
        # out = self.act2(out)
        # out = torch.clamp(out,min=0.4)
        out[:, :self.sign] = out[:, :self.sign] * -1
        return out

class Linear_v5_new(nn.Module):  ############ a function to get metric g, it return g(x), which is a diagonal matrix
    def __init__(self, size_in):
        super().__init__()
        self.dim = size_in
        self.linear1 = nn.Linear(in_features=self.dim, out_features=8 * self.dim)
        self.linear2 = nn.Linear(in_features=8 * self.dim, out_features=4*self.dim)
        self.linear3 = nn.Linear(in_features=4 * self.dim, out_features=self.dim)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()  ###### this must be some function bounded below >= -K, can shift other act using + K to ensure >0
        self.constant = 0.618  ## add sth to avoid unstable when do inverse, try 0.1; 0.01 also

        # self.act1 = nn.ReLU()
        # self.act2 = nn.ReLU()
        # self.act2 = nn.Tanh()
        self.constant1 = 1.618
    def forward(self, input_):
        out = self.linear1(input_)

        # out = torch.sin(out) #######change to possible relu, tanh, etc.
        # out = F.relu(out)
        out = torch.sin(out)
        out = self.linear2(out)
        out = torch.sin(out)
        out = self.linear3(out)
        out = torch.sin(out)+ self.constant1

        # out = 0.00000001 * out + 1
        # out = self.act2(out) + self.constant1
        # out = torch.sin(out) + self.constant1
        # out = self.act2(out)
        # out = torch.clamp(out,min=0.4)
        out[:, 0] = out[:, 0] * -1
        return out


class Connection_v5(nn.Module):
    def __init__(self, size_in):
        super().__init__()
        self.dim = size_in

        self.g = Linear_v5(self.dim)



    def forward(self, t,input_):
        ### input_ should be 2xdim as [x, v], where x is manifold position and v is the tangent vector
        ### If you only have v, set x as 0

        x = input_[..., 0:self.dim]
        v = input_[..., self.dim:]

        # x = input_
        # v = input_



        g = self.g(x)
        g_inv = 1 / g

        g_derivatie = batch_jacobian(lambda xx: self.g(xx), x, create_graph=True).view(-1, self.dim, self.dim)
        #permute to (0,2,1)
        # print(g_derivatie.shape)

        connection_qij_case1 = -1 * g_derivatie * g_inv[:, None, :]
        connection_qij_case2 = g_derivatie * g_inv[:, :, None]
        dx = v
        # vtemp_list = []
        #
        # dv = torch.einsum('...i,...j->...ij', v, v)
        # dv_diagonal = torch.diagonal(dv, dim1=1, dim2=2)
        #
        # for q in range(dv.shape[1]):
        #     vtemp1 = torch.sum(dv_diagonal * connection_qij_case1[..., q], dim=1)
        #     vtemp2 = torch.sum(dv[:, q, :] * connection_qij_case2[:, q, :], dim=1)
        #     vtemp3 = torch.sum(dv[:, :, q] * connection_qij_case2[:, q, :], dim=1)
        #     vtemp = vtemp1 + vtemp2 + vtemp3
        #
        #     vtemp_list.append(vtemp)
        #
        # dv = torch.stack(vtemp_list, dim=1)

        dv_vec = torch.einsum('...i,...j->...ij', v, v)
        dv_vec_diagonal = torch.diagonal(dv_vec, dim1=1, dim2=2)
        vtemp1_vec = torch.sum(dv_vec_diagonal[:, :, None] * connection_qij_case1, dim=1)
        vtemp2_vec = torch.einsum('hji,hji->hj', dv_vec, connection_qij_case2)
        vtemp3_vec = torch.einsum('hij,hji->hj', dv_vec, connection_qij_case2)

        dv_vec = vtemp1_vec + vtemp2_vec + vtemp3_vec
        dv = dv_vec

        out = torch.hstack([dx, dv])

        ######### i will add more to control curvature #######
        ######### currently, because of the metric we select, the curvature is ensured to be negative ##########
        # ####### scalar curvature #########
        # global ricci
        # Riemijkm = torch.einsum('...rjm,...iri->...jm', connection, connection) - torch.einsum('...rji,...irm->...jm', connection, connection)
        # ricci = torch.mean(torch.einsum('...ii', Riemijkm))
        ###########################

        return out

class Connection_v5new(nn.Module):
    def __init__(self, size_in,v):
        super().__init__()
        self.dim = size_in
        self.v=v
        self.g = Linear_v5(self.dim)
        self.linear3 = nn.Linear(in_features=self.dim * 2, out_features=self.dim)
        self.linear4 = nn.Linear(in_features=self.dim,out_features=self.dim,bias=False)

    def forward(self, t,input_):
        ### input_ should be 2xdim as [x, v], where x is manifold position and v is the tangent vector
        ### If you only have v, set x as 0

        # x = input_[..., 0:self.dim]
        # v = input_[..., self.dim:]

        x = input_
        v = self.v
        # v = self.linear4(x)


        g = self.g(x)
        g_inv = 1 / g

        g_derivatie = batch_jacobian(lambda xx: self.g(xx), x, create_graph=True).view(-1, self.dim, self.dim)
        #permute to (0,2,1)
        # print(g_derivatie.shape)

        connection_qij_case1 = -1 * g_derivatie * g_inv[:, None, :]
        connection_qij_case2 = g_derivatie * g_inv[:, :, None]

        vtemp_list = []

        dx = v

        # dv = torch.einsum('...i,...j->...ij', v, v)
        # dv_diagonal = torch.diagonal(dv, dim1=1, dim2=2)
        #
        # for q in range(dv.shape[1]):
        #     vtemp1 = torch.sum(dv_diagonal * connection_qij_case1[..., q], dim=1)
        #     vtemp2 = torch.sum(dv[:, q, :] * connection_qij_case2[:, q, :], dim=1)
        #     vtemp3 = torch.sum(dv[:, :, q] * connection_qij_case2[:, q, :], dim=1)
        #     vtemp = vtemp1 + vtemp2 + vtemp3
        #
        #     vtemp_list.append(vtemp)
        #
        #
        # dv = torch.stack(vtemp_list, dim=1)

        dv_vec = torch.einsum('...i,...j->...ij', v, v)
        dv_vec_diagonal = torch.diagonal(dv_vec, dim1=1, dim2=2)
        vtemp1_vec = torch.sum(dv_vec_diagonal[:, :, None] * connection_qij_case1, dim=1)
        vtemp2_vec = torch.einsum('hji,hji->hj', dv_vec, connection_qij_case2)
        vtemp3_vec = torch.einsum('hij,hji->hj', dv_vec, connection_qij_case2)

        dv_vec = vtemp1_vec + vtemp2_vec + vtemp3_vec
        dv = dv_vec


        out = torch.hstack([dx, dv])
        out = self.linear3(out)
        ##activate
        ######### i will add more to control curvature #######
        ######### currently, because of the metric we select, the curvature is ensured to be negative ##########
        # ####### scalar curvature #########
        # global ricci
        # Riemijkm = torch.einsum('...rjm,...iri->...jm', connection, connection) - torch.einsum('...rji,...irm->...jm', connection, connection)
        # ricci = torch.mean(torch.einsum('...ii', Riemijkm))
        ###########################

        return out



class mlpmap(nn.Module):  ############ a function to get metric g, it return g(x), which is a diagonal matrix
    def __init__(self, size_in):
        super().__init__()
        self.dim = size_in
        self.linear1 = nn.Linear(in_features=self.dim, out_features=2 * self.dim)
        self.linear2 = nn.Linear(in_features=2 * self.dim, out_features=self.dim)
        self.act1 = nn.ReLU()

    def forward(self, input_):
        out = self.linear1(input_)


        out = self.act1(out)
        out = self.linear2(out)
        return out

class Connection_v5extend(nn.Module):
    def __init__(self, size_in,sign):
        super().__init__()
        self.dim = size_in


        self.g = Linear_v5(self.dim,sign)
        # self.linear = nn.Linear(in_features=size_in*2,out_features=size_in,bias=False)
        # self.mlp = mlpmap(self.dim)
        # self.linear2 = nn.Linear(in_features=self.dim, out_features= self.dim)



    def forward(self, t,input_):
        ### input_ should be 2xdim as [x, v], where x is manifold position and v is the tangent vector
        ### If you only have v, set x as 0

        x = input_[..., 0:self.dim]
        v = input_[..., self.dim:]

        # x = input_
        # # v = input_
        # v =self.linear2(v)



        g = self.g(x)
        g_inv = 1 / g

        g_derivatie = batch_jacobian(lambda xx: self.g(xx), x, create_graph=False).view(-1, self.dim, self.dim)
        #permute to (0,2,1)
        # print(g_derivatie.shape)

        connection_qij_case1 = -1 * g_derivatie * g_inv[:, None, :]
        connection_qij_case2 = g_derivatie * g_inv[:, :, None]


        # print("g_derivatie",g_derivatie.shape)
        # print("g_inv",g_inv.shape)
        # print("connection_qij_case1",connection_qij_case1.shape)
        # print("connection_qij_case2",connection_qij_case2.shape)

        vtemp_list = []

        dx = v

        # dv = torch.einsum('...i,...j->...ij', v, v)
        # dv_diagonal = torch.diagonal(dv, dim1=1, dim2=2)
        #
        # for q in range(dv.shape[1]):
        #     vtemp1 = torch.sum(dv_diagonal * connection_qij_case1[..., q], dim=1)
        #     vtemp2 = torch.sum(dv[:, q, :] * connection_qij_case2[:, q, :], dim=1)
        #     vtemp3 = torch.sum(dv[:, :, q] * connection_qij_case2[:, q, :], dim=1)
        #     vtemp = vtemp1 + vtemp2 + vtemp3
        #
        #     vtemp_list.append(vtemp)
        #
        # dv = torch.stack(vtemp_list, dim=1)

        dv_vec = torch.einsum('...i,...j->...ij', v, v)
        dv_vec_diagonal = torch.diagonal(dv_vec, dim1=1, dim2=2)
        # print("dv_vec_diagonal",dv_vec_diagonal.shape)
        vtemp1_vec = torch.sum(dv_vec_diagonal[:, :, None] * connection_qij_case1, dim=1)
        # print("vtemp1_vec",vtemp1_vec.shape)
        vtemp2_vec = torch.einsum('hji,hji->hj', dv_vec, connection_qij_case2)
        vtemp3_vec = torch.einsum('hij,hji->hj', dv_vec, connection_qij_case2)

        dv_vec = vtemp1_vec + vtemp2_vec + vtemp3_vec
        dv = dv_vec

        out = torch.hstack([dx, dv])
        # out = torch.hstack([dv, dx])


        # out = self.linear(out)
        # print("out: ",out.shape)
        ######### i will add more to control curvature #######
        ######### currently, because of the metric we select, the curvature is ensured to be negative ##########
        # ####### scalar curvature #########
        # global ricci
        # Riemijkm = torch.einsum('...rjm,...iri->...jm', connection, connection) - torch.einsum('...rji,...irm->...jm', connection, connection)
        # ricci = torch.mean(torch.einsum('...ii', Riemijkm))
        ###########################

        return out


if __name__ == '__main__':
    dim = 10
    con = Connection_g(dim)
    a = torch.zeros(100, dim * 2)
    b = con(a)