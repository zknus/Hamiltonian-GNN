
import torch
from torch import nn


def batch_jacobian(func, x, create_graph=False):
    # x in shape (Batch, Length)
    def _func_sum(x):
        return func(x).sum(dim=0)

    # return torch.autograd.functional.jacobian(_func_sum, x, create_graph=create_graph).permute(1, 2, 0)
    out = torch.autograd.functional.jacobian(_func_sum, x, create_graph=create_graph)
    # print('out in batch_jacobian: ',out.shape)
    return out.permute(1, 2, 0)


class myLinear(nn.Module):
    def __init__(self, size_in):
        super().__init__()
        self.dim = size_in
        self.linear1 = nn.Linear(in_features=2 * self.dim, out_features= self.dim)
        self.linear2 = nn.Linear(in_features=self.dim, out_features=1)

        # @@@@@@@@@@ add more layers is possbile  @@@@@@@@@@

    def forward(self, input_):
        out = self.linear1(input_)

        # out = torch.tanh(out) #######change to possible relu, tanh, etc.
        out = torch.tanh(out)
        # out = torch.sin(out)
        # out = torch.relu(out)
        # out = torch.sigmoid(out)
        out = self.linear2(out)

        # out = torch.tanh(out)
        # out = torch.relu(out)
        # out = torch.tanh(out)
        # out = torch.sigmoid(out)
        # out = torch.sin(out)

        return out

# class myLinear(nn.Module):
#     def __init__(self, size_in):
#         super().__init__()
#         self.dim = size_in
#         self.linear1 = nn.Linear(in_features=2 * self.dim, out_features=5 * self.dim)
#         self.linear3 = nn.Linear(in_features=5 * self.dim, out_features=2 * self.dim)
#         self.linear2 = nn.Linear(in_features=2 * self.dim, out_features=1)
#
#         # @@@@@@@@@@ add more layers is possbile  @@@@@@@@@@
#
#     def forward(self, input_):
#         out = self.linear1(input_)
#
#         # out = torch.sin(out) #######change to possible relu, tanh, etc.
#         out = torch.tanh(out)
#         out = self.linear3(out)
#         # out = torch.relu(out)
#         out = torch.tanh(out)
#         out = self.linear2(out)
#         # out = torch.relu(out)
#         # out = torch.tanh(out)
#         # out =  torch.sin(out)
#
#         return out


class Hamilton(nn.Module):
    def __init__(self, size_in):
        super().__init__()
        self.dim = size_in

        self.H = myLinear(self.dim)

    def forward(self,t,  input_):
        ### input_ should be 2xdim as [x, v], where x is manifold position and v is the tangent vector
        ### If you only have v, set x as 0
        ## x is q, v is p

        x = input_[..., 0:self.dim]
        v = input_[..., self.dim:]
        # print("input_shape:", input_.shape)
        H_derivatie = batch_jacobian(lambda xx: self.H(xx), input_, create_graph=True).squeeze()
        # print(H_derivatie.shape)

        # dx = H_derivatie[..., 0:self.dim]
        # dv = -1 * H_derivatie[..., self.dim:]

        dx = H_derivatie[..., self.dim:]
        dv = -1 * H_derivatie[..., 0:self.dim]

        # out = torch.hstack([dx, dv])

        out = torch.hstack([dv, dx])


        return out

class Hamilton_learn(nn.Module):
    def __init__(self, size_in,v):
        super().__init__()
        self.dim = size_in
        self.v=v
        self.H = myLinear(self.dim)

    def forward(self, t, input_):
        ### input_ should be 2xdim as [x, v], where x is manifold position and v is the tangent vector
        ### If you only have v, set x as 0

        # x = input_[..., 0:self.dim]
        # v = input_[..., self.dim:]

        # input_ = torch.cat((input_,self.v),dim=1)
        # x =input_
        # input_ = torch.hstack([x, self.v])
        # print("self.v:", self.v.shape)
        # print("input_shape:",input_.shape)
        # print("self.dim:", self.dim)
        H_derivatie = batch_jacobian(lambda xx: self.H(xx), input_, create_graph=True).squeeze()
        # print(H_derivatie.shape)

        # dx = H_derivatie[..., 0:self.dim]
        # dv = -1 * H_derivatie[..., self.dim:]

        dv = -1 * H_derivatie[..., 0:self.dim]
        dx = H_derivatie[..., self.dim:]

        out = torch.hstack([dx, dv])

        # out = torch.hstack([dv, dx])

        return out

# dim = 16
# con = Hamilton(dim)
# a = torch.zeros(128,dim*2)
# b = con(a)