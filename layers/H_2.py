
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module

import torch.nn as nn
# from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint as odeint
import geotorch
import os



def act1(x): 
    act_main = torch.nn.ELU(inplace=False)
    return 0.5*(torch.pow(F.relu(x),2)+ act_main(x))

######### option 2 ############
#act = ln((exp(x)+1)/2) if x>0 and = -sqrt(|x-1|)+1 if x<0
def act2(xx): 
    m = 1.1 ##m >=1
    a = 0.1
    x = a*xx
    return -F.relu(torch.sqrt(torch.abs(torch.minimum(x-1,torch.tensor(0)))+1e-8)-1) + m*torch.log((torch.exp(F.relu(x))+1)/2+1e-8)

######### option 3 ############
#act = x^m+0.5x if x>0 and = -sqrt(|x-1|)+1 if x<0
def act3(xx): 
    m = 3 ##m>=2
    a = 0.1
    x = a*xx
    return -F.relu(torch.sqrt(torch.abs(torch.minimum(x-1,torch.tensor(0)))+1e-8)-1) + torch.pow(F.relu(x),m)+ 0.5*F.relu(x)


class my_act(nn.Module):
    def __init__(self, act):
        super(my_act, self).__init__()
        self.act = act
    
    def forward(self, x):
        return self.act(x)

#####################   pos_constraint for weight   ################

######### option 1 ############
# def pos_constraint(x):
# #     act = torch.sigmoaid()
#     return torch.sigmoid(x)*0.001

######### option 2 ############
def pos_constraint(x):
#     act = torch.sigmoaid()
    return torch.abs(x)


class ReHU(nn.Module):
    """ Rectified Huber unit"""
    def __init__(self, d):
        super().__init__()
        self.a = 1/d
        self.b = -d/2

    def forward(self, x):
        return torch.max(torch.clamp(torch.sign(x)*self.a/2*x**2,min=0,max=-self.b),x+self.b)
    
        
    

    
    
def batch_jacobian(func, x, create_graph=False):
    # x in shape (Batch, Length)
    def _func_sum(x):
        return func(x).sum(dim=0)

    return torch.autograd.functional.jacobian(_func_sum, x, create_graph=create_graph).permute(1,2,0)

class myLinear(nn.Module):
    def __init__(self, size_in):
        super().__init__()
                 
        ## input --> f0 (2*dim, 2*dim) --> f1 (2*dim, 8*dim) --> f2 (8*dim, 2*dim) --> f3 (2*dim, 1)
        #### from f1 to f3, we need the weight to be postive, that's why we call pos_constraint() below
        ##### all activation function need to be convex and non-decreasing #######
        
 
        self.dim = size_in
        
        # self.act = my_act(act1)
        # self.act = act2
        self.act = ReHU(d=0.1)

        
        max_ = +0.001
        min_ = -0.001
        # max_ = +0.0002
        # min_ = -0.0002
        
        w1_z = torch.Tensor(self.dim*8, self.dim*2)
        self.w1_z = nn.Parameter(w1_z)
        b1 = torch.Tensor(self.dim*8)
        self.b1 = nn.Parameter(b1)
        
        w2_z = torch.Tensor(self.dim*2, self.dim*8)
        self.w2_z = nn.Parameter(w2_z)
        b2 = torch.Tensor(self.dim*2)
        self.b2 = nn.Parameter(b2)
        
        w3_z = torch.Tensor(1, self.dim*2)
        self.w3_z = nn.Parameter(w3_z)
        b3 = torch.Tensor(1)
        self.b3 = nn.Parameter(b3)
        
        self.w0_y = nn.Linear(in_features=2*self.dim,out_features=2*self.dim) ####w0_y is free, and no constraints
        
        
        ####### initial the parameters ###########
        self.b1.data.uniform_(min_, max_)
        self.w1_z.data.uniform_(min_, max_)
        # nn.init.xavier_uniform_(self.w1_z.data, gain=1.414)
        # nn.init.kaiming_normal_(self.w1_z.data, mode="fan_out", nonlinearity="relu")

        # torch.nn.init.normal_(self.w1_z)
        # torch.nn.init.normal_(self.b1)

        # torch.nn.init.xavier_uniform_(self.w1_z, gain=1.0)
        # # torch.nn.init.xavier_uniform_(self.b1, gain=1.0)
        # torch.nn.init.constant_(self.b1, val=0.001)

        ####### initial the parameters ###########
        self.b2.data.uniform_(min_, max_)
        self.w2_z.data.uniform_(min_, max_)
        # nn.init.xavier_uniform_(self.w2_z.data, gain=1.414)
        # nn.init.kaiming_normal_(self.w2_z.data, mode="fan_out", nonlinearity="relu")
        # torch.nn.init.normal_(self.w2_z)
        # torch.nn.init.normal_(self.b2)

        # torch.nn.init.xavier_uniform_(self.w2_z, gain=1.0)
        # # torch.nn.init.xavier_uniform_(self.b2, gain=1.0)
        # torch.nn.init.constant_(self.b2, val=0.001)

        ####### initial the parameters ###########
        self.b3.data.uniform_(min_, max_)
        self.w3_z.data.uniform_(min_, max_)
        # nn.init.xavier_uniform_(self.w3_z.data, gain=1.414)
        # nn.init.kaiming_normal_(self.w3_z.data, mode="fan_out", nonlinearity="relu")
        # torch.nn.init.normal_(self.w3_z)
        # torch.nn.init.normal_(self.b3)

        # torch.nn.init.xavier_uniform_(self.w3_z, gain=1.0)
        # # torch.nn.init.xavier_uniform_(self.b3, gain=1.0)
        # torch.nn.init.constant_(self.b3, val=0.001)



          
    def forward(self, x):
        z1 = self.act(self.w0_y(x))
    
        w1_z = pos_constraint(self.w1_z)
        z2 = F.linear(z1, w1_z, bias=self.b1)
        z2 = self.act(z2)
        
    
        w2_z = pos_constraint(self.w2_z)
        z3 = F.linear(z2, w2_z, bias=self.b2)
        z3 = self.act(z3)
        
    
        w3_z = pos_constraint(self.w3_z)
        z4 = F.linear(z3, w3_z, bias=self.b3)
        z4 = self.act(z4)
        
        
        f = z4
        
        return f
    
class Hamilton_V2(nn.Module):
    def __init__(self, size_in):
        super().__init__()
        self.dim = size_in
        
        self.H = myLinear(self.dim)

        
    def forward(self,t, input_):
        ### input_ should be 2xdim as [x, v], where x is manifold position and v is the tangent vector
        ### If you only have v, set x as 0
        
        x = input_[...,0:self.dim]
        v = input_[...,self.dim:]
        
        
        H_derivatie = batch_jacobian(lambda xx: self.H(xx), input_, create_graph=True).squeeze()
        # print(H_derivatie.shape)
        
        dx = H_derivatie[...,0:self.dim]
        dv = -1*H_derivatie[...,self.dim:]

        
        out = torch.hstack([dx, dv])
       
        
        return out

if __name__=='__main__':
    ######## select convex activation function ###########
    act = act1


    dim = 16
    con = Hamilton_V2(dim)
    a = torch.zeros(128,dim*2)
    b = con(a)