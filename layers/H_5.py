
import torch
from torch import nn

def batch_jacobian(func, x, create_graph=False):
    # x in shape (Batch, Length)
    def _func_sum(x):
        return func(x).sum(dim=0)

    return torch.autograd.functional.jacobian(_func_sum, x, create_graph=create_graph).permute(1,2,0)

class myLinear(nn.Module):  
    def __init__(self, size_in):
        super().__init__()
        self.dim = size_in
        self.linear1 = nn.Linear(in_features=2*self.dim,out_features=5*self.dim)
        self.linear2 = nn.Linear(in_features=5*self.dim,out_features=1)
        
        #@@@@@@@@@@ add more layers is possbile  @@@@@@@@@@
         
    def forward(self, input_): 
        out = self.linear1(input_)
        
        # out = torch.sin(out) #######change to possible relu, tanh, etc.
        out = torch.tanh(out)
        out = self.linear2(out)
        out = torch.tanh(out)
        # out =  torch.sin(out)
        
        return out
    
class augment_f(nn.Module):  
    def __init__(self, size_in):
        super().__init__()
        self.dim = size_in
        self.linear1 = nn.Linear(in_features=self.dim,out_features=4*self.dim)
        self.linear2 = nn.Linear(in_features=4*self.dim,out_features=8*self.dim)
        self.linear3 = nn.Linear(in_features=8*self.dim,out_features=self.dim*self.dim)
        
        #@@@@@@@@@@ add more layers is possbile  @@@@@@@@@@
         
    def forward(self, input_): 
        out = self.linear1(input_)
        # out = torch.sin(out) #######change to possible relu, tanh, etc.
        out = torch.tanh(out)

        out = self.linear2(out)
        out = torch.tanh(out)
        # out =  torch.sin(out) #######change to possible relu, tanh, etc.
        
        out = self.linear3(out)
        # out =  torch.sin(out) #######change to possible relu, tanh, etc.
        out = torch.tanh(out)
        return out.view(-1,self.dim,self.dim)
    
class Hamilton_V5(nn.Module):
    def __init__(self, size_in):
        super().__init__()
        self.dim = size_in
        max_ = +0.001
        min_ = -0.001
        
        self.H = myLinear(self.dim)
        
        self.augment_f = augment_f(self.dim)
        
        w2_z = torch.Tensor(self.dim)
        self.u = nn.Parameter(w2_z)
        
        ####### initial the parameters ###########
        self.u.data.uniform_(min_, max_)

        
    def forward(self, t,input_):
        ### input_ should be 2xdim as [x, v], where x is manifold position and v is the tangent vector
        ### If you only have v, set x as 0
        
        x = input_[...,0:self.dim]
        v = input_[...,self.dim:]
        
        
        H_derivatie = batch_jacobian(lambda xx: self.H(xx), input_, create_graph=True).squeeze()
        # print(H_derivatie.shape)
        
        dx = H_derivatie[...,0:self.dim]
        dv = -1*H_derivatie[...,self.dim:]
        
        
        gq = self.augment_f(x)
        dv = dv + torch.torch.matmul(gq, self.u)

        
        out = torch.hstack([dx, dv])
       
        
        return out
    
# dim = 16
# con = Connection(dim)
# a = torch.zeros(128,dim*2)
# b = con(a)