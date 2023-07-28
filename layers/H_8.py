
import torch
from torch import nn
def b_inv(b_mat):
    eye = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat)
    b_inv = torch.linalg.solve(eye, b_mat)
    return b_inv


def batch_jacobian(func, x, create_graph=False):
    # x in shape (Batch, Length)
    def _func_sum(x):
        return func(x).sum(dim=0)

    return torch.autograd.functional.jacobian(_func_sum, x, create_graph=create_graph).permute(1,2,0)

class myLinear(nn.Module):  
    def __init__(self, size_in):
        super().__init__()
        self.dim = size_in
        self.linear1 = nn.Linear(in_features=2*self.dim,out_features=2*self.dim)
        self.linear2 = nn.Linear(in_features=2*self.dim,out_features=1)
        
        #@@@@@@@@@@ add more layers is possbile  @@@@@@@@@@
         
    def forward(self, input_): 
        out = self.linear1(input_)
        
        # out = torch.sin(out) #######change to possible relu, tanh, etc.
        out = torch.tanh(out)
        out = self.linear2(out)
        # #
        # out = torch.tanh(out)
        # out =  torch.sin(out)
        
        return out

    
class myLinear_W(nn.Module):  
    def __init__(self, size_in):
        super().__init__()
        self.dim = size_in
        self.linear1 = nn.Linear(in_features=2*self.dim,out_features=3*self.dim)
        self.linear2 = nn.Linear(in_features=3*self.dim,out_features=2*self.dim)
        
        #@@@@@@@@@@ add more layers is possbile  @@@@@@@@@@
         
    def forward(self, input_): 
        out = self.linear1(input_)
        
        # out = torch.sin(out) #######change to possible relu, tanh, etc.
        out = torch.tanh(out)
        out = self.linear2(out)
        # out = torch.tanh(out)
        # out =  torch.sin(out)
        
        return out
    
    
class Hamilton_V8(nn.Module):
    def __init__(self, size_in):
        super().__init__()
        self.dim = size_in
        
        self.H = myLinear(self.dim)
        self.W = myLinear_W(self.dim)

        
    def forward(self,t, input_):
        ### input_ should be 2xdim as [x, v], where x is manifold position and v is the tangent vector
        ### If you only have v, set x as 0
        
        x = input_[...,0:self.dim]
        v = input_[...,self.dim:]
        
        
        H_derivatie = batch_jacobian(lambda xx: self.H(xx), input_, create_graph=True).squeeze()
        # print(H_derivatie.shape)
        
        
        W = batch_jacobian(lambda xx: self.W(xx), input_, create_graph=True).squeeze()
        W = W - torch.transpose(W, 1,2)
        
        W = b_inv(W)
        
        out =  torch.einsum('hij,hj->hi', W, H_derivatie)
       
        
        return out
    
# dim = 16
# con = Connection(dim)
# a = torch.zeros(128,dim*2)
# b = con(a)