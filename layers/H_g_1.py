import torch
from torch import nn

def batch_jacobian(func, x, create_graph=False):
    # x in shape (Batch, Length)
    def _func_sum(x):
        return func(x).sum(dim=0)

    return torch.autograd.functional.jacobian(_func_sum, x, create_graph=create_graph).permute(1,2,0)

# class myLinear(nn.Module):  
#     def __init__(self, size_in):
#         super().__init__()
#         self.dim = size_in
#         self.linear1 = nn.Linear(in_features=2*self.dim,out_features=5*self.dim)
#         self.linear2 = nn.Linear(in_features=5*self.dim,out_features=1)
        
#         #@@@@@@@@@@ add more layers is possbile  @@@@@@@@@@
        
#     def forward(self, input_): 
#         out = self.linear1(input_)
        
#         out = torch.sin(out) #######change to possible relu, tanh, etc.
        
#         out = self.linear2(out)
        
#         out =  torch.sin(out) 
        
#         return out
    

class myLinear(nn.Module):  
    def __init__(self, size_in,rank=1):
        super().__init__()
        self.dim = size_in
        
        self.rank= rank
        
        self.linear1 = nn.Linear(in_features=self.dim,out_features=2* self.dim)
        self.linear2 = nn.Linear(in_features=2* self.dim,out_features=self.rank*self.dim)
        
        #@@@@@@@@@@ add more layers is possbile  @@@@@@@@@@
        
    def forward(self, input_): 
        
        x = input_[...,0:self.dim]
        v = input_[...,self.dim:]
        
        out = self.linear1(x)
        
        # out = torch.sin(out) #######change to possible relu, tanh, etc.
        out = torch.tanh(out)
        # out = torch.relu(out)
        out = self.linear2(out)
        out = torch.tanh(out).view(-1,self.dim, self.rank)
        # out =  torch.sin(out).view(-1,self.dim, self.rank)
        
        out = torch.einsum('mij,mhj,mi,mh->m', out, out, v, v)

        # out = torch.einsum('mij,mhj,mi,mh->m', out, out, x, x)
        
        
        return out.view(-1,1) 
    
    
class Connection_geognn(nn.Module):
    def __init__(self, size_in,rank):
        super().__init__()
        self.dim = size_in
        
        self.H = myLinear(self.dim,rank)

        
    def forward(self, t,input_):
        ### input_ should be 2xdim as [x, v], where x is manifold position and v is the tangent vector
        ### If you only have v, set x as 0
        
        x = input_[...,0:self.dim]
        v = input_[...,self.dim:]
        
        
        H_derivatie = batch_jacobian(lambda xx: self.H(xx), input_, create_graph=True).squeeze()
        # print(H_derivatie.shape)
        
        dx = H_derivatie[...,0:self.dim]
        dv = -1*H_derivatie[...,self.dim:]

        
        out = torch.hstack([dx, dv])
        # out = torch.hstack([dv, dx])
       
        
        return out
    
# dim = 16
# con = Connection(dim)
# a = torch.zeros(128,dim*2)
# b = con(a)