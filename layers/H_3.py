
import torch
import torch.nn.functional as F
from torch import nn


#####################  non-decreasing convex act functions  ################
######### option 1 ############
#act = 0.5x^2 if x>0 and = exp(x)-1 if x<0
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
    
        
    
class my_act(nn.Module):
    def __init__(self, act):
        super(my_act, self).__init__()
        self.act = act
    
    def forward(self, x):
        return self.act(x)
    
    

class myLinear(nn.Module):
    def __init__(self, size_in,kdim):
        super().__init__()
        
        self.dim = size_in
        # self.act = my_act(act)
        # self.act = act2


        self.act = ReHU(d=0.1)
        #self.act = act1




        self.lambda_ = 0.1
        self.kdim = kdim
        max_ = +0.001
        min_ = -0.001
        
        w1_z = torch.Tensor(self.dim*self.kdim, self.dim)
        self.w1_z = nn.Parameter(w1_z)
        
        w2_z = torch.Tensor(self.dim, self.dim*self.kdim)
        self.w2_z = nn.Parameter(w2_z)
        
        
        ####### initial the parameters ###########
        # self.b1.data.uniform_(min_, max_)
        nn.init.xavier_uniform_(self.w1_z.data, gain=1.414)

        ####### initial the parameters ###########
        # self.b2.data.uniform_(min_, max_)
        nn.init.xavier_uniform_(self.w2_z.data, gain=1.414)



          
    def forward(self, input_):
        x = input_[...,0:self.dim]
        z = input_[...,self.dim:]
        # print("input_shape: ", input_.shape)
        # print("x shape: ", x.shape)
        # print("z shape: ", z.shape)
        
        w1_z = self.w1_z
        z_out = F.linear(x, w1_z)
        # print("zout shape: ", z_out.shape)
        z_out = self.act(z_out - self.lambda_*z)
        
    
        w2_z = self.w2_z
        x_out = F.linear(z, w2_z)
        x_out = self.act(x_out - self.lambda_*x)
        
        out = torch.hstack([x_out, z_out])
       
        
        return out
    
class Hamilton_V3(nn.Module):
    def __init__(self, size_in,kdim):
        super().__init__()
        self.dim = size_in
        self.kdim = kdim
        self.H = myLinear(self.dim,self.kdim)

        
    def forward(self,t, input_):
        ### input_ should be Kxdim as [x, v], where x is manifold position and v is the some (K-1)*dim vector
        ### please adjust K, now I choose K as 9.
        
        H_derivatie = self.H(input_)
        # print(H_derivatie.shape)
        
        out = H_derivatie
       
        return out

    
######## select convex activation function or @@@@@ other general activation functions @@@@@ ###########


if __name__=='__main__':
    act = act1
    dim = 16
    con = Hamilton_V3(dim)
    a = torch.zeros(128,dim*9)
    b = con(a)


##### can convert (K-1)*dim to dim use a linear.