
import torch
from torch import nn
import torch.nn.functional as F
def batch_jacobian(func, x, create_graph=False):
    # x in shape (Batch, Length)
    def _func_sum(x):
        return func(x).sum(dim=0)

    return torch.autograd.functional.jacobian(_func_sum, x, create_graph=create_graph).permute(1,0,2)

class myLinear(nn.Module):  ############ a function to get metric g, it return g(x), which is a diagonal matrix
    def __init__(self, size_in):
        super().__init__()
        self.dim = size_in
        self.linear1 = nn.Linear(in_features=self.dim,out_features=2*self.dim)
        self.linear2 = nn.Linear(in_features=2*self.dim,out_features=self.dim)
        self.act2 = nn.Sigmoid() ###### this must be some function bounded below >= -K, can shift other act using + K to ensure >0
        self.constant = 0.618    ## add sth to avoid unstable when do inverse, try 0.1; 0.01 also
        
        
    def forward(self, input_): 
        out = self.linear1(input_)
        
        # out = torch.sin(out) #######change to possible relu, tanh, etc.
        # out = F.relu(out)
        out = torch.tanh(out)
        out = self.linear2(out)
        
        out =  self.act2(out) + self.constant
        
        out[:, 0] = out[:, 0] * -1
        return out    

# class augment_f(nn.Module):
#     def __init__(self, size_in):
#         super().__init__()
#         self.dim = size_in
#         # self.linear1 = nn.Linear(in_features=self.dim,out_features=4*self.dim)
#         # self.linear2 = nn.Linear(in_features=4*self.dim,out_features=8*self.dim)
#         # self.linear3 = nn.Linear(in_features=8*self.dim,out_features=self.dim*self.dim)
#
#         self.linear11 = nn.Linear(in_features=self.dim, out_features=4 * self.dim)
#         self.linear22 = nn.Linear(in_features=4* self.dim, out_features=self.dim* self.dim)
#         # self.linear3 = nn.Linear(in_features=8 * self.dim, out_features=self.dim * self.dim)
#
#         #@@@@@@@@@@ add more layers is possbile  @@@@@@@@@@
#
#     def forward(self, input_):
#         # out = self.linear1(input_)
#         # out = torch.sin(out) #######change to possible relu, tanh, etc.
#         # # out = torch.tanh(out)
#         # out = self.linear2(out)
#         # out =  torch.sin(out) #######change to possible relu, tanh, etc.
#         # # out = torch.tanh(out)
#         # out = self.linear3(out)
#         # out =  torch.sin(out) #######change to possible relu, tanh, etc.
#
#         out = self.linear11(input_)
#         out = torch.sin(out)  #######change to possible relu, tanh, etc.
#         # out = torch.tanh(out)
#         out = self.linear22(out)
#         # out = torch.tanh(out)
#         out = torch.sin(out)  #######change to possible relu, tanh, etc.
#
#
#         return out.view(-1,self.dim,self.dim)


class augment_f(nn.Module):
    def __init__(self, size_in):
        super().__init__()
        self.dim = size_in
        # self.linear1 = nn.Linear(in_features=self.dim, out_features=2 * self.dim)
        # self.linear2 = nn.Linear(in_features=2 * self.dim, out_features=2 * self.dim)
        # self.linear3 = nn.Linear(in_features=2 * self.dim, out_features=self.dim)

        self.linear11 = nn.Linear(in_features=self.dim, out_features=2 * self.dim)
        self.linear22 = nn.Linear(in_features=2 * self.dim, out_features=self.dim)

        # @@@@@@@@@@ add more layers is possbile  @@@@@@@@@@

    def forward(self, input_):
        out = self.linear11(input_)
        # out = torch.sin(out) #######change to possible relu, tanh, etc.
        out = torch.tanh(out)
        out = self.linear22(out)
        # out =  torch.sin(out) #######change to possible relu, tanh, etc.
        # out = torch.tanh(out)
        # out = self.linear3(out)
        # out =  torch.sin(out) #######change to possible relu, tanh, etc.
        # out = torch.tanh(out)

        return out


class Hamilton_V6(nn.Module):
    def __init__(self, size_in):
        super().__init__()
        self.dim = size_in
        max_ = +0.001
        min_ = -0.001
        
        self.g = myLinear(self.dim)
        
        self.augment_f = augment_f(self.dim)
        
        w2_z = torch.Tensor(self.dim)
        self.u = nn.Parameter(w2_z)
        
        ####### initial the parameters ###########
        self.u.data.uniform_(min_, max_)
        

        
    def forward(self,t, input_):
        ### input_ should be 2xdim as [x, v], where x is manifold position and v is the tangent vector
        ### If you only have v, set x as 0
        
        x = input_[...,0:self.dim]
        v = input_[...,self.dim:]
        
        g = self.g(x)
        g_inv = 1/g
        
        g_derivatie = batch_jacobian(lambda xx: self.g(xx), x, create_graph=True).view(-1,self.dim,self.dim)
        # print(g_derivatie.shape)
        
        
        connection_qij_case1 = -1*g_derivatie * g_inv[:, None, :]
        connection_qij_case2 = g_derivatie * g_inv[:,:,None]



        vtemp_list = []

        
        dx = v
        
        ################replace the following loop with block <*> ########################
        
#         dv = torch.einsum('...i,...j->...ij', v, v)
#         dv_diagonal = torch.diagonal(dv,dim1=1, dim2=2)

#         for q in range(dv.shape[1]):

#             vtemp1 =  torch.sum(dv_diagonal * connection_qij_case1[...,q],dim=1)
#             vtemp2 =  torch.sum(dv[:,q,:] * connection_qij_case2[:,q,:],dim=1)
#             vtemp3 =  torch.sum(dv[:,:,q] * connection_qij_case2[:,q,:],dim=1)
#             vtemp = vtemp1+ vtemp2 + vtemp3

#             vtemp_list.append(vtemp)

#         dv = torch.stack(vtemp_list, dim=1)
        
        ############### block <*> ########################
        
        dv_vec = torch.einsum('...i,...j->...ij', v, v)
        dv_vec_diagonal = torch.diagonal(dv_vec,dim1=1, dim2=2)
        vtemp1_vec = torch.sum(dv_vec_diagonal[: ,: ,None]*connection_qij_case1, dim=1)
        vtemp2_vec = torch.einsum('hji,hji->hj', dv_vec, connection_qij_case2)
        vtemp3_vec = torch.einsum('hij,hji->hj', dv_vec, connection_qij_case2)

        dv_vec = vtemp1_vec+vtemp2_vec+vtemp3_vec
        dv = dv_vec
        
        ####################################################
        
        
        ####compare to geognn, here is the difference###
        
        # gq = self.augment_f(x)
        #
        # dv = -dv + torch.matmul(gq, self.u) ########not sure whether we need -1*dv, can change to dv for try

        ### if this one does not work, change the augment_f to @@@@ H_4's augment_f @@@@ with the following:
        gq = self.augment_f(x)


        dv = dv + gq

        ###################################################
        
        out = torch.hstack([dx, dv])

        ######### i will add more to control curvature #######
        ######### currently, because of the metric we select, the curvature is ensured to be negative ##########
        # ####### scalar curvature #########
        # global ricci
        # Riemijkm = torch.einsum('...rjm,...iri->...jm', connection, connection) - torch.einsum('...rji,...irm->...jm', connection, connection)
        # ricci = torch.mean(torch.einsum('...ii', Riemijkm))
        ###########################
        
        
        return out
#
# dim = 16
# con = Connection(dim)
# a = torch.zeros(128,dim*2)
# b = con(a)