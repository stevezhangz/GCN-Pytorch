import torch
from torch import nn


class Graph_layer(nn.Module):
    def __init__(self,infea,outfea,bias,use_degree):
        super().__init__()
        self.weight=nn.Parameter(torch.rand(size=(infea,outfea)))
        if bias !=None:
            self.bias=nn.Parameter(torch.rand(size=(bias,)))
        else:
            self.register_buffer('bias',None)
        self.use_degree=use_degree
        self.weight_initialization()
    def forward(self,adj,x,degree):
        # DADHW
        if self.use_degree:
            gcn_out= torch.mm(torch.mm(torch.mm(torch.mm(degree**0.5,adj),adj),x),self.weight)
        else:
            gcn_out= torch.mm(torch.mm(adj,x),self.weight)
        if self.bias:
            return gcn_out+self.bias
        else:
            return gcn_out
    def weight_initialization(self):
        pass

class Graph_layer_withD(nn.Module):
    def __init__(self,infea,outfea,bias):
        super().__init__()
        self.weight=nn.Parameter(torch.rand(size=(infea,outfea)))
        if bias !=None:
            self.bias=nn.Parameter(torch.rand(size=(bias,)))
        else:
            self.register_buffer('bias',None)
        self.weight_initialization()
    def forward(self,adj,x):
        # DADHW
        mid=torch.mm(adj,x)
        gcn_out= torch.mm(mid,self.weight)

        if self.bias:
            return gcn_out+self.bias
        else:
            return gcn_out
    def weight_initialization(self):
        pass

class GCN_without_D(nn.Module):
    def __init__(self,infea,hidden,outfea,act,use_degree,drop=0.5):
        super().__init__()
        assert act in ['relu','sigmoid']
        self.act={'relu':nn.ReLU(),'sigmoid':nn.Sigmoid()}[act]
        self.gcn_layer1=Graph_layer(infea,hidden,bias=None,use_degree=use_degree)
        self.gcn_layer2=Graph_layer(hidden,outfea,bias=None,use_degree=use_degree)
        self.drop=nn.Dropout(drop)
        self.use_degree=use_degree
    def forward(self,adj,x,degree):
        gcn_output1=self.act(self.gcn_layer1(adj,x,degree))
        drop_fea=self.drop(gcn_output1)
        gcn_output2=self.gcn_layer2(adj,drop_fea,degree)
        return gcn_output2

class GCN_with_D(nn.Module):
    def __init__(self,infea,hidden,outfea,act,drop=0.5):
        super().__init__()
        assert act in ['relu','sigmoid']
        self.act={'relu':nn.ReLU(),'sigmoid':nn.Sigmoid()}[act]
        self.gcn_layer1=Graph_layer_withD(infea,hidden,bias=None)
        self.gcn_layer2=Graph_layer_withD(hidden,outfea,bias=None)
        self.drop=nn.Dropout(drop)
    def forward(self,adj,x):

        gcn_output1=self.act(self.gcn_layer1(adj,x))
        drop_fea=self.drop(gcn_output1)
        gcn_output2=self.gcn_layer2(adj,drop_fea)
        return gcn_output2


