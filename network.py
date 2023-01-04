import torch
import torch.nn as nn


class BatchNorm(nn.Module):
    def __init__(self,num_features):
        super(BatchNorm,self).__init__()
        self.bn=nn.BatchNorm1d(num_features=num_features)

    def forward(self,x):
        x_out=torch.transpose(x,1,2)
        x_out=self.bn(x_out)
        x_out=torch.transpose(x,1,2)
        return x_out


class Locally_feature_extraction(nn.Module):
    def __init__(self,stride,vector_shape,output_shape,normtype='BN',has_activation=True):
        super(Locally_feature_extraction,self).__init__()
        #self.scale=scale
        self.stride=stride             #stride: num_kernels/num_linears
        self.vector_shape=vector_shape
        #self.layers=layers
        self.normtype=normtype
        self.input_shape=self.vector_shape//self.stride
        self.output_shape=output_shape
        self.has_activation=has_activation

        if self.vector_shape%self.stride==0:
            self.division_exact=True
        else:
            self.division_exact=False
        self.local_layers=nn.ModuleList([nn.Linear(self.input_shape,self.output_shape)
            for i in range(self.stride)])
        if self.normtype=='BN':
            self.Norm=BatchNorm(num_features=self.output_shape)
        elif self.normtype=='LN':
            self.Norm=nn.LayerNorm((self.stride,self.output_shape))
        self.activation=nn.ReLU()

    def forward(self,x):
        if self.division_exact==True:
            x=x.reshape(x.shape[0],self.stride,self.input_shape)
        else:
            x=x[:,:self.stride*self.input_shape]
            x=x.reshape(x.shape[0],self.stride,self.input_shape)
        x_out=[]
        for i,l in enumerate(self.local_layers):
            x_temp=l(x[:,i:i+1,:])
            x_out.append(x_temp)
        x_out=torch.cat(x_out,1)
        x_out=self.Norm(x_out)
        if self.has_activation:
            x_out=self.activation(x_out)
        x_out=x_out.reshape(x_out.shape[0],-1)
        return x_out

class Multi_layer_locally_feature_extraction(nn.Module):
    def __init__(self,stride,vector_shape_list,output_shape_list,num_layers=4
                 ):
        super(Multi_layer_locally_feature_extraction,self).__init__()
        self.num_layers=num_layers
        self.stride=stride
        self.output_shape_list=output_shape_list
        self.vector_shape_list=vector_shape_list

        self.layers=nn.ModuleList(
                Locally_feature_extraction(self.stride,vector_shape=self.vector_shape_list[i],
                output_shape=self.output_shape_list[i])
                for i in range(num_layers)
            )
            
    def forward(self,x):
        for i,l in enumerate(self.layers):
            x=l(x)
        return x

class Nonlinear_feature_interaction(nn.Module):
    def __init__(self,input_shape,linear_out):
        super(Nonlinear_feature_interaction,self).__init__()
        self.sig=nn.Sigmoid()
        self.tanh=nn.Tanh()
        self.input_shape=input_shape
        self.linear_out=linear_out
        self.linear_layers=nn.ModuleList([nn.Linear(self.input_shape,self.linear_out) for i in range(4)])
        self.norm=nn.BatchNorm1d(self.linear_out)

    def forward(self,x):
        list_nl=[]
        for i,l in enumerate(self.linear_layers):
            x_nl=l(x)
            list_nl.append(x_nl)

        nl1=list_nl[0]
        nl2=list_nl[1]
        nl3=list_nl[2]
        nl4=list_nl[3]
        nlfs1=nl1*nl2
        nlfs1=self.norm(nlfs1)
        nl3=self.sig(nl3)
        nl4=self.tanh(nl4)
        nlfs2=nl3*nl4
        x=nlfs1+nlfs2
        return x