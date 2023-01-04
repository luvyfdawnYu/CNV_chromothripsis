import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv
from network import Multi_layer_locally_feature_extraction, Nonlinear_feature_interaction


class GECNVNet(nn.Module):
    def __init__(self,
                GE_dim,
                LF_stride,
                LF_input_shape,
                LF_output_shape,
                NL_input_shape,
                NL_output_shape,
                dropout_rate):
        super(GECNVNet,self).__init__()
        self.gnn1 = TransformerConv(10,GE_dim[0])
        self.gnn2 = TransformerConv(GE_dim[0],GE_dim[1])
        self.locally = Multi_layer_locally_feature_extraction(
                                                                stride=LF_stride,
                                                                vector_shape_list=LF_input_shape,
                                                                output_shape_list=LF_output_shape,
                                                            )
        self.nl = Nonlinear_feature_interaction(input_shape = NL_input_shape,linear_out = NL_output_shape)
        self.bn1 = nn.BatchNorm1d(GE_dim[0])
        self.bn2 = nn.BatchNorm1d(GE_dim[1])
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.activation2 = nn.ReLU()
        self.activation3 = nn.ReLU()
        self.classify = nn.Linear(NL_output_shape,2,bias=False)
        
    def forward(self,data):
        x = data.x
        edge_index = data.edge_index
        x = self.gnn1(x,edge_index)
        x = self.bn1(x)
        x = self.gnn2(x,edge_index)
        x = self.bn2(x)
        x = self.activation2(x)
        x = self.dropout2(x)
        x = x.reshape(x.shape[0]//7,-1,7)
        x = torch.bmm(x, x.permute(0, 2, 1)) ##1024
        x = x.view(-1, x.shape[-1] ** 2) 
        x = self.locally(x)
        x = self.nl(x)
        x = self.dropout3(x)
        x = self.classify(x)
        return x