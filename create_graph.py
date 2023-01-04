
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold,\
    train_test_split, StratifiedShuffleSplit, GridSearchCV, RandomizedSearchCV
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch_geometric.data import InMemoryDataset, Dataset
from torch_geometric.data import Data
from sklearn.preprocessing import (StandardScaler,MinMaxScaler,Normalizer,normalize,
                                   LabelEncoder, LabelBinarizer, OneHotEncoder,PolynomialFeatures)
from torch_geometric import transforms as T
from torch_geometric.utils import to_undirected, add_self_loops
import pickle


def create_data(feat,label):
    data_list = []
    num_nodes = feat.shape[0]*7
    num_graph = feat.shape[0]
    
    for m in range(num_graph):
        edge_index = torch.tensor([
        [],
        []
            ],dtype=torch.long)
        start_node = 0*7
        edge_t = torch.tensor(
            [    
                [start_node,  start_node+1,start_node+4,start_node,  start_node+3,start_node+1,
                start_node+3,start_node+5,
                start_node+3,start_node+4,start_node+4,start_node+5,start_node+4,start_node+1],
                [start_node+1,start_node,  start_node,  start_node+4,start_node+1,start_node+3,
                start_node+5,start_node+3,
                start_node+4,start_node+3,start_node+5,start_node+4,start_node+1,start_node+4]
            ],dtype=torch.long
        )
        edge_t = edge_t[:,0::2]
        edge_v_t = torch.tensor(
            [
                [start_node,start_node+1,start_node+2,start_node+3,start_node+4,start_node+5,
                #start_node+6,start_node+6,start_node+6,start_node+6,start_node+6,start_node+6
                ],
                [start_node+6,start_node+6,start_node+6,start_node+6,start_node+6,start_node+6,
                #start_node,start_node+1,start_node+2,start_node+3,start_node+4,start_node+5
                ]
            ],dtype=torch.long
        )
        edge_self_loop = torch.tensor(
            [
                [start_node,start_node+1,start_node+2,start_node+3,start_node+4,start_node+5,start_node+6],
                [start_node,start_node+1,start_node+2,start_node+3,start_node+4,start_node+5,start_node+6]
            ],dtype=torch.long
        )
        edge_t = torch.cat((edge_t,
        edge_v_t,
        edge_self_loop
        ),1)
        #print(edge_index.shape)
        edge_index = torch.cat((edge_index,edge_t),1)
    #### to undirected
    #edge_index = to_undirected(edge_index)
    #### add self loop
    #edge_index = add_self_loops(edge_index)[0]
        #print(edge_index)
        x = torch.tensor(feat[m].reshape(7,10),dtype=torch.float32)
    #print(x.shape)
        #print(label[m].shape)
        data = Data(x=x,edge_index=edge_index,y=torch.tensor(label[m,:].reshape(1,2),dtype=torch.long))
        #print(data.has_isolated_nodes())
    #print(data.edge_index[:,:100])
        #assert not data.has_isolated_nodes()
        #assert data.num_nodes == 7
    #print(data.edge_index)
    #
        #assert data.is_undirected()
        data_list.append(data)
    #print()
    #Data
    return data_list




########  padding noise


class TrainDataset(InMemoryDataset):
    def __init__(self, root, feat,label, transform=None, pre_transform=None,):
        self.feat = feat
        self.label = label
        super().__init__(root, transform, pre_transform)
        
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['file_1', 'file_2', ]
    
    @property
    def processed_file_names(self):
        return ['data.pkl']
    
    def download(self):
        pass
    
    def process(self):
        # Read data into huge `Data` list.
        #
        #print(self.feat)
        data_list = create_data(self.feat,self.label)
        #data_list = [data]
        #print(data_list)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        #print(self.processed_paths)
        torch.save((data,slices), self.processed_paths[0])

class ValDataset(InMemoryDataset):
    def __init__(self, root, feat,label, transform=None, pre_transform=None,):
        self.feat = feat
        self.label = label
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]
    
    @property
    def processed_file_names(self):
        return ['data.pkl']

    def process(self):
        # Read data into huge `Data` list.
        #
        #print(self.feat)
        data_list = create_data(self.feat,self.label)
        #data_list = [data]
        #print(data_list)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        #print(self.processed_paths)
        torch.save((data,slices), self.processed_paths[0])

class TestDataset(InMemoryDataset):
    def __init__(self, root, feat,label, transform=None, pre_transform=None,):
        self.feat = feat
        self.label = label
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]
    
    @property
    def processed_file_names(self):
        return ['data.pkl']

    def process(self):
        # Read data into huge `Data` list.
        #
        #print(self.feat)
        data_list = create_data(self.feat,self.label)
        #data_list = [data]
        #print(data_list)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        #print(self.processed_paths)
        torch.save((data,slices), self.processed_paths[0])




### mean = 0

###CV SPLIT


