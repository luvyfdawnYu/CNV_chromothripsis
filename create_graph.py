import torch
from torch_geometric.data import InMemoryDataset, Dataset
from torch_geometric.data import Data




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
                
                ],
                [start_node+6,start_node+6,start_node+6,start_node+6,start_node+6,start_node+6,
                
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
       
        edge_index = torch.cat((edge_index,edge_t),1)
        x = torch.tensor(feat[m].reshape(7,10),dtype=torch.float32)
        data = Data(x=x,edge_index=edge_index,y=torch.tensor(label[m,:].reshape(1,2),dtype=torch.long))
        data_list.append(data)
    return data_list


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
        data_list = create_data(self.feat,self.label)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
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
        data_list = create_data(self.feat,self.label)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
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
        data_list = create_data(self.feat,self.label)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data,slices), self.processed_paths[0])



