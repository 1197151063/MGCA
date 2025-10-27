from torch.utils.data import Dataset
import pandas as pd
from world import cprint
import world
import torch
import numpy as np
from torch_sparse import SparseTensor
from torch_geometric.utils import degree
seed = world.seed
    
class Loader(Dataset):
    """
    Loading data from datasets
    supporting:['baby','sports']
    """
    def __init__(self,config=world.config,path='../data/'):
        dir_path = path + config['dataset']
        cprint(f'loading from {dir_path}')
        self.n_user = 0
        self.n_item = 0
        inter_file = dir_path + '/' + config['dataset'] + '.inter'
        image_file = dir_path + '/image_feat.npy'
        text_file = dir_path + '/text_feat.npy'
        df = pd.read_csv(inter_file, sep='\t', header=None, 
                         names=['user_id', 'item_id', 'rating', 'timestamp', 'x_label'],
                        low_memory=False)
        for col in ['user_id', 'item_id', 'rating', 'timestamp', 'x_label']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()
        df['user_id'] = df['user_id'].astype(int)
        df['item_id'] = df['item_id'].astype(int)
        df['rating'] = df['rating'].astype(float)
        df['timestamp'] = df['timestamp'].astype(int)
        df['x_label'] = df['x_label'].astype(int)

        image_feat = np.load(image_file)
        text_feat = np.load(text_file)
        self.v_feat = torch.from_numpy(image_feat).float().to(world.device)
        self.t_feat = torch.from_numpy(text_feat).float().to(world.device)
        self.n_user = df['user_id'].max() + 1
        self.n_item = df['item_id'].max() + 1
        train_df = df[df['x_label'] == 0]
        valid_df = df[df['x_label'] == 1]
        test_df  = df[df['x_label'] == 2]
        # train_df = train_df[train_df['rating'] > 1]
        # valid_df = valid_df[valid_df['rating'] > 1]
        train_users = set(train_df['user_id'].values)
        valid_df = valid_df[valid_df['user_id'].isin(train_users)]
        test_df = test_df[test_df['user_id'].isin(train_users)]
        test_ratings = test_df['rating']
        # test_df = test_df[test_ratings > 2]
        # valid_users = user_item_counts[user_item_counts >= 3].index
        # test_df = test_df[test_df['user_id'].isin(valid_users)]
        train_edge_index = torch.LongTensor(np.array(train_df[['user_id', 'item_id']]).T)
        valid_edge_index = torch.LongTensor(np.array(valid_df[['user_id', 'item_id']]).T)
        test_edge_index = torch.LongTensor(np.array(test_df[['user_id', 'item_id']]).T)
        self.train_edge_index = train_edge_index
        self.valid_edge_index = valid_edge_index
        self.test_edge_index = test_edge_index
        self.train_value = torch.tensor(np.array(train_df['rating']).T)
        self.sampling_weights = self.get_edge_weights(train_edge_index)
        self.edge_index = torch.cat([self.train_edge_index,self.valid_edge_index,self.test_edge_index],dim=1).to(world.device)
        print(self.train_edge_index.shape[1])
        print(f"num_users:{self.n_user}, num_items:{self.n_item}")
        print(f"{world.dataset} is ready to go")

    @property
    def num_users(self):
        return self.n_user
    @property
    def num_items(self):
        return self.n_item

    

    '''
    A = |0   R|
        |R^T 0|
    R : user-item bipartite graph
    '''
    def getSparseGraph(self,edge_weights):
        cprint("generate Adjacency Matrix A")
        user_index = self.train_edge_index[0]
        item_index = self.train_edge_index[1]
        row_index = torch.cat([user_index,item_index+self.n_user])
        col_index = torch.cat([item_index+self.n_user,user_index])
        value = torch.cat([edge_weights,edge_weights])
        return SparseTensor(row=row_index,col=col_index,value=value,sparse_sizes=(self.n_item+self.n_user,self.n_item+self.n_user))

    def getSparseBipartite(self):
        user_index = self.train_edge_index[0]
        item_index = self.train_edge_index[1]
        return SparseTensor(row=user_index,col=item_index,sparse_sizes=(self.num_users,self.num_items))
    
    def get_edge_weights(self,edge_index):
        user_degree = degree(edge_index[0])
        sampling_weight = 1 / user_degree[edge_index[0]]
        return sampling_weight