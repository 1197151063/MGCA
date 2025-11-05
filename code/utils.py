import world
import torch
import numpy as np
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader,WeightedRandomSampler
from dataloader import Loader
import sys
import os
import logging
from datetime import datetime
device = world.device
config = world.config
seed = world.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def maybe_num_nodes(edge_index:Tensor):
    return edge_index[0].max()+1,edge_index[1].max()+1

def Full_Sampling(dataset:Loader):
    """
    With Normlized Sampling on Graph
    """
    train_edge_index = dataset.train_edge_index.to(device)
    num_items = dataset.num_items
    weights = dataset.sampling_weights
    batch_size = config['bpr_batch_size']
    mini_batch = []
    sampler = WeightedRandomSampler(
        weights,
        num_samples=train_edge_index.size(1)
    )
    train_loader = DataLoader(
        range(train_edge_index.size(1)),
        sampler=sampler,
        batch_size=batch_size
    )
    for index in train_loader:
        pos_edge_label_index = train_edge_index[:,index]
        neg_edge_label_index = negative_sampling_bipartite(dataset,pos_edge_label_index[0])
        edge_label_index = torch.stack([
            pos_edge_label_index[0],
            pos_edge_label_index[1],
            neg_edge_label_index,
        ])
        mini_batch.append(edge_label_index)
    return mini_batch
    
    
    

def Fast_Sampling(dataset:Loader):
    """
    With Uniformal Sampling on Graph
    """
    train_edge_index = dataset.train_edge_index.to(device)
    num_items = dataset.num_items
    batch_size = config['bpr_batch_size']
    mini_batch = []
    train_loader = DataLoader(
            range(train_edge_index.size(1)),
            shuffle=True,
            batch_size=batch_size)
    for index in train_loader:
        pos_edge_label_index = train_edge_index[:,index]
        # neg_edge_label_index = negative_sampling_bipartite(dataset,pos_edge_label_index[0])
        neg_edge_label_index = torch.randint(0, num_items,(index.numel(),), device=device)
        edge_label_index = torch.stack([
            pos_edge_label_index[0],
            pos_edge_label_index[1],
            neg_edge_label_index,
        ])
        mini_batch.append(edge_label_index)
    return mini_batch
        
def Multi_Neg_Sampling(dataset:Loader,num_neg):
    train_edge_index = dataset.train_edge_index.to(device)
    num_items = dataset.num_items
    weights = dataset.sampling_weights
    batch_size = config['bpr_batch_size']
    sampler = WeightedRandomSampler(
        weights,
        num_samples=train_edge_index.size(1)
    )
    mini_batch = []
    train_loader = DataLoader(
        range(train_edge_index.size(1)),
        sampler=sampler,
        batch_size=batch_size
    )
    for index in train_loader:
        pos_edge_label_index = train_edge_index[:,index]
        neg_edge_label_index = torch.randint(0, num_items,(index.numel(),num_neg), device=device)
        edge_label_index = torch.stack([
            pos_edge_label_index[0],
            pos_edge_label_index[1],
            neg_edge_label_index,
        ])
        mini_batch.append(edge_label_index)
    return mini_batch



def InfoNCE(view1, view2, temperature: float, b_cos: bool = True):
    """
    Args:
        view1: (torch.Tensor - N x D)
        view2: (torch.Tensor - N x D)
        temperature: float
        b_cos (bool)

    Return: Average InfoNCE Loss
    """
    if b_cos:
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)

    pos_score = (view1 @ view2.T) / temperature
    score = torch.diag(F.log_softmax(pos_score, dim=1))
    return -score.mean()

def early_stopping(recall,
                   ndcg,
                   best,
                   patience,
                   model):
    if patience < 30: 
        if recall + ndcg > best: 
            patience = 0
            # print('[BEST]')
            best = recall + ndcg
            # torch.save(model.state_dict(), save_file_name)
            # torch.save(model.state_dict(),'./models/' + save_file_name)
        else:
            patience += 1
        return 0,best,patience
    else:
        return 1,best,patience # Perform Early Stopping 

def eval(node_count,topk_index,logits,ground_truth,k):
    isin_mat = ground_truth.gather(1, topk_index)
    # Calculate recall
    recall = float((isin_mat.sum(dim=-1) / node_count.clamp(1e-6)).sum())
    # Calculate NDCG
    log_positions = torch.log2(torch.arange(2, k + 2, device=logits.device).float())
    dcg = (isin_mat / log_positions).sum(dim=-1)
    ideal_dcg = torch.zeros_like(dcg)
    for i in range(len(dcg)):
        ideal_dcg = (1.0 / log_positions[:node_count[i].clamp(max=k).int()]).sum()
    ndcg = float((dcg / ideal_dcg.clamp(min=1e-6)).sum())
    return recall,ndcg

# ====================end Metrics=============================
# =========================================================

def init_logger(model_name,dataset_name,log_dir='../log'):
    global log_writer
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"{model_name}_{dataset_name}_{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.handlers = [] 
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    log_writer = logger
    return log_path


def print_log(*args, **kwargs):
    message = ' '.join(str(a) for a in args)
    if log_writer:
        log_writer.info(message)
    else:
        print(message, **kwargs)


def negative_sampling_bipartite(dataset:Loader,batch_users):
    train_edge_index = dataset.edge_index
    num_items = dataset.num_items
    device = train_edge_index.device
    users = batch_users
    batch_size = users.size(0)
    user_all = train_edge_index[0]
    item_all = train_edge_index[1]
    pos_pair_codes = (user_all * num_items + item_all).cpu().numpy()
    pos_pair_set = set(pos_pair_codes)
    neg_items = torch.randint(0, num_items, (batch_size,), device=device)
    user_ids = users
    codes = (user_ids * num_items + neg_items).cpu().numpy()
    mask = torch.from_numpy(np.isin(codes, list(pos_pair_set))).to(device)
    retry_idx = torch.where(mask)[0]
    while retry_idx.numel() > 0:
        tmp = torch.randint(0, num_items, (retry_idx.numel(),), device=device)
        neg_items[retry_idx] = tmp
        codes = (user_ids[retry_idx] * num_items + tmp).cpu().numpy()
        mask = torch.from_numpy(np.isin(codes, list(pos_pair_set))).to(device)
        retry_idx = retry_idx[mask]
    return neg_items