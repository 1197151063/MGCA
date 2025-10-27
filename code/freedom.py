import torch_sparse
from model import RecModel
from torch import LongTensor,Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch
from dataloader import Loader
import world
from procedure import train_bpr,test
import utils
import time
from world import cprint
from torch_sparse import SparseTensor
if world.config['dataset'] == 'baby':
    config = {
        'init':'uniform',#NORMAL DISTRIBUTION
        'init_weight':1,#INIT WEIGHT
        'K':2,#GCN_LAYER
        'dim':64,#EMBEDDING_SIZE
        'decay':1e-4,#L2_NORM
        'lr':1e-3,#LEARNING_RATE
        'seed':0,#RANDOM_SEED
    }

if world.config['dataset'] == 'amazon-book':
    config = {
        'init':'normal',#NORMAL DISTRIBUTION
        'init_weight':0.1,#INIT WEIGHT
        'K':3,#GCN_LAYER
        'dim':64,#EMBEDDING_SIZE
        'decay':1e-4,#L2_NORM
        'lr':1e-3,#LEARNING_RATE
        'seed':0,#RANDOM_SEED
    }

if world.config['dataset'] == 'gowalla':
    config = {
        'init':'normal',#NORMAL DISTRIBUTION
        'init_weight':0.1,#INIT WEIGHT
        'K':4,#GCN_LAYER
        'dim':64,#EMBEDDING_SIZE
        'decay':1e-4,#L2_NORM
        'lr':1e-3,#LEARNING_RATE
        'seed':0,#RANDOM_SEED
    }

if world.config['dataset'] == 'iFashion':
    config = {
        'init':'normal',#Normal DISTRIBUTION
        'init_weight':0.01,#INIT WEIGHT
        'K':3,#GCN_LAYER
        'dim':64,#EMBEDDING_SIZE
        'decay':1e-3,#L2_NORM
        'lr':1e-3,#LEARNING_RATE
        'seed':0,#RANDOM_SEED
    }

class FREEDOM(RecModel):
    def __init__(self,
                 num_users:int,
                 num_items:int,
                 edge_index:LongTensor,
                 v_feat,
                 t_feat,
                 config):
        super().__init__(
            num_users=num_users,
            num_items=num_items,
            config=config,
            edge_index=edge_index
        )
        self.user_emb = nn.Embedding(num_embeddings=num_users,
                                     embedding_dim=config['dim'])
        self.item_emb = nn.Embedding(num_embeddings=num_items,
                                     embedding_dim=config['dim'])
        self.init_weight()
        self.device = world.device
        self.K = config['K']
        edge_index = self.get_sparse_graph(edge_index=edge_index,use_value=False,value=None)
        self.edge_index = gcn_norm(edge_index)
        self.v_feat = v_feat
        self.t_feat = t_feat
        self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
        self.image_trs = nn.Linear(self.v_feat.shape[1], self.config['dim'],device=self.device)
        self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
        self.text_trs = nn.Linear(self.t_feat.shape[1], self.config['dim'],device=self.device)
        image_knn_sim,self.image_knn_idx = self.get_knn_ind(self.image_embedding.weight.detach())
        text_knn_sim,self.text_knn_idx = self.get_knn_ind(self.text_embedding.weight.detach())

        img_adj = self.build_knn_adj(self.image_knn_idx)
        txt_adj = self.build_knn_adj(self.text_knn_idx)
        img_adj_w = self.scale_sparse(img_adj, 0.1)
        txt_adj_w = self.scale_sparse(txt_adj, 0.9)
        self.mm_adj = (img_adj_w + txt_adj_w).coalesce()
        print('Go FREEDOM')
        print(f"params settings: \n emb_size:{config['dim']}\n L2 reg:{config['decay']}\n layer:{self.K}")

    def init_weight(self):
        if config['init'] == 'normal':
            nn.init.normal_(self.user_emb.weight,std=config['init_weight'])
            nn.init.normal_(self.item_emb.weight,std=config['init_weight'])
        else:
            nn.init.xavier_uniform_(self.user_emb.weight,gain=config['init_weight'])
            nn.init.xavier_uniform_(self.item_emb.weight,gain=config['init_weight'])
   
    def get_knn_sim(self,mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        return sim
    

    def get_knn_ind(self,mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        knn_sim, knn_ind = torch.topk(sim, 10 , dim=-1,sorted=False)
        return knn_sim, knn_ind
    def scale_sparse(self,adj: SparseTensor, alpha: float) -> SparseTensor:
        vals = adj.storage.value()
        if vals is None:
            adj = adj.fill_value(alpha)
        else:
            adj = adj.set_value(vals * alpha, layout='coo')
        return adj
    def build_knn_adj(
            self,
        knn_idx: torch.Tensor,                   
        add_self_loops: bool = False,  
        improved: bool = False        
    ) -> SparseTensor:
        device = knn_idx.device
        N, k = knn_idx.size()
        row = torch.arange(N, device=device).unsqueeze(1).expand(-1, k).reshape(-1)
        col = knn_idx.reshape(-1)
        adj = SparseTensor(row=row, col=col,sparse_sizes=(N, N)).coalesce()
        return gcn_norm(adj, add_self_loops=add_self_loops, improved=improved)
    
    def get_embedding(self):
        h = self.item_emb.weight
        h = self.propagate(edge_index=self.mm_adj, x=h)
        x_u=self.user_emb.weight
        x_i=self.item_emb.weight
        x = torch.cat([x_u,x_i])
        out = [x]
        for i in range(self.K):
            x = self.propagate(edge_index=self.edge_index,x=x)
            out += [x]
        out = torch.stack(out,dim=1)
        out = out.mean(dim=1)
        u_out , i_out = torch.split(out,[self.num_users,self.num_items])
        return u_out, i_out+h
    
    def bpr_loss(self,edge_label_index:Tensor,user_emb,item_emb):
        out_src = user_emb[edge_label_index[0]]
        out_dst = item_emb[edge_label_index[1]]
        out_dst_neg = item_emb[edge_label_index[2]]
        pos_score = (out_src * out_dst).sum(dim=-1)
        neg_score = (out_src * out_dst_neg).sum(dim=-1)
        loss = F.softplus(neg_score - pos_score).mean()
        return loss
    
    def l2_reg(self,edge_label_index:Tensor):
        user_emb = self.user_emb.weight
        item_emb = self.item_emb.weight
        embedding = torch.cat([user_emb[edge_label_index[0]],
                               item_emb[edge_label_index[1]],
                               item_emb[edge_label_index[2]]])
        regularization = self.config['decay'] * (1/2) * embedding.norm(p=2).pow(2)
        regularization = regularization / edge_label_index.size(1)
        return regularization
    def forward(self):
        return self.get_embedding()
    def get_loss(self,edge_label_index:Tensor):
        text_feats = self.text_trs(self.text_embedding.weight)
        image_feats = self.image_trs(self.image_embedding.weight)     
        ua_embeddings, ia_embeddings = self.get_embedding()
        batch_mf_loss = self.bpr_loss(edge_label_index=edge_label_index,
                                      user_emb=ua_embeddings,
                                      item_emb=ia_embeddings)
        mf_t_loss = self.bpr_loss(edge_label_index=edge_label_index,
                                      user_emb=ua_embeddings,
                                      item_emb=text_feats)
        mf_v_loss = self.bpr_loss(edge_label_index=edge_label_index,
                                      user_emb=ua_embeddings,
                                      item_emb=image_feats)
        return batch_mf_loss + 0.01 * (mf_t_loss + mf_v_loss)
    def item_alignment(self,items,knn_ind,knn_sim):
        knn_neighbour = knn_ind[items] # [num_items_batch * knn_k]
        user_emb = self.item_emb.weight[items].unsqueeze(1)
        item_emb = self.item_emb.weight[knn_neighbour]
        sim_score = knn_sim[items][:,knn_neighbour]
        loss = -sim_score * (user_emb * item_emb).sum(dim=-1).sigmoid().log()
        return loss.sum()
    
    def InfoNCE_U(self,view1,view2,u_idx,pos,neg,t):
        view1 = F.normalize(view1,dim=1)
        view2 = F.normalize(view2,dim=1)
        view1_pos = view1[u_idx]
        view2_pos = view2[pos]
        view2_neg = view2[neg]
        info_pos = (view1_pos * view2_pos).sum(dim=1)/ t
        info_pos_score = torch.exp(info_pos)
        info_neg = (view1_pos @ view2_neg.t())/ t
        info_neg = torch.exp(info_neg)
        info_neg = torch.sum(info_neg,dim=1,keepdim=True)
        info_neg = info_neg.T
        ssl_logits = -torch.log(info_pos_score / info_neg).mean()
        return ssl_logits

def train(datset,model:FREEDOM,opt):
    model = model
    model.train()
    S = utils.Full_Sampling(dataset=dataset)
    aver_loss = 0.
    total_batch = len(S)
    for edge_label_index in S:
        opt.zero_grad()
        loss = model.get_loss(edge_label_index)
        loss.backward()
        opt.step()   
        aver_loss += (loss)
    aver_loss /= total_batch
    return f"average loss {aver_loss:5f}"
device = world.device
dataset = Loader()
train_edge_index = dataset.train_edge_index.to(device)
test_edge_index = dataset.test_edge_index.to(device)
val_edge_index = dataset.valid_edge_index.to(device)
num_users = dataset.num_users
num_items = dataset.num_items
v_feat = dataset.v_feat
t_feat = dataset.t_feat
model = FREEDOM(num_users=num_users,
                 num_items=num_items,
                 edge_index=train_edge_index,
                 config=config,
                v_feat=v_feat,
                t_feat=t_feat).to(device)
opt = torch.optim.Adam(params=model.parameters(),lr=config['lr'])
best = 0.
patience = 0.
max_score = 0.
best_result = {}
best_model = None
for epoch in range(1, 1001):
    start_time = time.time()
    loss = train(dataset,model=model,opt=opt)
    end_time = time.time()
    recall_val, ndcg_val = test([10,20],model,train_edge_index,val_edge_index,num_users)
    recall,ndcg = test([10,20],model,train_edge_index,test_edge_index,num_users)
    if recall[20] + ndcg[20] > max_score:
        cprint('[BEST]')
        max_score = recall[20] + ndcg[20]
        best_result = {
            'recall@10': recall[10],
            'ndcg@10': ndcg[10],
            'recall@20': recall[20],
            'ndcg@20': ndcg[20]
        }
        best_model = model
    flag,best,patience = utils.early_stopping(recall_val[20],ndcg_val[20],best,patience,model)
    if flag == 1:
        break
    print(f'Epoch: {epoch:03d}, {loss}, Time: {end_time - start_time:.2f}s')
    print(f'Valid - R@10: {recall_val[10]:.4f}, N@10: {ndcg_val[10]:.4f}, R@20: {recall_val[20]:.4f}, N@20: {ndcg_val[20]:.4f}')
    print(f'Test  - R@10: {recall[10]:.4f}, N@10: {ndcg[10]:.4f}, R@20: {recall[20]:.4f}, N@20: {ndcg[20]:.4f}')
torch.save(best_model.state_dict(),f'../checkpoints/{world.config["dataset"]}_FREEDOM.pth')
print("\n========== Best ==========")
print(f'Test - R@10: {best_result["recall@10"]:.4f}, N@10: {best_result["ndcg@10"]:.4f}, '
      f'R@20: {best_result["recall@20"]:.4f}, N@20: {best_result["ndcg@20"]:.4f}')
