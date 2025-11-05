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
import math
from world import cprint
from torch_sparse import SparseTensor,matmul
from utils import init_logger, print_log
seed = world.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
if world.config['dataset'] == 'baby':
    config = {
        'init':world.init,#NORMAL DISTRIBUTION
        'init_weight':world.init_weight,#INIT WEIGHT
        'K':world.K,#GCN_LAYER
        'ii_k':world.ii_k,
        'dim':world.dim,#EMBEDDING_SIZE
        'decay':world.decay,#L2_NORM
        'lr':1e-3,#LEARNING_RATE
        'seed':0,#RANDOM_SEED
        'tau1':world.tau1,
        'tau2':world.tau2,
        '_lambda':world.lambda_1,
        'gamma':world.gamma,
        'alpha':world.alpha,
        'dropout':world.dropout,
        'num_neg':world.num_neg,
        'zero_layer':world.zero_layer,
    }

if world.config['dataset'] == 'sports':
    config = {
        'init':world.init,#NORMAL DISTRIBUTION
        'init_weight':world.init_weight,#INIT WEIGHT
        'K':world.K,#GCN_LAYER
        'ii_k':world.ii_k,
        'dim':world.dim,#EMBEDDING_SIZE
        'decay':world.decay,#L2_NORM
        'lr':1e-3,#LEARNING_RATE
        'seed':0,#RANDOM_SEED
        'tau1':world.tau1,
        'tau2':world.tau2,
        '_lambda':world.lambda_1,
        'gamma':world.gamma,
        'alpha':world.alpha,
        'dropout':world.dropout,
        'num_neg':world.num_neg,
        'zero_layer':world.zero_layer,
    }

if world.config['dataset'] == 'clothing':
    config = {
        'init':world.init,#NORMAL DISTRIBUTION
        'init_weight':world.init_weight,#INIT WEIGHT
        'K':world.K,#GCN_LAYER
        'ii_k':world.ii_k,
        'dim':world.dim,#EMBEDDING_SIZE
        'decay':world.decay,#L2_NORM
        'lr':1e-3,#LEARNING_RATE
        'seed':0,#RANDOM_SEED
        'tau1':world.tau1,
        'tau2':world.tau2,
        '_lambda':world.lambda_1,
        'gamma':world.gamma,
        'alpha':world.alpha,
        'dropout':world.dropout,
        'num_neg':world.num_neg,
        'zero_layer':world.zero_layer,
        's_weight':world.s_weight,
    }

log_path = init_logger(model_name='MR', dataset_name=world.config['dataset'])

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims=[128, 64], dropout=0.1, activation='relu'):
        super().__init__()
        layers = []
        last_dim = in_dim

        if activation == 'relu':
            act = nn.ReLU()
        elif activation == 'gelu':
            act = nn.GELU()
        elif activation == 'leakyrelu':
            act = nn.LeakyReLU(0.2)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        for h_dim in hidden_dims:
            layers += [
                nn.Linear(last_dim, h_dim),
                nn.BatchNorm1d(h_dim),   # 可选：去掉则更简单
                act,
                nn.Dropout(dropout)
            ]
            last_dim = h_dim

        layers.append(nn.Linear(last_dim, out_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
    
class PositionalEncoding(nn.Module):
    def __init__(self, pos_dim, max_len, device):
        """
        初始化位置编码。
        :param d_model: 嵌入的维度
        :param max_len: 最大序列长度
        """
        super(PositionalEncoding, self).__init__()
        self.pos_dim = pos_dim
        self.max_len = max_len
        self.device = device
        self.pe = torch.zeros(max_len, pos_dim, device=self.device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, pos_dim, 2).float() * (-math.log(10000.0) / pos_dim))

        self.pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
        self.pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度
        # self.pe = nn.Parameter(torch.zeros(max_len, pos_dim, device=self.device))
        # nn.init.xavier_uniform_(self.pe)


    def forward(self, x):
        """
        将位置编码添加到输入嵌入中。
        :param x: 输入嵌入，形状为 (seq_len, d_model)
        :return: 添加位置编码后的嵌入
        """
        x = x + self.pe
        return x
class FREEDOM(RecModel):
    def __init__(self,
                 num_users:int,
                 num_items:int,
                 edge_index:LongTensor,
                 v_feat,
                 t_feat,
                 value,
                 config):
        super().__init__(
            num_users=num_users,
            num_items=num_items,
            config=config,
            edge_index=edge_index
        )
        self.i_pe = PositionalEncoding(self.config['dim'], max_len=num_items, device=world.device)
        self.u_pe = PositionalEncoding(self.config['dim'], max_len=num_users, device=world.device)
        self.user_emb = nn.Embedding(num_embeddings=num_users,
                                     embedding_dim=config['dim'])
        self.item_emb = nn.Embedding(num_embeddings=num_items,
                                     embedding_dim=config['dim'])

        self.device = world.device
        self.K = config['K']
        # self.val = value
        # self.G = self.to_sparse(edge_index=edge_index,transpose=False)
        edge_index = self.get_sparse_graph(edge_index=edge_index,use_value=False,value=None)
        self.edge_index = gcn_norm(edge_index)
        self.v_feat = v_feat
        self.t_feat = t_feat
        self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
        # self.image_trs = nn.Linear(self.v_feat.shape[1], self.config['dim'],device=self.device)
        self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
        # self.text_trs = nn.Linear(self.t_feat.shape[1], self.config['dim'],device=self.device)
        # self.image_perf   = nn.Linear(self.v_feat.shape[1], self.config['dim'],device=self.device)
        # self.text_perf    = nn.Linear(self.t_feat.shape[1], self.config['dim'],device=self.device)
        self.image_mlp = MLP(in_dim=self.v_feat.shape[1], 
                             out_dim=self.config['dim'], 
                             hidden_dims=[256, 128], dropout=0.1, 
                             activation='relu')
        self.text_mlp = MLP(in_dim=self.t_feat.shape[1], 
                            out_dim=self.config['dim'], 
                            hidden_dims=[256, 128], dropout=0.1, 
                            activation='relu')
        # self.config = config
        # image_knn_sim,self.image_knn_idx = self.get_knn_ind(self.image_embedding.weight.detach())
        # text_knn_sim,self.text_knn_idx = self.get_knn_ind(self.text_embedding.weight.detach())
        # ii_graph = self.build_ppr_graph(train_edge_index, num_users , num_items, alpha=0.15, iters=20, topk=5,sparse=True)
        # self.image_knn_sim = self.get_knn_sim(self.image_embedding.weight.detach()).cuda()
        # self.text_knn_sim = self.get_knn_sim(self.text_embedding.weight.detach()).cuda()
        # img_adj = self.build_knn_adj(self.image_knn_idx)
        # txt_adj = self.build_knn_adj(self.text_knn_idx)
        # img_adj_w = self.scale_sparse(img_adj, 0.1)
        # txt_adj_w = self.scale_sparse(txt_adj, 0.9)
        
        # self.mm_adj = (img_adj_w + txt_adj_w).coalesce()
        # self.mm_adj = self.max_agreement(self.mm_adj,ii_graph).to(self.device)
        # self.mm_dense = self.mm_adj.to_dense()
        self.dropout = nn.Dropout(p=config['dropout'])
        self.init_weight()
        print('Go FREEDOM')
        print(f"params settings: \n emb_size:{config['dim']}\n L2 reg:{config['decay']}\n layer:{self.K}")
    
    def _scale_rows(self, A: SparseTensor, w_row: torch.Tensor) -> SparseTensor:
        row, col, val = A.coo()
        if val is None:
            val = torch.ones_like(row, dtype=torch.float32, device=row.device)
        val = val * w_row[row]
        return SparseTensor(row=row, col=col, value=val, sparse_sizes=A.sparse_sizes()).coalesce()
    def to_sparse(self,edge_index,transpose:False):
        if not transpose:
            r,c = edge_index
            return SparseTensor(row=r,col=c,sparse_sizes=(self.num_users,self.num_items))
        else:
            r,c = edge_index
            return SparseTensor(row=c,col=r,sparse_sizes=(self.num_items,self.num_users))
    def build_ppr_graph(self,edge_index, num_users , num_items, alpha=0.15, iters=20, topk=10,sparse=True):
        u, i = edge_index
        device = u.device
        # B: U×I
        B = SparseTensor(row=u, col=i, value=torch.ones_like(u, dtype=torch.float32, device=device),
                        sparse_sizes=(num_users, num_items)).coalesce()
        deg_u = B.sum(dim=1).to_dense().clamp_min_(1.0)     # (U,)
        deg_i = B.sum(dim=0).to_dense().clamp_min_(1.0)     # (I,)

        # P_{U->I} = D_U^{-1} B ； P_{I->U} = D_I^{-1} B^T
        P_ui = B.mul_row(1.0 / deg_u) if hasattr(B, 'mul_row') else self._scale_rows(B, 1.0 / deg_u)
        P_iu = B.t().mul_row(1.0 / deg_i) if hasattr(B.t(), 'mul_row') else self._scale_rows(B.t(), 1.0 / deg_i)

        P_ii = (P_iu @ P_ui).coalesce().set_diag(0) 

        I = num_items
        eye = torch.eye(I, device=device)
        X = (1 - alpha) * eye  
        P = P_ii

        # power iteration
        for _ in range(iters):
            X = alpha * (P @ X) + (1 - alpha) * eye  
        if sparse:
            rows = []
            cols = []
            vals = []
            X_dense = X 
            for r in range(I):
                v = X_dense[r]
                if (v > 0).sum() == 0:
                    continue
                k = min(topk, I)
                sv, si = torch.topk(v, k=k)
                rows.append(torch.full_like(si, r))
                cols.append(si)
                vals.append(sv)
            row = torch.cat(rows) if rows else torch.empty(0, dtype=torch.long, device=device)
            col = torch.cat(cols) if cols else torch.empty(0, dtype=torch.long, device=device)
            val = torch.cat(vals) if vals else torch.empty(0, dtype=X.dtype, device=device)
            return SparseTensor(row=row, col=col, value=val, sparse_sizes=(I, I)).coalesce()
        else:
            return X  

    def build_incidence(self, adj: SparseTensor):
        row, col, val = adj.coo()                      
        mask = row < col
        row, col, val = row[mask], col[mask], val[mask]
        w_sqrt = val.sqrt()
        E = row.numel()
        idx_i = torch.stack([torch.arange(E, device=row.device), row], dim=0)
        idx_j = torch.stack([torch.arange(E, device=row.device), col], dim=0)
        indices = torch.cat([idx_i, idx_j], dim=1)     # [2, 2E]
        values  = torch.cat([w_sqrt, -w_sqrt], dim=0)  # [2E]
        B = SparseTensor(row=indices[0], col=indices[1], value=values, sparse_sizes=(E, adj.size(0))).to(self.device)
        return B
    def init_weight(self):
        if config['init'] == 'normal':
            nn.init.normal_(self.user_emb.weight,std=config['init_weight'])
            nn.init.normal_(self.item_emb.weight,std=config['init_weight'])
        else:
            nn.init.xavier_uniform_(self.user_emb.weight,gain=config['init_weight'])
            nn.init.xavier_uniform_(self.item_emb.weight,gain=config['init_weight'])
    # def tv_l21(self, Z: torch.Tensor, B: torch.Tensor, eps: float = 1e-12):
    #     diff = matmul(B,Z)         
    #     edge_l2 = (diff**2).sum(dim=1).add_(eps).sqrt_()
    #     return edge_l2.sum()
    def get_knn_sim(self,mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        return sim
    
    def norm_loss(self):
        loss = 0.0
        for parameter in self.parameters():
            loss += torch.sum(parameter ** 2)
        return loss / 2
    
    def get_knn_ind(self,mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        knn_sim, knn_ind = torch.topk(sim, self.config['ii_k'] , dim=-1,sorted=False)
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
        return adj

    def max_agreement(self,mm_adj:SparseTensor,ii_adj:SparseTensor):
        print(mm_adj.storage.value().mean())
        print(ii_adj.storage.value().mean())
        D_mm  = mm_adj.to_dense()
        D_ii  = ii_adj.to_dense()
        D_agree = 0.9 * D_mm + 0.1 * D_ii
        print(D_agree.nonzero().shape)
        agree_adj = SparseTensor.from_dense(D_agree)
        print(agree_adj.nnz())
        return gcn_norm(agree_adj,add_self_loops=False)
    
    def forward(self):
        user_emb = self.user_emb.weight
        item_emb = self.item_emb.weight
        h = torch.cat([user_emb, item_emb], dim=0)
        all_emb = [h]
        for k in range(self.K):
            h = self.propagate(edge_index=self.edge_index, x = h)
            all_emb.append(h)
        all_emb = torch.stack(all_emb, dim=1)
        all_emb = torch.mean(all_emb, dim=1)
        user_emb,item_emb = all_emb[:self.num_users], all_emb[self.num_users:]
        return user_emb,item_emb

    def forward_i(self):
        item_image_emb = self.image_mlp(self.image_embedding.weight)
        item_text_emb = self.text_mlp(self.text_embedding.weight)
        alpha = self.config['alpha']
        item_emb = alpha * item_text_emb + (1 - alpha) * item_image_emb
        return matmul(self.G, item_emb)

    # def align_loss(self,edge_label_index):
    #     user_emb = self.user_emb.weight
    #     dst_emb = self.forward_i()
    #     src_emb = user_emb[edge_label_index[0]]
    #     dst_emb = dst_emb[edge_label_index[1]]
    #     return self.alignment(src_emb,dst_emb)

    def alignment(self,x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(dim=1).pow(2).mean()

    def bpr_loss(self,user_emb,item_emb, edge_label_index:Tensor):
        out_src = user_emb[edge_label_index[0]]
        out_dst = item_emb[edge_label_index[1]]
        out_dst_neg = item_emb[edge_label_index[2]]
        # out_scr = F.normalize(out_src,dim=-1)
        # out_dst = F.normalize(out_dst,dim=-1)
        # out_dst_neg = F.normalize(out_dst_neg,dim=-1)
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
    
    def get_loss(self,edge_label_index:Tensor):
        user_emb,item_emb = self.forward()
        # user_raw_emb = self.user_emb.weight
        # rec_loss = self.bpr_loss(user_emb,item_pos,edge_label_index)
        cl_loss = self.ssm_loss(user_emb,item_emb,edge_label_index)
        # align_loss = self.align_loss(edge_label_index)
        return cl_loss
    
    def item_alignment(self,items,knn_ind,knn_sim):
        knn_neighbour = knn_ind[items] # [num_items_batch * knn_k]
        user_emb = self.item_emb.weight[items].unsqueeze(1)
        item_emb = self.item_emb.weight[knn_neighbour]
        sim_score = knn_sim[items][:,knn_neighbour]
        loss = -sim_score * (user_emb * item_emb).sum(dim=-1).sigmoid().log()
        return loss.sum()
    
    def InfoNCE_U_ALL(self,view1,view2,pos_idx,t):
        view1 = F.normalize(view1,dim=1)
        view2 = F.normalize(view2,dim=1)
        view1_pos = view1[pos_idx]
        view2_pos = view2[pos_idx]
        info_pos = (view1_pos * view2_pos).sum(dim=1)/ t
        info_pos_score = torch.exp(info_pos)
        info_neg = (view1_pos @ view2.t())/ t
        info_neg = torch.exp(info_neg)
        info_neg = torch.sum(info_neg,dim=1,keepdim=True)
        info_neg = info_neg.T
        ssl_logits = -torch.log(info_pos_score / info_neg).mean()
        return ssl_logits
    def link_prediction(self,
                        src_index:Tensor=None,
                        dst_index:Tensor=None):
        out_u, out_i = self.forward()

        if src_index is None:
            src_index = torch.arange(self.num_users).long()
        if dst_index is None:
            dst_index = torch.arange(self.num_items).long()
        out_src = out_u[src_index]
        out_dst = out_i[dst_index]
        # out_src = F.normalize(out_src,dim=-1)
        # out_dst = F.normalize(out_dst,dim=-1)
        pred = out_src @ out_dst.t()
        return pred
    def ssm_loss(self,emb1,emb2,edge_label_index):
        neg_edge_index = torch.randint(0, self.num_items,(edge_label_index[1].numel(),self.config['num_neg']), device=emb1.device)
        emb_neg = emb2[neg_edge_index]
        emb1 = emb1[edge_label_index[0]]
        emb2 = emb2[edge_label_index[1]]
        # emb1 = self.dropout(emb1)
        emb1 = F.normalize(emb1, dim=-1)
        item_emb = torch.cat([emb2.unsqueeze(1), emb_neg], dim=1)
        item_emb = F.normalize(item_emb, dim=-1)
        y_pred = torch.bmm(item_emb, emb1.unsqueeze(-1)).squeeze(-1)
        pos_logits = torch.exp(y_pred[:, 0] / self.config['tau2']) 
        neg_logits = torch.exp(y_pred[:, 1:]/ self.config['tau2']) 
        Ng = neg_logits.sum(dim=-1)
        loss = (- torch.log(pos_logits / Ng))
        return loss.mean() 


    
def train(datset,model:FREEDOM,opt):
    model = model
    model.train()
    neg_sampling = world.neg_sampling
    if neg_sampling == 'uniform':
        S = utils.Fast_Sampling(dataset=dataset)
    else:
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
print(test_edge_index.shape)
val_edge_index = dataset.valid_edge_index.to(device)
vals = dataset.train_value.to(device)
num_users = dataset.num_users
num_items = dataset.num_items
v_feat = dataset.v_feat
t_feat = dataset.t_feat
model = FREEDOM(num_users=num_users,
                 num_items=num_items,
                 edge_index=train_edge_index,
                 config=config,
                 value=vals,
                v_feat=v_feat,
                t_feat=t_feat).to(device)
opt = torch.optim.AdamW(params=model.parameters(),lr=config['lr'])
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
    if recall[20] + ndcg[20] + recall[10] + ndcg[10] > max_score:
        print_log('[BEST]')
        best_model = model.state_dict()
        max_score = recall[20] + ndcg[20] + recall[10] + ndcg[10]
        best_result = {
            'recall@10': recall[10],
            'ndcg@10': ndcg[10],
            'recall@20': recall[20],
            'ndcg@20': ndcg[20]
        }
    flag,best,patience = utils.early_stopping(recall_val[20]+recall_val[10],ndcg_val[20]+ndcg_val[10],best,patience,model)
    if flag == 1:
        break
    
    print_log(f'Epoch: {epoch:03d}, {loss}, Time: {end_time - start_time:.2f}s')
    print_log(f'Valid - R@10: {recall_val[10]:.4f}, N@10: {ndcg_val[10]:.4f}, R@20: {recall_val[20]:.4f}, N@20: {ndcg_val[20]:.4f}')
    print_log(f'Test  - R@10: {recall[10]:.4f}, N@10: {ndcg[10]:.4f}, R@20: {recall[20]:.4f}, N@20: {ndcg[20]:.4f}')
print_log("\n========== Best ==========")
torch.save(best_model, f'{world.config["dataset"]}_LGCN.best.pth')
print_log(f'Test - R@10: {best_result["recall@10"]:.4f}, N@10: {best_result["ndcg@10"]:.4f}, '
      f'R@20: {best_result["recall@20"]:.4f}, N@20: {best_result["ndcg@20"]:.4f}')
print_log("=========================\n")
print_log(f'Config: {config}')