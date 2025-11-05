# from model import FREEDOM
# from scipy.stats import gaussian_kde
# import utils
# import torch
# import world
# import time
# from torch import Tensor
# from procedure import train_bpr
# from dataloader import Loader
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.manifold import TSNE
# from mpl_toolkits.mplot3d import Axes3D
# from shapely.geometry import Point, MultiPoint
# from shapely.ops import unary_union
# from scipy.ndimage import gaussian_filter
# from descartes import PolygonPatch
# from sklearn.decomposition import PCA
# import torch_sparse
# from model import RecModel
# from torch import LongTensor,Tensor
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn.conv.gcn_conv import gcn_norm
# import torch
# from dataloader import Loader
# import world
# from procedure import train_bpr,test
# import utils
# import time
# import math
# from world import cprint
# from torch_sparse import SparseTensor,matmul
# from utils import init_logger, print_log
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
import numpy as np
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
class LGCN(RecModel):
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
        # self.item_emb = nn.Embedding(num_embeddings=num_items,
        #                              embedding_dim=config['dim'])

        self.device = world.device
        self.K = config['K']
        # self.val = value
        # self.G = self.to_sparse(edge_index=edge_index,transpose=False)
        # edge_index = self.get_sparse_graph(edge_index=edge_index,use_value=False,value=self.val)
        # self.edge_index = gcn_norm(edge_index)
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
            # nn.init.normal_(self.item_emb.weight,std=config['init_weight'])
        else:
            nn.init.xavier_uniform_(self.user_emb.weight,gain=config['init_weight'])
            # nn.init.xavier_uniform_(self.item_emb.weight,gain=config['init_weight'])
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
        item_image_emb = self.image_mlp(self.image_embedding.weight)
        item_text_emb = self.text_mlp(self.text_embedding.weight)
        i_v_pos = self.i_pe(item_image_emb)
        i_t_pos = self.i_pe(item_text_emb)
        # u_pos = user_emb
        # print(u_pos.shape)
        u_pos = self.u_pe(user_emb)
        # print(u_pos.shape)
        # print(u_pos.shape,i_v_pos.shape,i_t_pos.shape)
        alpha = self.config['alpha']
        item_emb = alpha * item_text_emb + (1 - alpha) * item_image_emb
        item_pos_emb = alpha * i_t_pos + (1 - alpha) * i_v_pos
        return u_pos,item_emb,item_pos_emb
    
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
        user_emb,item_emb,item_pos = self.forward()
        # user_raw_emb = self.user_emb.weight
        # rec_loss = self.bpr_loss(user_emb,item_pos,edge_label_index)
        cl_loss = self.ssm_loss(user_emb,item_pos,edge_label_index)
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
        out_u, out_i,_ = self.forward()

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
def _topk_indices_per_row(M: np.ndarray, k: int):
    """返回每行的 top-k 列索引（按数值从大到小），忽略对角元素。"""
    n = M.shape[0]
    assert M.shape[0] == M.shape[1], "M must be square"
    # 复制一份，置对角为 -inf 以排除自身
    M_ = M.copy()
    np.fill_diagonal(M_, -np.inf)

    k_eff = min(k, n - 1)  # 防止 k >= n
    # 用 argpartition 取前 k 个（无序），再按值排序
    # idx_topk: (n, k_eff)
    idx_part = np.argpartition(M_, -k_eff, axis=1)[:, -k_eff:]
    # 对每行将这 k 个按值降序排
    row_indices = np.arange(n)[:, None]
    vals = M_[row_indices, idx_part]
    order = np.argsort(-vals, axis=1)
    idx_topk = idx_part[row_indices, order]
    return idx_topk  # shape (n, k_eff)

best = 0.
patience = 0.
max_score = 0.
best_result = {}
best_model = None
lgcn = LGCN(num_users=num_users,
                 num_items=num_items,
                 edge_index=train_edge_index,
                 config=config,
                 value=vals,
                v_feat=v_feat,
                t_feat=t_feat).to(device)
lgcn_opt = torch.optim.AdamW(params=lgcn.parameters(),lr=config['lr'])

state_dict = torch.load('baby_best.pth', map_location=device)
lgcn_state_dict = torch.load('baby_LGCN.best.pth', map_location=device)
model.load_state_dict(state_dict)
lgcn.load_state_dict(lgcn_state_dict)
user_emb,item_emb = lgcn.forward()
u_pos,modal_item,item_pos_emb = model.forward()
pos_for_item = item_pos_emb 
item_sim = pos_for_item @ pos_for_item.t()
lgcn_item_sim = item_emb @ item_emb.t()
# R = SparseTensor.from_edge_index(
#         train_edge_index, sparse_sizes=(num_users, num_items)
#     ).to_dense().to(device)                  # [U, I] 0/1
# G = (R.t() @ R).float()                  # [I, I] 共现次数
# G.fill_diagonal_(0.0)
# pos_mask = G > 0
# S = pos_for_item @ pos_for_item.t()
# S = S * (G > 0)
# print(S.shape)
def compute_nearest_neighbors(feats, topk=1):
    """
    Compute the nearest neighbors of feats
    Args:
        feats: a torch tensor of shape N x D
        topk: the number of nearest neighbors to return
    Returns:
        knn: a torch tensor of shape N x topk
    """
    assert feats.ndim == 2, f"Expected feats to be 2D, got {feats.ndim}"
    knn = (
        (feats @ feats.T).fill_diagonal_(-1e8).argsort(dim=1, descending=True)[:, :topk]
    )
    return knn

def mutual_knn(feats_A, feats_B, topk):
        """
        Computes the mutual KNN accuracy.

        Args:
            feats_A: A torch tensor of shape N x feat_dim
            feats_B: A torch tensor of shape N x feat_dim

        Returns:
            A float representing the mutual KNN accuracy
        """
        knn_A = compute_nearest_neighbors(feats_A, topk)
        knn_B = compute_nearest_neighbors(feats_B, topk)   

        n = knn_A.shape[0]
        topk = knn_A.shape[1]

        # Create a range tensor for indexing
        range_tensor = torch.arange(n, device=knn_A.device).unsqueeze(1)

        # Create binary masks for knn_A and knn_B
        lvm_mask = torch.zeros(n, n, device=knn_A.device)
        llm_mask = torch.zeros(n, n, device=knn_A.device)

        lvm_mask[range_tensor, knn_A] = 1.0
        llm_mask[range_tensor, knn_B] = 1.0
        
        acc = (lvm_mask * llm_mask).sum(dim=1) / topk
        
        return acc.mean().item()


for k in range(10,105,5):
    acc = mutual_knn(item_sim,lgcn_item_sim, topk=k)
    print(f"Top-{k} Mutual KNN Accuracy: {acc:.4f}")
  

# target_dim = sample_item_emb.size(1)
# sample_user_emb = pca_to(sample_user_emb, target_dim)
# sample_item_emb = pca_to(sample_item_emb, target_dim)
# sample_v_feat   = pca_to(sample_v_feat,   target_dim)
# sample_t_feat   = pca_to(sample_t_feat,   target_dim)

# # 移到 CPU 再做 sklearn 的降维
# sample_user_emb = sample_user_emb.detach().cpu()
# sample_item_emb = sample_item_emb.detach().cpu()
# sample_v_feat   = sample_v_feat.detach().cpu()
# sample_t_feat   = sample_t_feat.detach().cpu()
# X = torch.cat([sample_user_emb, sample_item_emb, sample_v_feat, sample_t_feat], dim=0).numpy()
# labels = (['user_emb'] * num_user_samples +
#           ['item_emb'] * num_item_samples +
#           ['v_feat']   * num_item_samples +
#           ['t_feat']   * num_item_samples)

# # 依据总点数自动设定合适的 perplexity
# n_points = X.shape[0]
# perp = min(30, max(5, n_points // 20))  # 经验：点数/20，夹在[5,30]

# use_tsne = True  # 想用 PCA-3D 则设为 False
# if use_tsne:
#     reducer = TSNE(n_components=3, perplexity=perp, n_iter=1500,
#                    init="pca", learning_rate="auto", random_state=42)
#     X_3d = reducer.fit_transform(X)
#     title = f"3D t-SNE of FREEDOM embeddings (perplexity={perp})"
# else:
#     reducer = PCA(n_components=3, random_state=42)
#     X_3d = reducer.fit_transform(X)
#     title = "3D PCA of FREEDOM embeddings"

# # ====== 画三维散点并保存 ======
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')

# color_map = {
#     'user_emb': 'tab:blue',
#     'item_emb': 'tab:orange',
#     'v_feat'  : 'tab:green',
#     't_feat'  : 'tab:red',
# }

# labels_np = np.array(labels)
# for k, c in color_map.items():
#     mask = (labels_np == k)
#     ax.scatter(X_3d[mask, 0], X_3d[mask, 1], X_3d[mask, 2],
#                s=10, alpha=0.75, label=k, color=c)

# ax.set_title(title)
# ax.set_xlabel("dim-1"); ax.set_ylabel("dim-2"); ax.set_zlabel("dim-3")
# ax.grid(False)
# ax.legend(loc="best")
# plt.tight_layout()
# plt.savefig('FREEDOM_embeddings_3d.png', dpi=300)
# plt.close()
# print("Saved to FREEDOM_embeddings_3d.png")

# # t-SNE降维
# tsne = TSNE(n_components=2, perplexity=30, n_iter=1500, random_state=42)
# X_2d = tsne.fit_transform(X)

# # 可视化
# plt.figure(figsize=(8, 6))
# colors = {
#     'user_emb': 'blue',
#     # 'item_emb': 'orange',
#     'v_feat': 'green',
#     't_feat': 'red'
# }

# for label in colors.keys():
#     mask = np.array(labels) == label
#     plt.scatter(X_2d[mask, 0], X_2d[mask, 1],
#                 label=label, alpha=0.7, s=20, color=colors[label])

# plt.title("t-SNE Visualization of FREEDOM Embeddings")
# plt.legend(
#     loc='upper left',          # 位置在图外右边            # 去掉边框
#     fontsize=12,                # 放大字体
#     labelspacing=0.8,           # 标签间距
#     handlelength=1.8,           # 颜色标识长度
#     handletextpad=0.8           # 颜色与文字间距
# )


# # plt.xlabel("t-SNE dim 1")
# # plt.ylabel("t-SNE dim 2")
# plt.grid(False)
# plt.savefig(f'FREEDOM_embeddings.png', dpi=300)

# tsne = TSNE(n_components=2, perplexity=30, n_iter=1500, random_state=42)
# X_2d = tsne.fit_transform(X)
# from matplotlib.patches import Patch

# # 假设你已有 X_2d (n,2) 和 labels (n,)
# X2 = np.asarray(X_2d)
# labs = np.asarray(labels)

# # 类别与颜色（按你的语义来）
# classes = ['user_emb', 'v_feat', 't_feat']  # 如有其他类，补上
# colors  = {'user_emb':'tab:blue', 'v_feat':'tab:green', 't_feat':'tab:red'}

# # 1) 设定绘制网格（带些边距）
# xmin, xmax = X2[:,0].min(), X2[:,0].max()
# ymin, ymax = X2[:,1].min(), X2[:,1].max()
# padx = 0.05 * (xmax - xmin + 1e-9)
# pady = 0.05 * (ymax - ymin + 1e-9)
# xmin, xmax = xmin - padx, xmax + padx
# ymin, ymax = ymin - pady, ymax + pady

# H = W = 500  # 分辨率（越大越细腻）
# xx, yy = np.meshgrid(np.linspace(xmin, xmax, W), np.linspace(ymin, ymax, H))
# grid = np.vstack([xx.ravel(), yy.ravel()])  # (2, H*W)

# # 2) 每类做 KDE 得到密度图
# dens_list = []
# valid_classes = []
# bw = 0.2  # 带宽（0.1~0.3 可调；越小边界越锋利）
# for cls in classes:
#     pts = X2[labs == cls]
#     if len(pts) < 3:
#         continue  # 样本太少跳过
#     kde = gaussian_kde(pts.T, bw_method=bw)
#     dens = kde(grid).reshape(H, W)
#     dens_list.append(dens)
#     valid_classes.append(cls)

# if len(dens_list) == 0:
#     raise RuntimeError("没有足够的点绘制KDE语义色块。")

# dens_stack = np.stack(dens_list, axis=0)  # (C,H,W)

# # 3) 生成语义分区：每个像素取密度最大的类别
# dom_idx = np.argmax(dens_stack, axis=0)               # (H,W) -> 类别索引
# tot_den = np.clip(np.sum(dens_stack, axis=0), 1e-12, None)  # 总密度，用于做背景遮罩

# # 4) 可选：平滑一下，让边界更柔和
# dom_idx_smooth = dom_idx.copy()
# # 用总密度做背景阈值（低密度区域不画，避免“漂移的色块”）
# thr = np.percentile(tot_den, 40)  # 40分位阈值可调（30~60）
# mask = tot_den >= thr

# # 5) 把类别索引映射为颜色
# rgb = np.zeros((H, W, 3), dtype=float)
# for i, cls in enumerate(valid_classes):
#     rgb[dom_idx_smooth == i] = plt.get_cmap('tab10')(i)[:3]

# # 6) 叠加连续“强度”作为透明度（密度越高越不透明）
# alpha = (tot_den - tot_den[mask].min()) / (tot_den[mask].ptp() + 1e-9)
# alpha = np.clip(alpha, 0, 1)
# alpha = gaussian_filter(alpha, sigma=1.0)  # 进一步柔化透明度
# alpha[~mask] = 0.0

# # 7) 绘制
# fig, ax = plt.subplots(figsize=(7, 6))
# ax.imshow(
#     np.transpose(rgb, (0,1,2)), origin='lower',
#     extent=[xmin, xmax, ymin, ymax], alpha=None  # 颜色本身不透明
# )
# # 再叠一层灰色密度热影，增强“连续感”（可去掉）
# ax.imshow(
#     alpha.T, origin='lower', extent=[xmin, xmax, ymin, ymax],
#     cmap='Greys', alpha=0.35  # 轻微影子
# )

# # 可选：叠少量散点作锚点（让读者知道语义来源；不需要可注释）
# for cls in valid_classes:
#     m = (labs == cls)
#     P = X2[m]
#     ax.scatter(P[:,0], P[:,1], s=6, alpha=0.25, color=colors.get(cls, 'k'), edgecolors='none')

# # 轴样式
# ax.set_xticks([]); ax.set_yticks([])
# ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
# ax.set_title("Embedding Semantic Field (KDE-based patches)", fontsize=14)

# # 图例（用色块代理）
# handles = [Patch(facecolor=colors.get(c, plt.get_cmap('tab10')(i)[:3]), edgecolor='none', label=c)
#            for i, c in enumerate(valid_classes)]
# ax.legend(handles=handles, frameon=False, loc='best')

# plt.tight_layout()
# plt.savefig("embedding_semantic_patches.png", dpi=400)