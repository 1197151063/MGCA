from torch import nn,Tensor,LongTensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import SparseTensor
import torch.nn.functional as F
from torch_sparse import SparseTensor,matmul
from torch_geometric.utils import negative_sampling
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch
import world
device = world.device
seed = world.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
"""
Already supported:[LightGCN, SGL, SimGCL, NCL, DirectAU]
"""

class RecModel(MessagePassing):
    def __init__(self,
                 num_users:int,
                 num_items:int,
                 config,
                 edge_index:LongTensor):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_nodes = num_users + num_items
        self.config = config
        self.f = nn.Sigmoid()

    def get_sparse_heter(self,edge_index,val):
        num_nodes = edge_index[0].max()+1
        return SparseTensor(row=edge_index[0],col=edge_index[1],
                            value=val,sparse_sizes=(num_nodes,num_nodes))

    def get_sparse_graph(self,
                         edge_index,
                         use_value=False,
                         value=None):
        num_users = self.num_users
        num_nodes = self.num_nodes
        r,c = edge_index
        row = torch.cat([r , c + num_users])
        col = torch.cat([c + num_users , r])
        if use_value:
            value = torch.cat([value,value])
            return SparseTensor(row=row,col=col,value=value,sparse_sizes=(num_nodes,num_nodes))
        else:
            return SparseTensor(row=row,col=col,sparse_sizes=(num_nodes,num_nodes))
    
    def get_embedding(self):
        raise NotImplementedError
    
    def forward(self,
                edge_label_index:Tensor):
        out = self.get_embedding()
        out_u,out_i = torch.split(out,[self.num_users,self.num_items])
        out_src = out_u[edge_label_index[0]]
        out_dst = out_i[edge_label_index[1]]
        out_dst_neg = out_i[edge_label_index[2]]
        return (out_src * out_dst).sum(dim=-1),(out_src * out_dst_neg).sum(dim=-1)
    
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
    
    # BPR+L2
    def recommendation_loss(self,
                            pos_rank,
                            neg_rank,
                            edge_label_index):
        rec_loss = torch.nn.functional.softplus(neg_rank - pos_rank).mean()
        user_emb = self.user_emb.weight
        item_emb = self.item_emb.weight
        embedding = torch.cat([user_emb[edge_label_index[0]],
                               item_emb[edge_label_index[1]],
                               item_emb[edge_label_index[2]]])
        regularization = self.config['decay'] * (1/2) * embedding.norm(p=2).pow(2)
        regularization = regularization / pos_rank.size(0)
        return rec_loss , regularization
    
    def message(self, x_j: Tensor) -> Tensor:
        return x_j
    
    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t,x)
    
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
        self.config = config
        self.user_emb = nn.Embedding(num_embeddings=num_users,
                                     embedding_dim=self.config['dim'])
        self.item_emb = nn.Embedding(num_embeddings=num_items,
                                     embedding_dim=self.config['dim'])
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
        if self.config['init'] == 'normal':
            nn.init.normal_(self.user_emb.weight,std=self.config['init_weight'])
            nn.init.normal_(self.item_emb.weight,std=self.config['init_weight'])
        else:
            nn.init.xavier_uniform_(self.user_emb.weight,gain=self.config['init_weight'])
            nn.init.xavier_uniform_(self.item_emb.weight,gain=self.config['init_weight'])

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