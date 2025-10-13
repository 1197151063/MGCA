# coding: utf-8
# @email: enoche.chow@gmail.com
r"""
FREEDOM
# Update: 01/08/2022
"""


import os
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss, L2Loss
from utils.utils import build_sim, compute_normalized_laplacian


class FREEDOM(GeneralRecommender):
    def __init__(self, config, dataset):
        super(FREEDOM, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.knn_k = config['knn_k']
        self.lambda_coeff = config['lambda_coeff']
        self.cf_model = config['cf_model']
        self.n_layers = config['n_mm_layers']
        self.n_ui_layers = config['n_ui_layers']
        self.reg_weight = config['reg_weight']
        self.build_item_graph = True
        self.mm_image_weight = config['mm_image_weight']
        self.dropout = config['dropout']
        self.degree_ratio = config['degree_ratio']

        self.n_nodes = self.n_users + self.n_items

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        self.masked_adj, self.mm_adj = None, None
        self.edge_indices, self.edge_values = self.get_edge_info()
        self.edge_indices, self.edge_values = self.edge_indices.to(self.device), self.edge_values.to(self.device)
        self.edge_full_indices = torch.arange(self.edge_values.size(0)).to(self.device)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)

        nn.init.xavier_uniform_(self.user_embedding.weight,2)
        nn.init.xavier_uniform_(self.item_id_embedding.weight,2)
        # nn.init.xavier_normal_(self.user_embedding.weight,5)
        # nn.init.xavier_normal_(self.item_id_embedding.weight,5)
        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        mm_adj_file = os.path.join(dataset_path, 'mm_adj_freedomdsp_{}_{}.pt'.format(self.knn_k, int(10*self.mm_image_weight)))

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False).to(self.device)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim,device=self.device)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False).to(self.device)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim,device=self.device)
        
        # self.image_sim_norm = self.get_knn_sim(self.image_embedding.weight.detach()).cuda()
        # self.text_sim_norm = self.get_knn_sim(self.text_embedding.weight.detach()).cuda()

        # if os.path.exists(mm_adj_file):
        #     self.mm_adj = torch.load(mm_adj_file)
        # else:
        #     if self.v_feat is not None:
        #         indices, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
        #         self.mm_adj = image_adj
        #     if self.t_feat is not None:
        #         indices, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
        #         self.mm_adj = text_adj
        #     if self.v_feat is not None and self.t_feat is not None:
        #         self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
        #         del text_adj
        #         del image_adj
        #     torch.save(self.mm_adj, mm_adj_file)
        self.image_knn_idx = self.get_knn_ind(self.image_embedding.weight.detach()).cuda()
        self.text_knn_idx = self.get_knn_ind(self.text_embedding.weight.detach()).cuda()
        self.image_knn_sim = self.get_knn_sim(self.image_embedding.weight.detach()).cuda()
        self.text_knn_sim = self.get_knn_sim(self.text_embedding.weight.detach()).cuda()

        # self.mm_adj = self.mm_adj.to_dense()
        # print(self.mm_adj)

    def uniformity_rescaling(self,embedding,dim):
        limit = torch.sqrt(torch.tensor(6.0 / (dim + dim))) 
        xavier_range = (-limit.item(), limit.item())
        min_val, max_val = xavier_range
        scaled_features = (embedding - embedding.min()) / (embedding.max() - embedding.min())
        scaled_features = scaled_features * (max_val - min_val) + min_val
        return scaled_features
    
    def get_knn_sim(self,mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        return sim
    
    def get_knn_ind(self,mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, 20 , dim=-1,sorted=False)
        return knn_ind
    
    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def pre_epoch_processing(self):
        if self.dropout <= .0:
            self.masked_adj = self.norm_adj
            return
        # degree-sensitive edge pruning
        degree_len = int(self.edge_values.size(0) * (1. - self.dropout))
        degree_idx = torch.multinomial(self.edge_values, degree_len)
        # random sample
        keep_indices = self.edge_indices[:, degree_idx]
        # norm values
        keep_values = self._normalize_adj_m(keep_indices, torch.Size((self.n_users, self.n_items)))
        all_values = torch.cat((keep_values, keep_values))
        # update keep_indices to users/items+self.n_users
        keep_indices[1] += self.n_users
        all_indices = torch.cat((keep_indices, torch.flip(keep_indices, [0])), 1)
        self.masked_adj = torch.sparse.FloatTensor(all_indices, all_values, self.norm_adj.shape).to(self.device)

    def _normalize_adj_m(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        cols_inv_sqrt = c_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return values

    def get_edge_info(self):
        rows = torch.from_numpy(self.interaction_matrix.row)
        cols = torch.from_numpy(self.interaction_matrix.col)
        edges = torch.stack([rows, cols]).type(torch.LongTensor)
        # edge normalized values
        values = self._normalize_adj_m(edges, torch.Size((self.n_users, self.n_items)))
        return edges, values


    def item_alignment(self,items,knn_ind,knn_sim):
        knn_neighbour = knn_ind[items] # [num_items_batch * knn_k]
        user_emb = self.item_id_embedding.weight[items].unsqueeze(1)
        item_emb = self.item_id_embedding.weight[knn_neighbour]
        sim_score = knn_sim[items][:,knn_neighbour]
        loss = -sim_score * (user_emb * item_emb).sum(dim=-1).sigmoid().log()
        return loss.sum()

    def forward(self, adj):
        # h = self.item_id_embedding.weight
        # for i in range(self.n_layers):
        #     h = torch.sparse.mm(self.mm_adj, h)
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings 
    
    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        return mf_loss

    def au_loss(self,users,items):
        alignment = self.alignment(users,items)
        u_uniformity = self.uniformity(users)
        i_uniformity = self.uniformity(items)
        return alignment +  0.1* (u_uniformity + i_uniformity)
    
    def InfoNCE_I_ALL(self,view1,view2,pos,t):
        view1 = F.normalize(view1,dim=1)
        view2 = F.normalize(view2,dim=1)
        view1_pos = view1[pos]
        view2_pos = view2[pos]
        info_pos = (view1_pos * view2_pos).sum(dim=1)/ t
        info_pos_score = torch.exp(info_pos)
        info_neg = (view1_pos @ view2.t())/ t
        info_neg = torch.exp(info_neg)
        info_neg = torch.sum(info_neg,dim=1,keepdim=True)
        info_neg = info_neg.T
        ssl_logits = -torch.log(info_pos_score / info_neg).mean()
        return ssl_logits
    
    def InfoNCE_U_ALL(self,view1,view2,u_idx,pos,t):
        view1 = F.normalize(view1,dim=1)
        view2 = F.normalize(view2,dim=1)
        view1_pos = view1[u_idx]
        view2_pos = view2[pos]
        info_pos = (view1_pos * view2_pos).sum(dim=1)/ t
        info_pos_score = torch.exp(info_pos)
        info_neg = (view1_pos @ view2.t())/ t
        info_neg = torch.exp(info_neg)
        info_neg = torch.sum(info_neg,dim=1,keepdim=True)
        info_neg = info_neg.T
        ssl_logits = -torch.log(info_pos_score / info_neg).mean()
        return ssl_logits
    
    def InfoNCE_U_BATCH(self,view1,view2,u_idx,pos,neg,t):
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
    
    def norm_loss(self):
        loss = 0.0
        for parameter in self.parameters():
            loss += torch.sum(parameter ** 2)
        return loss / 2

    def InfoNCE_UI(self,users,items,u_idx,i_idx,t):
        view1 = F.normalize(users,dim=1)
        view2 = F.normalize(items,dim=1)
        view1_pos = view1[u_idx]
        view2_pos = view2[i_idx]
        info_pos = (view1_pos * view2_pos).sum(dim=1)/ t
        info_pos_score = torch.exp(info_pos)
        info_neg = (view1_pos @ view2.t())/ t
        info_neg = torch.exp(info_neg)
        info_neg = torch.sum(info_neg,dim=1,keepdim=True)
        info_neg = info_neg.T
        ssl_logits = -torch.log(info_pos_score / info_neg).mean()
        return ssl_logits
    
    def multimodal_contrastive_loss(self,view1,view2,pos_indices,temperature=0.2):
        anchor_embeddings = view1[pos_indices]
        pos_embeddings = view2[pos_indices]
        neg_embeddings = view2
        pos_similarity = F.cosine_similarity(anchor_embeddings, pos_embeddings)
        anchor_embeddings = anchor_embeddings.unsqueeze(1)  # (batch_size, 1, embedding_dim)
        neg_similarity = F.cosine_similarity(anchor_embeddings, neg_embeddings, dim=2)
        exp_pos = torch.exp(pos_similarity / temperature)  # (batch_size,)
        exp_neg = torch.exp(neg_similarity / temperature).sum(dim=1)  # (batch_size,)
        loss = -torch.log(exp_pos / (exp_pos + exp_neg)).mean()
        return loss
    
    
    def align_loss(self,item_embeddings, image_embeddings, text_embeddings, alpha=0.5):
        image_loss = F.mse_loss(item_embeddings, image_embeddings)
        text_loss = F.mse_loss(item_embeddings, text_embeddings)
        loss = alpha * image_loss + (1 - alpha) * text_loss
        return loss
    
    def center_alignment(self,x,y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        x = x.mean(dim=0)
        y = y.mean(dim=0)
        return (x - y).pow(2).mean()


    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        text_feats = self.text_trs(self.text_embedding.weight)
        image_feats = self.image_trs(self.image_embedding.weight)     
        ua_embeddings, ia_embeddings = self.forward(self.norm_adj)
        self.build_item_graph = False

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]
        batch_mf_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
                                                neg_i_g_embeddings)
        # batch_mf_loss = self.au_loss(u_g_embeddings, pos_i_g_embeddings)
        # batch_mf_loss = self.InfoNCE_UI(ua_embeddings,ia_embeddings,users,pos_items,0.2)
        #align user embedding with mm embedding works
        # mf_v_loss = self.InfoNCE_U_ALL(ua_embeddings,image_feats,users,pos_items,0.1)
        # mf_t_loss = self.InfoNCE_U_ALL(ua_embeddings,text_feats,users,pos_items,0.1)
        mf_v_loss = self.InfoNCE_U_BATCH(ua_embeddings,image_feats,users,pos_items,neg_items,0.1)
        mf_t_loss = self.InfoNCE_U_BATCH(ua_embeddings,text_feats,users,pos_items,neg_items,0.1)
        v_alignment = self.item_alignment(pos_items,self.image_knn_idx,self.image_knn_sim)
        t_alignment = self.item_alignment(pos_items,self.text_knn_idx,self.text_knn_sim)
        alignment_loss = 0.1 * v_alignment + 0.9 * t_alignment
        # v_i_align = self.alignment(pos_i_g_embeddings,image_feats[pos_items])
        # t_i_align = self.alignment(pos_i_g_embeddings,text_feats[pos_items])
        # lambda_align = 0.3
        # v_i_align = self.structured_awared_alignment(ia_embeddings,pos_items,'v')
        # t_i_align = self.structured_awared_alignment(ia_embeddings,pos_items,'t')
        #intra align mm embedding 
        # mm_align = self.InfoNCE_I_ALL(image_feats,text_feats,pos_items,0.1) + self.InfoNCE_I_ALL(text_feats,image_feats,pos_items,0.1)
        # norm_loss = self.norm_loss() * 1e-4
        return batch_mf_loss + 0.01 * (mf_t_loss + mf_v_loss) + 5e-4* alignment_loss

    def InfoNCE(self,view1, view2, temperature: float, b_cos: bool = True):
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
    
    def cosine_similarity_loss(self,embedding1, embedding2):
        cos_sim = F.cosine_similarity(embedding1, embedding2)
        loss = 1 - torch.mean(cos_sim) 
        return loss
    
    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores

    def alignment(self,x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    def alignment_loss(self,edge_label_index):
        x_u,x_i = self.forward(self.norm_adj)
        batch_x_u,batch_x_i = x_u[edge_label_index[0]],x_i[edge_label_index[1]]
        return self.alignment(batch_x_u,batch_x_i)
    
    def uniformity_loss(self,edge_label_index):
        x_u,x_i = self.forward(self.norm_adj)
        batch_x_u,batch_x_i = x_u[edge_label_index[0]],x_i[edge_label_index[1]]
        return  (self.uniformity(batch_x_u) + self.uniformity(batch_x_i))

    def uniformity(self,x, t=2):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()
    
    def L2_regloss(self,user_emb,item_emb,users,pos,neg):
        u_idx = torch.unique(users)
        i_idx = torch.unique(torch.cat([pos,neg]))
        u_emb = user_emb[u_idx]
        i_emb = item_emb[i_idx]
        return  (1/2)*(u_emb.norm(2).pow(2) + i_emb.norm(2).pow(2))/ u_idx.size(0)
    