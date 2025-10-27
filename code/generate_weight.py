import world
import torch 
import numpy as np
from torch_sparse import SparseTensor
from  dataloader import Loader
from torch_geometric.utils import degree
from freedom import FREEDOM
dataset = Loader()
edge_index = dataset.train_edge_index
num_users = dataset.num_users
num_items = dataset.num_items
v_feat = dataset.v_feat
t_feat = dataset.t_feat
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

# model = FREEDOM(num_users=num_users,
#                  num_items=num_items,
#                  edge_index=edge_index,
#                  config=config,
#                 v_feat=v_feat,
#                 t_feat=t_feat).to(world.device)
def ra_item_item_pyg(edge_index, num_users=None, num_items=None):
    # edge_index: (2,E), row=user, col=item
    row, col = edge_index
    if num_users is None: num_users = int(row.max()) + 1
    if num_items is None: num_items = int(col.max()) + 1

    A = SparseTensor(row=row, col=col,
                     value=torch.ones_like(row, dtype=torch.float32),
                     sparse_sizes=(num_users, num_items)).coalesce()

    # 用户度 deg_u
    deg_u = A.sum(dim=1).to_dense().clamp_min_(1.0)  # (U,)
    w_u = 1.0 / deg_u

    # S = A^T * diag(w_u) * A
    A_w = scale_rows_sparse(A, w_u)
    S = A.t() @ A_w
    S = S.set_diag(0).coalesce()

    ii, jj, vv = S.coo()  # item_i, item_j, score
    return ii, jj, vv
def ppr_item_item_from_bipartite(edge_index, num_users, num_items, alpha=0.15, iters=20, topk=20):
    """
    在二部图上计算 item→item 的 PPR（经由 user 中转的两跳转移）。
    返回：PPR 矩阵（稀疏或按需 topk 裁剪）。
    edge_index: (2, E)  row=user, col=item
    """
    u, i = edge_index
    device = u.device

    # B: U×I（二值）
    B = SparseTensor(row=u, col=i, value=torch.ones_like(u, dtype=torch.float32, device=device),
                     sparse_sizes=(num_users, num_items)).coalesce()

    # 度
    deg_u = B.sum(dim=1).to_dense().clamp_min_(1.0)     # (U,)
    deg_i = B.sum(dim=0).to_dense().clamp_min_(1.0)     # (I,)

    # P_{U->I} = D_U^{-1} B ； P_{I->U} = D_I^{-1} B^T
    P_ui = B.mul_row(1.0 / deg_u) if hasattr(B, 'mul_row') else _scale_rows(B, 1.0 / deg_u)
    P_iu = B.t().mul_row(1.0 / deg_i) if hasattr(B.t(), 'mul_row') else _scale_rows(B.t(), 1.0 / deg_i)

    # 有效 item→item 转移：P_ii = P_iu @ P_ui   (I × I)，行随机
    P_ii = (P_iu @ P_ui).coalesce().set_diag(0)  # 去自环可选

    # PPR 迭代（全体 items 的种子矩阵，一次性计算所有源的 PPR）
    I = num_items
    eye = torch.eye(I, device=device)
    X = (1 - alpha) * eye  # 初始：每列为对应种子 e_s 的 (1-alpha)
    P = P_ii

    # power iteration
    for _ in range(iters):
        X = alpha * (P @ X) + (1 - alpha) * eye  # 每列一条个性化游走

    # 按需裁剪 Top-K（可极大缩小稀疏度）
    if topk is not None:
        # 对每一行取 topk，构造稀疏
        rows = []
        cols = []
        vals = []
        X_dense = X  # 这里 X 是稠密；若太大，改按批或改用 push 算法
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
        X_sparse = SparseTensor(row=row, col=col, value=val, sparse_sizes=(I, I)).coalesce()
        return X_sparse

    return X  # (I, I) 稠密；大图建议 topk 或 push

def _scale_rows(A: SparseTensor, w_row: torch.Tensor) -> SparseTensor:
    row, col, val = A.coo()
    if val is None:
        val = torch.ones_like(row, dtype=torch.float32, device=row.device)
    val = val * w_row[row]
    return SparseTensor(row=row, col=col, value=val, sparse_sizes=A.sparse_sizes()).coalesce()
def aa_item_item_pyg(edge_index, num_users=None, num_items=None):
    row, col = edge_index  # row=user, col=item (shape=[E])

    A = SparseTensor(row=row, col=col, value=torch.ones_like(row, dtype=torch.float32),
                     sparse_sizes=(num_users, num_items))  # U×I

    # 用户度 deg_u
    deg_u = degree(edge_index[0],num_users)
    w_u = 1.0 / torch.log1p(torch.clamp(deg_u, min=1.0))  # AA 权重

    # 等价于 A^T * diag(w_u) * A
    # 先左乘 diag(w_u) -> 对每个用户行加权
    # torch_sparse 支持行缩放：
    A_w = scale_rows_sparse(A, w_u)

    S = A.t() @ A_w       # I×I
    S = S.set_diag(0)

    # 取稀疏 COO
    ii, jj, vv = S.coo()
    return ii, jj, vv
def cosine_item_item_pyg(edge_index, num_users=None, num_items=None, eps: float = 1e-12):
    row, col = edge_index
    if num_users is None: num_users = int(row.max()) + 1
    if num_items is None: num_items = int(col.max()) + 1

    A = SparseTensor(row=row, col=col,
                     value=torch.ones_like(row, dtype=torch.float32),
                     sparse_sizes=(num_users, num_items)).coalesce()

    # item 度（被多少用户交互）
    deg_i = A.sum(dim=0).to_dense().clamp_min_(eps)  # (I,)
    S = (A.t() @ A).set_diag(0).coalesce()           # 共现计数

    ii, jj, vv = S.coo()
    norm = (deg_i[ii] * deg_i[jj]).sqrt_().clamp_min_(eps)
    vv = vv / norm
    return ii, jj, vv
def scale_rows_sparse(A: SparseTensor, w_row: torch.Tensor) -> SparseTensor:
    """
    返回 D @ A，其中 D = diag(w_row)，等价于对 A 的每一行做缩放。
    A: SparseTensor, shape = (num_rows, num_cols)
    w_row: Tensor, shape = (num_rows,)
    """
    # 取 COO（三元组）
    row, col, val = A.coo()                # row, col: (nnz,), val: (nnz,) or None
    if val is None:
        val = torch.ones_like(row, dtype=torch.float32, device=row.device)
    # 每条边的值乘以其行对应的权重
    val = val * w_row[row]
    # 重新构造稀疏张量（保持原尺寸、设备、排序状态）
    return SparseTensor(row=row, col=col, value=val,
                        sparse_sizes=A.sparse_sizes()).coalesce()
# ii, jj, vv = aa_item_item_pyg(edge_index, num_users, num_items)
X = ppr_item_item_from_bipartite(edge_index, num_users, num_items, alpha=0.15, iters=20, topk=20)
jj, ii, vv = X.coo()
test_edge_index = dataset.test_edge_index
test_user = test_edge_index[0]
test_item = test_edge_index[1]
test_graph = SparseTensor(row=test_user, col=test_item, value=torch.ones_like(test_user, dtype=torch.float32), sparse_sizes=(num_users, num_items))
print(test_graph.nnz())
test_graph = test_graph.to_dense()
train_graph = SparseTensor(row=edge_index[0], col=edge_index[1], value=torch.ones_like(edge_index[0], dtype=torch.float32), sparse_sizes=(num_users, num_items))
train_graph = train_graph.to_dense()
item_edge_index = torch.stack([jj, ii], dim=0)
item_edge_value = vv
item_graph = SparseTensor(row=item_edge_index[0], col=item_edge_index[1], value=item_edge_value, sparse_sizes=(num_items, num_items))
item_graph = item_graph.to_dense()
item_indicies,_ = torch.topk(item_graph, k=10, dim=1)
# mm_adj = model.mm_adj.to_dense()
# print(mm_adj)
for uid in range(10):  # 只看前10个用户
    # 用户在 test 里的真实物品列表
    user_item_list = train_graph[uid].nonzero(as_tuple=False)
    test_user_item_list = test_graph[uid].nonzero(as_tuple=False)   
    for i in user_item_list:
        print(item_graph[i,test_user_item_list])
    # print(user_item_list)
    # for a in user_item_list:
    #     item_item = item_indicies[a]
    # print(item_graph[user_item_list,user_item_list])
    # # print(f"User {uid} 的测试物品列表: ")
    # # print(user_item_list)
    # if len(user_item_list) == 0:
    #     print(f"User {uid} 没有测试物品，跳过。")
    #     continue

    # 收集所有真实物品的 Top-K 邻居
    neighbor_items = item_indicies[user_item_list].reshape(-1)  # [len(user_item_list)*20]

    # 命中：邻居中出现在 test 集的比例
    hit_items = torch.isin(neighbor_items, test_user_item_list)  # 返回 bool 向量
    hit_count = hit_items.sum().item()
    hit_rate = hit_count / len(neighbor_items)

    # print(f"User {uid} 命中率: {hit_rate:.4f} ({hit_count}/{len(neighbor_items)})")