from model import FREEDOM
from scipy.stats import gaussian_kde
import utils
import torch
import world
import time
from torch import Tensor
from procedure import train_bpr
from dataloader import Loader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from shapely.geometry import Point, MultiPoint
from shapely.ops import unary_union
from scipy.ndimage import gaussian_filter
from descartes import PolygonPatch
from sklearn.decomposition import PCA
config = {
        'init':'uniform',#NORMAL DISTRIBUTION
        'init_weight':1,#INIT WEIGHT
        'K':2,#GCN_LAYER
        'dim':64,#EMBEDDING_SIZE
        'decay':1e-4,#L2_NORM
        'lr':1e-3,#LEARNING_RATE
        'seed':0,#RANDOM_SEED
    }
dataset = Loader()
device = world.device
@torch.no_grad()
def pca_to(x: torch.Tensor, out_dim: int, seed: int = 42) -> torch.Tensor:
    """把任意维度的张量投到 out_dim 维（用于对齐特征维度）"""
    if x.size(1) == out_dim:
        return x
    x_np = x.detach().cpu().numpy()
    z = PCA(n_components=out_dim, random_state=seed).fit_transform(x_np)
    return torch.from_numpy(z).to(x.device, dtype=x.dtype)

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
state_dict = torch.load('../checkpoints/baby_FREEDOM.pth', map_location=device)
model.load_state_dict(state_dict)
user_emb = model.user_emb.weight
item_emb = model.item_emb.weight

num_user_samples = 200
num_item_samples = 400

user_idx = torch.randint(0, user_emb.size(0), (num_user_samples,))
item_idx = torch.randint(0, item_emb.size(0), (num_item_samples,))

sample_user_emb = user_emb[user_idx].detach().cpu()
sample_item_emb = item_emb[item_idx].detach().cpu()
image_trs = model.image_trs
text_trs = model.text_trs
v_feat = image_trs(model.image_embedding.weight)
t_feat = text_trs(model.text_embedding.weight)
# 同步采样对应的多模态特征（保证索引一致）
sample_v_feat = v_feat[item_idx].detach().cpu()
sample_t_feat = t_feat[item_idx].detach().cpu()

# 拼接所有要可视化的 embedding
X = torch.cat([
    sample_user_emb,
    # sample_item_emb,
    sample_v_feat,
    sample_t_feat
], dim=0).numpy()

# 构造标签用于着色
labels = (
    ['user_emb'] * num_user_samples +
    # ['item_emb'] * num_item_samples +
    ['v_feat'] * num_item_samples +
    ['t_feat'] * num_item_samples
)

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

tsne = TSNE(n_components=2, perplexity=30, n_iter=1500, random_state=42)
X_2d = tsne.fit_transform(X)
from matplotlib.patches import Patch

# 假设你已有 X_2d (n,2) 和 labels (n,)
X2 = np.asarray(X_2d)
labs = np.asarray(labels)

# 类别与颜色（按你的语义来）
classes = ['user_emb', 'v_feat', 't_feat']  # 如有其他类，补上
colors  = {'user_emb':'tab:blue', 'v_feat':'tab:green', 't_feat':'tab:red'}

# 1) 设定绘制网格（带些边距）
xmin, xmax = X2[:,0].min(), X2[:,0].max()
ymin, ymax = X2[:,1].min(), X2[:,1].max()
padx = 0.05 * (xmax - xmin + 1e-9)
pady = 0.05 * (ymax - ymin + 1e-9)
xmin, xmax = xmin - padx, xmax + padx
ymin, ymax = ymin - pady, ymax + pady

H = W = 500  # 分辨率（越大越细腻）
xx, yy = np.meshgrid(np.linspace(xmin, xmax, W), np.linspace(ymin, ymax, H))
grid = np.vstack([xx.ravel(), yy.ravel()])  # (2, H*W)

# 2) 每类做 KDE 得到密度图
dens_list = []
valid_classes = []
bw = 0.2  # 带宽（0.1~0.3 可调；越小边界越锋利）
for cls in classes:
    pts = X2[labs == cls]
    if len(pts) < 3:
        continue  # 样本太少跳过
    kde = gaussian_kde(pts.T, bw_method=bw)
    dens = kde(grid).reshape(H, W)
    dens_list.append(dens)
    valid_classes.append(cls)

if len(dens_list) == 0:
    raise RuntimeError("没有足够的点绘制KDE语义色块。")

dens_stack = np.stack(dens_list, axis=0)  # (C,H,W)

# 3) 生成语义分区：每个像素取密度最大的类别
dom_idx = np.argmax(dens_stack, axis=0)               # (H,W) -> 类别索引
tot_den = np.clip(np.sum(dens_stack, axis=0), 1e-12, None)  # 总密度，用于做背景遮罩

# 4) 可选：平滑一下，让边界更柔和
dom_idx_smooth = dom_idx.copy()
# 用总密度做背景阈值（低密度区域不画，避免“漂移的色块”）
thr = np.percentile(tot_den, 40)  # 40分位阈值可调（30~60）
mask = tot_den >= thr

# 5) 把类别索引映射为颜色
rgb = np.zeros((H, W, 3), dtype=float)
for i, cls in enumerate(valid_classes):
    rgb[dom_idx_smooth == i] = plt.get_cmap('tab10')(i)[:3]

# 6) 叠加连续“强度”作为透明度（密度越高越不透明）
alpha = (tot_den - tot_den[mask].min()) / (tot_den[mask].ptp() + 1e-9)
alpha = np.clip(alpha, 0, 1)
alpha = gaussian_filter(alpha, sigma=1.0)  # 进一步柔化透明度
alpha[~mask] = 0.0

# 7) 绘制
fig, ax = plt.subplots(figsize=(7, 6))
ax.imshow(
    np.transpose(rgb, (0,1,2)), origin='lower',
    extent=[xmin, xmax, ymin, ymax], alpha=None  # 颜色本身不透明
)
# 再叠一层灰色密度热影，增强“连续感”（可去掉）
ax.imshow(
    alpha.T, origin='lower', extent=[xmin, xmax, ymin, ymax],
    cmap='Greys', alpha=0.35  # 轻微影子
)

# 可选：叠少量散点作锚点（让读者知道语义来源；不需要可注释）
for cls in valid_classes:
    m = (labs == cls)
    P = X2[m]
    ax.scatter(P[:,0], P[:,1], s=6, alpha=0.25, color=colors.get(cls, 'k'), edgecolors='none')

# 轴样式
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
ax.set_title("Embedding Semantic Field (KDE-based patches)", fontsize=14)

# 图例（用色块代理）
handles = [Patch(facecolor=colors.get(c, plt.get_cmap('tab10')(i)[:3]), edgecolor='none', label=c)
           for i, c in enumerate(valid_classes)]
ax.legend(handles=handles, frameon=False, loc='best')

plt.tight_layout()
plt.savefig("embedding_semantic_patches.png", dpi=400)