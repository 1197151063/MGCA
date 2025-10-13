import torch
from parse import parse_args
args = parse_args()


config = {}
config['bpr_batch_size'] = args.bpr_batch

config['latent_dim_rec'] = args.recdim

config['K'] = args.K

config['test_u_batch_size'] = args.testbatch

config['epochs'] = args.epochs

config['dataset'] = args.dataset

GPU = torch.cuda.is_available()

device = torch.device('cuda' if GPU else "cpu")

seed = args.seed

dataset = args.dataset

TRAIN_epochs = args.epochs

num_neg = args.num_neg

gamma = args.gamma

alpha = args.alpha

dropout = args.dropout
tau1 = args.tau1
tau2 = args.tau2
decay = args.decay
lambda_1 = args.lambda_1
decay = args.decay
flag = 0
seed = args.seed
def cprint(words: str):
    print(f"\033[0;30;43m{words}\033[0m")

def bprint(words:str):
    print(f"\033[0;30;45m{words}\033[0m")