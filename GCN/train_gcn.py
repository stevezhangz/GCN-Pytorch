from torch import nn
import argparse
from GCN.utils import *
from model import GCN_with_D
from train_and_eval import *
from load_data import *
import logging as log
import time
# just test one time. on Cora dataset the find top1_acc about 86.04%, top5_acc about 99.93%
# train about 5000 epoch
# random seed set to 1e-3
# dont alow the adj matrix to update itself
log.basicConfig(
    level = log.INFO,
    format = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
    datefmt = '%a, %d %b %Y %H:%M:%S',
    filename = './report.log',
    filemode = 'w'
)
log.info('build args')
parser=argparse.ArgumentParser()
parser.add_argument('--random_seed',default=33)
parser.add_argument('--data_path',default='data/cora/')
parser.add_argument('--output_dir',default='result')
parser.add_argument('--learning_rate',default=1e-3)
parser.add_argument('--epoch',default=5000)
parser.add_argument('--use_adj',default=True)
parser.add_argument('--validation',default=True)
parser.add_argument('--hidden_size',default=3000)
parser.add_argument('--act',default='relu')
parser.add_argument('--use_degree',default=True)
parser.add_argument('--drop',default=0.5)
parser.add_argument('--learn_adj',default=False)
parser.add_argument('--device',default='cuda')
args=parser.parse_args()
log.info('set up random seed')
setup_seed(args.random_seed)
log.info('data loading')
feature,label,adj,degree,label_mapping=load_cora(args.data_path)
feature=torch.Tensor(feature)
label=torch.Tensor(label)
adj=torch.Tensor(adj)
feature.requires_grad=False
label.requires_grad=False
if args.learn_adj:
    adj.requires_grad=True
else:
    adj.requires_grad = False
log.info('model initialize')
GCN=GCN_with_D(infea=feature.size()[1],hidden=args.hidden_size,outfea=len(list(label_mapping.keys())),act=args.act,drop=args.drop)
log.info(f'default device: {args.device}')
device=torch.device(args.device)
GCN=GCN.to(device)
label=label.to(device)
feature=feature.to(device)
adj=adj.to(device)
optimizer=torch.optim.Adam(GCN.parameters(),lr=args.learning_rate)
log.info('model training')
loss_val, top1_acc, top5_acc=train(GCN,optimizer,args.epoch,feature=feature,label=label,adj=adj,loss=nn.CrossEntropyLoss(),
      save_dir=args.output_dir,validation=args.validation,val_data=None,log=log)
log.info('train end')
log.info(f'Current time: [{time.ctime()}]')
log.info(f'training result is shown as below:]')
log.info(f'\t training loss:       [{loss_val}]')
log.info(f'\t training top1_acc:   [{top1_acc}]')
log.info(f'\t training top5_acc:   [{top5_acc}]')