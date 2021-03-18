# TITAN V / 2080Ti / TITAN Xp
# 1.60 + cu101
# torch-geometric 1.6.3

# run 12 times and remove(min(acc)), remove(max(acc))
# pubmed: 79.0 ± 0.6
# citeseer: 70.8 ± 0.5
# cora: 81.4 ± 0.4

import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import numpy as np
import random
import torch_geometric.transforms as T
import argparse
from torch_geometric.nn import GCNConv, ChebConv
# from loss_gaussian import focal_loss
# loss_fn = focal_loss(alpha=[1]*7, gamma=0.2, num_classes=7)

parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
parser.add_argument('--dataset', default='cora')
args = parser.parse_args()

dataset = args.dataset
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]

if args.use_gdc:
    gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                normalization_out='col',
                diffusion_kwargs=dict(method='ppr', alpha=0.05),
                sparsification_kwargs=dict(method='topk', k=128,
                                           dim=0), exact=True)
    data = gdc(data)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 16, cached=True,
                             normalize=not args.use_gdc)

        self.conv2 = GCNConv(16, dataset.num_classes, cached=True,
                             normalize=not args.use_gdc)

        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training, p=0.0 if args.dataset=='pubmed' else 0.5)
        x = self.conv2(x, edge_index, edge_weight)
        return x


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train():
    model.train()
    optimizer.zero_grad()
    logit, d_loss = model()
    pred = F.log_softmax(logit, dim=1)
    loss = F.nll_loss(pred[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()


def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

seed_acc = []
for seed in [150, 200, 250, 300, 350, 400, 500, 550, 600, 650]:
    setup_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, data = Net().to(device), data.to(device)
    optimizer = torch.optim.Adam([
        dict(params=model.conv1.parameters(), weight_decay=5e-4),
        dict(params=model.conv2.parameters(), weight_decay=0)
    ], lr=0.01 if args.dataset=='cora' else 0.005)  # Only perform weight-decay on first convolution

    best_val_acc = test_acc = 0
    for epoch in range(1, 201):
        train()
        train_acc, val_acc, tmp_test_acc = test()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        if epoch%50==0:
            log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
            print(log.format(epoch, train_acc, best_val_acc, test_acc))
    print('==' * 20)
    seed_acc.append(test_acc)
print(np.mean(seed_acc), np.std(seed_acc))
