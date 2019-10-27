from __future__ import division
from __future__ import print_function

import math
import random
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy, accuracy_InnerProduct
from models import KernelGCN


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=31, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, #default = 0.01
                    help='Initial learning rate.')
parser.add_argument('--samples', type=int, default=10000,
                    help='samples per triplet loss.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

parser.add_argument('--dataroot', type=str, default='data', help='path')
parser.add_argument('--dataset', type=str, default='cora', help='[cora]')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adjs, features, labels, idx_train, idx_val, idx_test, adj = load_data(path=args.dataroot, dataset=args.dataset)

nstep = len(adjs)
# Model and optimizer
model = KernelGCN(nfeat=features.shape[1],
                  nh1 = 16,
                  nclass=labels.max().item() + 1,
                  dropout=args.dropout)

model.apply(weights_init)

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    for i in range(nstep):
        adjs[i] = adjs[i].cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    

def split_labels(labels):
    nclass = torch.max(labels) + 1
    labels_split = []
    labels_split_numpy = []
    for i in range(nclass):
        labels_split.append((labels == i).nonzero().view([-1]))
    for i in range(nclass):
        labels_split_numpy.append(labels_split[i].cpu().numpy())

    labels_split_dif = []
    for i in range(nclass):
        dif_type = [x for x in range(nclass) if x != i]
        labels_dif = torch.cat([ labels_split[x] for x in dif_type ])
        labels_split_dif.append(labels_dif)
    return nclass, labels_split, labels_split_numpy, labels_split_dif


def triplet_loss_InnerProduct(nclass, labels_split, labels_split_dif, logits):
    n_sample = args.samples
   
    n_sample_class = (int)(n_sample / nclass)
    thre = 0.1
    loss = 0
    for i in range(nclass):
        # python2: xrange, python3: range
        randInds1 = random.choices(labels_split[i], k=n_sample_class)
        randInds2 = random.choices(labels_split[i], k=n_sample_class)
        
        feats1 = logits[randInds1]
        feats2 = logits[randInds2]
        randInds_dif = random.choices(labels_split_dif[i], k=n_sample_class)

        feats_dif = logits[randInds_dif]
        # inner product: same class inner product should > dif class inner product
        inner_products = torch.sum(torch.mul(feats1, feats_dif-feats2), dim=1)
        dists = inner_products + thre
        mask = dists > 0

        loss += torch.sum(torch.mul(dists, mask.float()))

    loss /= n_sample_class*nclass
    return loss


def getClassMean(nclass, labels_split, logits):
    class_mean = torch.cat([torch.mean(logits[labels_split[x]], dim=0).view(-1,1) for x in range(nclass)], dim=1)
    return class_mean


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    logits, output = model(features, adjs)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_triplet = triplet_loss_InnerProduct(nclass.item(), labels_split, labels_split_dif, logits)
    loss_all = loss_train + 0.1*loss_triplet

    loss_all.backward()
    optimizer.step()

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_all: {:.4f}'.format(loss_all.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def load_model(net, name):
    state_dict = torch.load(name)
    own_state = net.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            continue
        own_state[name].copy_(param)


def test():
    print('Epoch: {:04d}'.format(args.epochs),
          'lr: {:.4f}'.format(args.lr),
          'samples: {:04d}'.format(args.samples))
    
    model.eval()
    logits, output = model(features, adjs)

    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])

    class_mean = getClassMean(nclass.item(), labels_split, logits)

    if args.cuda:
        acc_test_innerproduct = accuracy_InnerProduct(labels.cpu().numpy(), logits.detach().cpu().numpy(), class_mean.detach().cpu().numpy(), idx_test.cpu().numpy())
    else:
        acc_test_innerproduct = accuracy_InnerProduct(labels.numpy(), logits.detach().numpy(), class_mean.detach().numpy(), idx_test.numpy())

    print("Test set results (last model):",
          "accuracy_innerproduct= {:.4f}".format(acc_test_innerproduct),
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    
    
# Train model
t_total = time.time()
nclass, labels_split, labels_split_numpy, labels_split_dif = split_labels(labels[idx_train])
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
