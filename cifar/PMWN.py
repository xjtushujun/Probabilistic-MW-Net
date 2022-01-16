# -*- coding: utf-8 -*-

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
# import sklearn.metrics as sm
# import pandas as pd
# import sklearn.metrics as sm
import random
import numpy as np

from torch.distributions.beta import Beta
from torch.distributions.kl import kl_divergence

# from wideresnet import WideResNet, VNet
from resnet import ResNet32,VNet,MetaSGD
from load_corrupted_data import CIFAR10, CIFAR100

parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--corruption_prob', type=float, default=0.4,
                    help='label noise')
parser.add_argument('--corruption_type', '-ctype', type=str, default='unif',
                    help='Type of corruption ("unif" or "flip" or "flip2").')
parser.add_argument('--num_meta', type=int, default=1000)
parser.add_argument('--epochs', default=60, type=int,
                    help='number of total epochs to run')
parser.add_argument('--iters', default=60000, type=int,
                    help='number of total iters to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', '--batch-size', default=100, type=int,
                    help='mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--layers', default=28, type=int,
                    help='total number of layers (default: 28)')
parser.add_argument('--widen-factor', default=10, type=int,
                    help='widen factor (default: 10)')
parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='WideResNet-28-10', type=str,
                    help='name of experiment')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--prefetch', type=int, default=0, help='Pre-fetching threads.')
parser.set_defaults(augment=True)

args = parser.parse_args()
use_cuda = True
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")


print()
print(args)

def build_dataset():
    if args.dataset == 'cifar10':
        train_transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
        train_data_meta = CIFAR10(
            root='../data', train=True, meta=True, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True)
        train_data = CIFAR10(
            root='../data', train=True, meta=False, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True, seed=args.seed)
        test_data = CIFAR10(root='../data', train=False, transform=test_transform, download=True)


    elif args.dataset == 'cifar100':
        train_transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
             transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)), ])
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
            (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)), ])
        train_data_meta = CIFAR100(
            root='../data', train=True, meta=True, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True)
        train_data = CIFAR100(
            root='../data', train=True, meta=False, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True, seed=args.seed)
        test_data = CIFAR100(root='../data', train=False, transform=test_transform, download=True)


    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.prefetch, pin_memory=True)
    train_meta_loader = torch.utils.data.DataLoader(
        train_data_meta, batch_size=args.batch_size, shuffle=True,
        num_workers=args.prefetch, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.prefetch, pin_memory=True)

    return train_loader, train_meta_loader, test_loader


def build_model():
    model = ResNet32(args.dataset == 'cifar10' and 10 or 100)

    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True

    return model

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def adjust_learning_rate(optimizer, epochs):
    lr = args.lr * ((0.1 ** int(epochs >= 40)) * (0.1 ** int(epochs >= 50)))  # For WRN-28-10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def test(model, test_loader):
    model.eval()
    correct = 0
    test_loss = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_loss +=F.cross_entropy(outputs, targets).item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))

    return accuracy


def train(train_loader,train_meta_loader,model, vnet_alpha,vnet_beta,optimizer_model,optimizer_alpha,optimizer_beta,v_prior,epoch):
    print('\nEpoch: %d' % epoch)

    train_loss = 0
    meta_loss = 0

    train_meta_loader_iter = iter(train_meta_loader)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        model.train()
        inputs, targets = inputs.to(device), targets.to(device)
        meta_model = build_model().cuda()
        meta_model.load_state_dict(model.state_dict())
        outputs = meta_model(inputs)

        cost = F.cross_entropy(outputs, targets, reduce=False)
        cost_v = torch.reshape(cost, (len(cost), 1))
        alpha = vnet_alpha(cost_v.data)
        beta = vnet_beta(cost_v.data)
        v_post = Beta(alpha, beta)
        v_lambda = v_post.rsample()
        l_f_meta = torch.sum(cost_v * v_lambda) / len(cost_v)
        meta_model.zero_grad()
        grads = torch.autograd.grad(l_f_meta, (meta_model.params()), create_graph=True)

        meta_lr = args.lr * ((0.1 ** int(epoch >= 40)) * (0.1 ** int(epoch >= 50)))   # For ResNet32
        meta_model.update_params(lr_inner=meta_lr, source_params=grads)
        del grads

        try:
            inputs_val, targets_val = next(train_meta_loader_iter)
        except StopIteration:
            train_meta_loader_iter = iter(train_meta_loader)
            inputs_val, targets_val = next(train_meta_loader_iter)
        inputs_val, targets_val = inputs_val.to(device), targets_val.to(device)
        y_g_hat = meta_model(inputs_val)
        l_g_meta = F.cross_entropy(y_g_hat, targets_val) +0.001*meta_lr* kl_divergence(v_post, v_prior).mean()
        prec_meta = accuracy(y_g_hat.data, targets_val.data, topk=(1,))[0]


        optimizer_alpha.zero_grad()
        optimizer_beta.zero_grad()
        l_g_meta.backward()
        optimizer_alpha.step()
        optimizer_beta.step()

        outputs = model(inputs)
        cost_w = F.cross_entropy(outputs, targets, reduce=False)
        cost_v = torch.reshape(cost_w, (len(cost_w), 1))
        prec_train = accuracy(outputs.data, targets.data, topk=(1,))[0]

        with torch.no_grad():
            a = vnet_alpha(cost_v)
            b = vnet_beta(cost_v)
            w_post = Beta(a, b)
            w_new = w_post.rsample()

        loss = torch.sum(cost_v * w_new)/len(cost_v)
        v_prior = Beta(a.mean(),b.mean())

        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()


        train_loss += loss.item()
        meta_loss += l_g_meta.item()


        if (batch_idx + 1) % 100 == 0:
            print('Epoch: [%d/%d]\t'
                  'Iters: [%d/%d]\t'
                  'Loss: %.4f\t'
                  'MetaLoss:%.4f\t'
                  'Prec@1 %.2f\t'
                  'Prec_meta@1 %.2f' % (
                      (epoch + 1), args.epochs, batch_idx + 1, len(train_loader.dataset)/args.batch_size, (train_loss / (batch_idx + 1)),
                      (meta_loss / (batch_idx + 1)), prec_train, prec_meta))


train_loader, train_meta_loader, test_loader = build_dataset()
# create model
model = build_model()
vnet_alpha = VNet(1, 100, 1).cuda()
vnet_beta = VNet(1, 100, 1).cuda()
v_prior = Beta(torch.tensor([1.0]).cuda(), torch.tensor([1.0]).cuda())

if args.dataset == 'cifar10':
    num_classes = 10
if args.dataset == 'cifar100':
    num_classes = 100


optimizer_model = torch.optim.SGD(model.params(), args.lr,
                                  momentum=args.momentum, weight_decay=args.weight_decay)
optimizer_alpha = torch.optim.Adam(vnet_alpha.params(), lr=1e-3,
                                 weight_decay=1e-4)

optimizer_beta = torch.optim.Adam(vnet_beta.params(), lr=1e-3,weight_decay=1e-4)
                              

def main():
    time_start = time.time()
    time_all = []
    best_acc = 0
    for epoch in range(args.epochs):
        time_epoch_start = time.time()
        adjust_learning_rate(optimizer_model, epoch)
        train(train_loader,train_meta_loader,model, vnet_alpha,vnet_beta,optimizer_model,optimizer_alpha,optimizer_beta,v_prior,epoch)
        time_epoch_train = time.time()
        test_acc = test(model=model, test_loader=test_loader)
        time_epoch_test = time.time()
        # torch.save(vnet_alpha.state_dict(), './result/vnet_alpha_%s_%d.pth' % (args.dataset,epoch + 1))
        # torch.save(vnet_beta.state_dict(), './result/vnet_beta_%s_%d.pth' % (args.dataset, epoch + 1))

        if test_acc >= best_acc:
            best_acc = test_acc

        time_all.append([round(time_epoch_train - time_epoch_start, 3), round(time_epoch_test- time_epoch_train, 3), round(time_epoch_test - time_epoch_start, 3)])        

    print('best accuracy:', best_acc)
    torch.save(time_all, 'pmwn_time_all.pth')
    time_end = time.time()
    print('all running time is ', round(time_end - time_start, 3))


if __name__ == '__main__':
    main()
