'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pickle
import random

import os
import argparse
from mganet import MGANET

from utils import progress_bar
from leafvein import Leafvein

## reproducility
seed=21
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

test_acc=[]
train_acc=[]
train_ce_losses=[]
test_ce_losses=[]
train_mse_losses=[]
test_mse_losses=[]

parser = argparse.ArgumentParser(description='PyTorch Testing')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--max_epoch', default=150,type=int,
                    help='resume from checkpoint')
parser.add_argument('--backbone_class', type=str,default='densenet161',choices=['densenet161','vgg19','resnet50',
                                                                               'mobilenet_v2','inception_v3'])
parser.add_argument('--dataset', type=str,default='soybean',choices=['soybean','btf','hainan_leaf'])
parser.add_argument('--data_dir', type=str,default='./data')
parser.add_argument('--checkpoint_path', type=str,default='./checkpoint/ckpt.pth')
parser.add_argument('--num_classes', default=200, type=int, help='num class')
parser.add_argument('--batch_size', default=32, type=int, help='8 samples for 1 gpu')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
best_epoch = 0 # the epoch for the best accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

num_classes=args.num_classes
att_type='two'
mask_guided=True
model_name=args.backbone_class

test = Leafvein(args,mode='test')

testloader = DataLoader(test, batch_size=1, shuffle=False, num_workers=8)


# Model
print('==> Building model..')

model=MGANET(backbone_name=model_name,num_classes=num_classes,att_type=att_type,mask_guided=mask_guided)
net = model.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
checkpoint = torch.load(args.checkpoint_path)
net.load_state_dict(checkpoint['net'])

criterion = nn.CrossEntropyLoss()
'''
optimizer = optim.SGD([{'params': net.module.features.parameters(),'lr': 0.2*args.lr},
                       {'params': net.module.attention.parameters()},
                       {'params': net.module.classifier.parameters()}], 
                      lr=args.lr,momentum=0.9, weight_decay=5e-4)
                      '''
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)


def test(epoch):
    global best_acc
    global best_epoch
    net.eval()
    test_loss = 0
    ce_loss = 0
    mse_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets,masks) in enumerate(testloader):
            inputs, targets= inputs.to(device), targets.to(device)
            outputs = net(inputs)
            ce_loss_ = criterion(outputs, targets)
            #mse_loss_=0
            loss =  ce_loss_ 
            
            ce_loss += ce_loss_.item()
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'CE: %.3f | Acc: %.3f%% (%d/%d)'
                         % (ce_loss/(batch_idx+1), 100.*correct/total, correct, total))
            inputs,targets,outputs=None,None,None

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
        best_epoch=epoch
    test_ce_losses.append(ce_loss/(batch_idx+1))
    test_mse_losses.append(mse_loss/(batch_idx+1))
    test_acc.append(acc)
    print('cur_acc:{0},best_acc:{1}:'.format(acc,best_acc))



test(0)
print('best epoch:{0},best acc:{1}'.format(best_epoch,best_acc))
data={'train_acc':train_acc,'test_acc':test_acc,'train_ce_losses':train_ce_losses,'test_ce_losses':test_ce_losses,\
     'train_mse_losses':train_mse_losses,'test_mse_losses':test_mse_losses}
with open('data/summary/stats.pkl','wb') as df:
    pickle.dump(data,df)
