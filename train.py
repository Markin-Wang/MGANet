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
seed=213
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

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--max_epoch', default=150,type=int,
                    help='resume from checkpoint')
parser.add_argument('--backbone_class', type=str,default='densenet161',choices=['densenet161','vgg19','resnet50',
                                                                               'mobilenet_v2','inception_v3'],
                    help='resume from checkpoint')
parser.add_argument('--dataset', type=str,default='soybean',choices=['soybean_1_1','soybean_2_1','btf','hainan_leaf'],
                    help='resume from checkpoint')
parser.add_argument('--data_dir', type=str,default='./data')
parser.add_argument('--num_classes', default=200, type=int, help='num class')
parser.add_argument('--batch_size', default=32, type=int, help='8 samples for 1 gpu')

args = parser.parse_args()

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
best_epoch = 0 # the epoch for the best accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

dataset=args.dataset

batchsize = args.batch_size
num_classes=args.num_classes
att_type= 'two'
mask_guided=True
model_name=args.backbone_class
'''
    transforms.Normalize(mean=[0.616, 0.638, 0.589],
                         std=[0.430, 0.406, 0.461])
'''


train = Leafvein(args,crop=[448,448],hflip=True,vflip=False,erase=True,mode='train')

test = Leafvein(args,mode='test')

trainloader = DataLoader(train, batch_size=batchsize, shuffle=True, num_workers=8)
testloader = DataLoader(test, batch_size=1, shuffle=False, num_workers=8)


# Model
print('==> Building model..')

model=MGANET(backbone_name=model_name,num_classes=num_classes,att_type=att_type,mask_guided=mask_guided)
#print(model)

'''
Inception V3 need to be processed seperately, please delete the annotation if you use inception_v3 as backbone_class
model = torchvision.models.inception_v3(pretrained=True)
print(model)
model.aux_logits=False
in_features = model.fc.in_features

model.fc = nn.Linear(in_features, 10)


ct = 0
for name, child in model.named_children():
    ct += 1
    if ct < 14:
        for name2,param in child.named_parameters():
            param.requires_grad = False
            print(name,name2,ct)
        
'''

net = model.to(device)
if device[0:4] == 'cuda':
    net = torch.nn.DataParallel(net,device_ids=[1,0])
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
mse = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    mse_loss = 0
    ce_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets, masks) in enumerate(trainloader):
        inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)
        optimizer.zero_grad()
        if mask_guided:
            outputs,fg_atts,masks = net(inputs,masks)
            mse_loss_ = mse(fg_atts,masks)
            ce_loss_ = criterion(outputs, targets)
            loss =  ce_loss_ + 0.1 * mse_loss_
            mse_loss += mse_loss_.item()
        else:
            outputs = net(inputs)
            ce_loss_ = criterion(outputs, targets)
            mse_loss_=0
            loss =  ce_loss_
            mse_loss += 0
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        
        ce_loss += ce_loss_.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'CE: %.3f, MSE: %.5f | Acc: %.3f%% (%d/%d)'
                     % (ce_loss/(batch_idx+1), mse_loss/(batch_idx+1) ,100.*correct/total, correct, total))
    train_ce_losses.append(ce_loss/(batch_idx+1))
    train_mse_losses.append(mse_loss/(batch_idx+1))
    train_acc.append(100.*correct/total)


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
            inputs, targets,masks= inputs.to(device), targets.to(device),masks.to(device)
            if mask_guided:
                outputs,fg_atts,masks = net(inputs,masks)
                mse_loss_ = mse(fg_atts,masks)
                ce_loss_ = criterion(outputs, targets)
                loss =  ce_loss_ + 0.1 * mse_loss_
                mse_loss += mse_loss_.item()
            else:
                outputs = net(inputs)
                ce_loss_ = criterion(outputs, targets)
                mse_loss_=0
                loss =  ce_loss_
                mse_loss += 0
            ce_loss += ce_loss_.item()
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'CE: %.3f, MSE: %.5f | Acc: %.3f%% (%d/%d)'
                         % (ce_loss/(batch_idx+1), mse_loss/(batch_idx+1), 100.*correct/total, correct, total))
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


for epoch in range(start_epoch, start_epoch+args.max_epoch):
    train(epoch)
    test(epoch)
    scheduler.step()
