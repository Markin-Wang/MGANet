'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.nn import TripletMarginLoss
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import os
import argparse
import torchvision.models as models

from leafvein import Leafvein

class MGANET(nn.Module):
    def __init__(self,backbone_name,num_classes,att_type=None,mask_guided=False):
        super(MGANET, self).__init__()
        
        self.mask_guided=mask_guided
        if backbone_name=='densenet161':
            self.features=getattr(models,backbone_name)(pretrained=True).features
            self.classifier=nn.Linear(self.features[-1].num_features,num_classes)
        elif backbone_name[:6]=='resnet':
            self.features=getattr(models,backbone_name)(pretrained=True)
            self.classifier=nn.Linear(512,num_classes)
        elif backbone_name=='mobilenet_v2':
            self.features=getattr(models,backbone_name)(pretrained=True)
            self.classifier=nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.features.last_channel, num_classes)
            )
            self.features=self.features.features
        elif backbone_name=='vgg19':
            self.features=getattr(models,backbone_name)(pretrained=True)
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )
            self.features=self.features.features
        else:
            self.features=getattr(models,backbone_name)(pretrained=True).features
            num_features=self.features[-1].num_features
            self.classifier=nn.Linear(num_features,num_classes)
            
        self.backbone_name=backbone_name
        
        self._freeze_layers(self.features)
        
        if backbone_name[:6]=='resnet':
            self.features=nn.Sequential(*list(self.features.children())[:-2])
        self.att_type=att_type
        '''
        if self.att_type=='three':
            self.classifier=nn.Linear(num_features*2,num_classes)
        '''
        kernel_size=1
        self.attention=nn.Conv2d(2,1,kernel_size=kernel_size, bias=False)

    def getAttFeats(self,att_map,features,type=None):
        # params: one for simple att*features
        # two for cat att*feat and features
        if type=='one':
            features=att_map*features
        elif type=='two':
            features=0.5*features+0.5*(att_map*features)
        elif type=='three':
            features=torch.cat((features,att_map*features),dim=1)
        else:
            pass
        return features
    
        
        
        
    def forward(self,x,mask=None):
        
        features = self.features(x)
        # output size 14*14
        #print(features.shape)
        #foreground attention
        fg_att=self.attention(torch.cat((torch.mean(features,dim=1).unsqueeze(1),\
                                                torch.max(features,dim=1)[0].unsqueeze(1)),dim=1))
        #fg_att=torch.flatten(torch.sigmoid(fg_att),1)
        fg_att=torch.sigmoid(fg_att)  
        features=self.getAttFeats(fg_att,features,type=self.att_type)
        if self.backbone_name=='densenet161':
            features = F.relu(features, inplace=True)
        if self.backbone_name=='vgg19':
            out=F.adaptive_avg_pool2d(features, (7, 7))
        else:
            out = F.adaptive_avg_pool2d(features, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        
        #### for calculating mse loss
        if self.mask_guided:
            h,w = fg_att.shape[2],fg_att.shape[3]
            mask=F.adaptive_avg_pool2d(mask, (h, w))
            fg_att = fg_att.view(fg_att.shape[0],-1)
            mask = mask.view(mask.shape[0],-1)
            
            mask += 1e-12
            max_elmts=torch.max(mask,dim=1)[0].unsqueeze(1)
            mask = mask/max_elmts
                                  
        return (out,fg_att,mask) if self.mask_guided else out
    
    def _freeze_layers(self,model):
        cnt,th=0,0
        print('freeze layers:')
        if self.backbone_name=='densenet161':
            th=9
        elif self.backbone_name[:6]=='resnet':
            th=7
        elif self.backbone_name=='mobilenet_v2':
            th=11 # 10/19
        elif self.backbone_name=='vgg19':
            th=22 #9/16 conv
        else:
            th=10    
        for name, child in model.named_children():
            cnt+=1
            if cnt<th:
                for name2, params in child.named_parameters():
                    params.requires_grad = False
                    print(name,name2,cnt)
                    
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale
   

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


    

