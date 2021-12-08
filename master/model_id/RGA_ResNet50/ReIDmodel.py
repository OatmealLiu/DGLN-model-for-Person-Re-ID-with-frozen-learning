import os
import sys
import math
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models # import pretrained model
from torch.autograd import Variable
from backbone import RGA_ResNet50_alpha
from torch.nn import functional as F

__all__ = ['resnet50_rga']
WEIGHT_PATH = os.path.join(os.path.dirname(__file__), '../') + '/weights/pre_train/resnet50-19c8e357.pth'

# Define kaiming-initialization
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class_list = ['age', 'backpack', 'bag', 'handbag', 'clothes',
              'down', 'up', 'hair', 'hat', 'gender',
              'upcolor', 'downcolor']
numclass_dict = {class_list[i]: 2 for i in range(len(class_list))}
numclass_dict['age'] = 4
numclass_dict['upcolor'] = 9
numclass_dict['downcolor'] = 10


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True,
                 num_bottleneck=512, linear=True, returnF = False):
        super(ClassBlock, self).__init__()
        self.returnF = returnF
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.returnF:
            f = x
            x = self.classifier(x)
            return [x,f]
        else:
            x = self.classifier(x)
            return x


# +++++++++++++++++++++++++++++++++++++++
# RGA model
# +++++++++++++++++++++++++++++++++++++++
class ResNet50_RGA_Model(nn.Module):
    """
	Backbone: ResNet-50 + RGA modules.
	"""
    def __init__(self, pretrained=True, num_feat=2048, height=256, width=128,
                 num_classes=751, dropout=0, last_stride=1, scale=8, d_scale=8,
                 model_path=WEIGHT_PATH):
        super(ResNet50_RGA_Model, self).__init__()

        self.pretrained = pretrained
        self.num_feat = num_feat
        self.dropout = dropout
        self.num_classes = num_classes
        print('Num of features: {}.'.format(self.num_feat))

        # initialize backbone as ResNet-50 + RGA attention modules model
        self.backbone = RGA_ResNet50_alpha(pretrained=pretrained, last_stride=last_stride, height=height,
                                           width=width,s_ratio=scale, c_ratio=scale, d_ratio=d_scale,
                                           model_path=model_path)

        # define our last FC and Classifier layers
        self.feat_bn = nn.BatchNorm1d(self.num_feat)
        self.feat_bn.bias.requires_grad_(False)
        if self.dropout > 0:
            self.drop = nn.Dropout(self.dropout)
        self.cls = nn.Linear(self.num_feat, self.num_classes, bias=False)

        #self.logsoft = nn.LogSoftmax()
        # FC layer + weight initialization
        self.feat_bn.apply(weights_init_kaiming)
        # Classifier + weight initialization
        self.cls.apply(weights_init_classifier)
        self.train

    def forward(self, inputs, training=True):
        global cls_feat
        im_input = inputs                                                    # input layer
        # print(im_input.shape)
        feat_ = self.backbone(im_input)                                         # RestNet-50 with RGA modules
        # print(feat_.shape)
        feat_ = F.avg_pool2d(feat_, feat_.size()[2:]).view(feat_.size(0), -1)   # average2d-pooling
        # print(feat_.shape)
        feat = self.feat_bn(feat_)                                              # Batch normalization
        # print(feat_.shape)
        if self.dropout > 0:                                                    # drop out procedure
            feat = self.drop(feat)
        if training and self.num_classes is not None:                           # classifier
            cls_feat = self.cls(feat)
        # print(cls_feat.shape)

        if training:
            return feat_, feat, cls_feat
        else:
            return feat_, feat



def resnet50_rga(*args, **kwargs):
    return ResNet50_RGA_Model(*args, **kwargs)
