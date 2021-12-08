import os
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from backbone import RGA_ResNet50_alpha
from torch.nn import functional as F


WEIGHT_PATH = os.path.join(os.path.dirname(__file__), '../') + '/weights/pre_train/resnet50-19c8e357.pth'

# Define kaiming-initialization
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

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

# Define the ResNet50-based Model
class ResNet50Model(nn.Module):

    def __init__(self, droprate=0.5, stride=2, returnF=False):
        super(ResNet50Model, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # we want to change avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.returnF = returnF

        self.classifier_age       = ClassBlock(2048, 4, droprate, returnF = returnF)
        self.classifier_backpack  = ClassBlock(2048, 2, droprate, returnF = returnF)
        self.classifier_bag       = ClassBlock(2048, 2, droprate, returnF = returnF)
        self.classifier_handbag   = ClassBlock(2048, 2, droprate, returnF = returnF)
        self.classifier_clothes   = ClassBlock(2048, 2, droprate, returnF = returnF)
        self.classifier_down      = ClassBlock(2048, 2, droprate, returnF = returnF)
        self.classifier_up        = ClassBlock(2048, 2, droprate, returnF = returnF)
        self.classifier_hair      = ClassBlock(2048, 2, droprate, returnF = returnF)
        self.classifier_hat       = ClassBlock(2048, 2, droprate, returnF = returnF)
        self.classifier_gender    = ClassBlock(2048, 2, droprate, returnF = returnF)
        self.classifier_upcolor   = ClassBlock(2048, 9, droprate, returnF = returnF)
        self.classifier_downcolor = ClassBlock(2048, 10, droprate, returnF = returnF)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x_age       = self.classifier_age(x)
        x_backpack  = self.classifier_backpack(x)
        x_bag       = self.classifier_bag(x)
        x_handbag   = self.classifier_handbag(x)
        x_clothes   = self.classifier_clothes(x)
        x_down      = self.classifier_down(x)
        x_up        = self.classifier_up(x)
        x_hair      = self.classifier_hair(x)
        x_hat       = self.classifier_hat(x)
        x_gender    = self.classifier_gender(x)
        x_upcolor   = self.classifier_upcolor(x)
        x_downcolor = self.classifier_downcolor(x)

        x_dict = {'age': x_age,
                  'backpack':x_backpack,
                  'bag':x_bag,
                  'handbag':x_handbag,
                  'clothes':x_clothes,
                  'down':x_down,
                  'up':x_up,
                  'hair':x_hair,
                  'hat':x_hat,
                  'gender':x_gender,
                  'upcolor':x_upcolor,
                  'downcolor':x_downcolor}

        return x_dict

# Define the DenseNet121-based Model
class DenseNet121Model(nn.Module):

    def __init__(self, droprate=0.5, returnF=False):
        super().__init__()
        model_ft = models.densenet121(pretrained=True)
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.fc = nn.Sequential()
        self.model = model_ft
        self.returnF = returnF
        self.classifier_age       = ClassBlock(1024, 4, droprate, returnF = returnF)
        self.classifier_backpack  = ClassBlock(1024, 2, droprate, returnF = returnF)
        self.classifier_bag       = ClassBlock(1024, 2, droprate, returnF = returnF)
        self.classifier_handbag   = ClassBlock(1024, 2, droprate, returnF = returnF)
        self.classifier_clothes   = ClassBlock(1024, 2, droprate, returnF = returnF)
        self.classifier_down      = ClassBlock(1024, 2, droprate, returnF = returnF)
        self.classifier_up        = ClassBlock(1024, 2, droprate, returnF = returnF)
        self.classifier_hair      = ClassBlock(1024, 2, droprate, returnF = returnF)
        self.classifier_hat       = ClassBlock(1024, 2, droprate, returnF = returnF)
        self.classifier_gender    = ClassBlock(1024, 2, droprate, returnF = returnF)
        self.classifier_upcolor   = ClassBlock(1024, 9, droprate, returnF = returnF)
        self.classifier_downcolor = ClassBlock(1024, 10, droprate, returnF = returnF)

    def forward(self, x):
        x = self.model.features(x)
        x = x.view(x.size(0), x.size(1))
        x_age       = self.classifier_age(x)
        x_backpack  = self.classifier_backpack(x)
        x_bag       = self.classifier_bag(x)
        x_handbag   = self.classifier_handbag(x)
        x_clothes   = self.classifier_clothes(x)
        x_down      = self.classifier_down(x)
        x_up        = self.classifier_up(x)
        x_hair      = self.classifier_hair(x)
        x_hat       = self.classifier_hat(x)
        x_gender    = self.classifier_gender(x)
        x_upcolor   = self.classifier_upcolor(x)
        x_downcolor = self.classifier_downcolor(x)

        x_dict = {'age': x_age,
                  'backpack':x_backpack,
                  'bag':x_bag,
                  'handbag':x_handbag,
                  'clothes':x_clothes,
                  'down':x_down,
                  'up':x_up,
                  'hair':x_hair,
                  'hat':x_hat,
                  'gender':x_gender,
                  'upcolor':x_upcolor,
                  'downcolor':x_downcolor}

        return x_dict

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
        im_input = inputs                                                       # input layer
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



if __name__ == '__main__':
# Here I left a simple forward function.
# Test the model, before you train it.
    net = ResNet50Model()
    print(net)
    input = Variable(torch.FloatTensor(8, 3, 256, 128))
    output = net(input)
    print('net output size:')
    print(output['upcolor'].shape)
    print(net.parameters())















