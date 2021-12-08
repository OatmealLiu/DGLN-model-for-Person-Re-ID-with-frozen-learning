# -*- coding: utf-8 -*-

from __future__ import print_function, division

import god

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import time
import os
import yaml
from shutil import copyfile
import numpy as np
from prepare import MarketDataset
from model import ResNet50Model, DenseNet121Model


######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name', default='ResNet50', type=str, help='output model name')
parser.add_argument('--data_dir', default='../Market', type=str, help='training dir path')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--stride', default=2, type=int, help='stride')
parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate. for pretrained params, use 0.01')
parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
parser.add_argument('--num_epochs', default=60, type=int, help='the number of epochs')

opt = parser.parse_args()

data_dir = opt.data_dir
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >= 0:
        gpu_ids.append(gid)

# set gpu ids
if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

name = opt.name
label_name = 'age'
nclasses = 4

######################################################################
# Load Data
# ---------
#
transform_train_list = [
    transforms.Resize((256, 128), interpolation=3),
    transforms.Pad(10),
    #transforms.RandomCrop((256, 128)),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

transform_AUGtrain_list = [
    transforms.Resize((256, 128), interpolation=3),
    transforms.Pad(10),
    #transforms.RandomCrop((256, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    #transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)
]

transform_val_list = [
    transforms.Resize(size=(256, 128), interpolation=3),  # Image.BICUBIC
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

data_transforms = {
    'train': transforms.Compose(transform_train_list),
    'AUGtrain': transforms.Compose(transform_AUGtrain_list),
    'val': transforms.Compose(transform_val_list),
}

train_path = data_dir + "/train"
val_path = data_dir + "/val"
annotation_path = data_dir + "/annotations_train.csv"

image_datasets = {'train' : torch.utils.data.ConcatDataset(
                                                [MarketDataset(image_dir=train_path,
                                                              csv_file=annotation_path,
                                                              transform=data_transforms['train']),
                                                MarketDataset(image_dir=train_path,
                                                              csv_file=annotation_path,
                                                              transform=data_transforms['AUGtrain'])]),
                  'val' : MarketDataset(image_dir=val_path,
                                          csv_file=annotation_path,
                                          transform=data_transforms['val'])
                  }



dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                              shuffle=True, num_workers=0, pin_memory=True)
               for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

print("Train dataset size {}\nValidation dataset size {}".format(dataset_sizes['train'], dataset_sizes['val']))

use_gpu = torch.cuda.is_available()

since = time.time()
inputs, classes, _ = next(iter(dataloaders['train']))
print(time.time() - since)

######################################################################
# Training the model
# ------------------
label_list = ['age', 'backpack', 'bag', 'handbag', 'clothes',
              'down', 'up', 'hair', 'hat', 'gender',
              'upblack', 'upwhite', 'upred', 'uppurple', 'upyellow',
              'upgray', 'upblue', 'upgreen', 'upmulticolor', 'downblack',
              'downwhite', 'downpink', 'downpurple', 'downyellow', 'downgray',
              'downblue', 'downgreen', 'downbrown', 'downmulticolor']

class_list = ['age', 'backpack', 'bag', 'handbag', 'clothes',
              'down', 'up', 'hair', 'hat', 'gender',
              'upcolor', 'downcolor']

y_loss_dict = {class_list[i]: {'train': [], 'val': []}
               for i in range(len(class_list))}
y_err_dict = {class_list[i]: {'train': [], 'val': []}
               for i in range(len(class_list))}

label_finder = {label_list[i]: i+1 for i in range(len(label_list))}
class_finder = {class_list[i]: i+1 for i in range(len(class_list))}

def train_model(model, criterion_group, optimizer_group, scheduler_group, num_epochs=25):
    since = time.time()

    # best_model_wts = model.state_dict()
    # best_acc = 0.0
    warm_up = 0.1  # We start from the 0.1*lrRate
    warm_iteration = round(dataset_sizes['train'] / opt.batchsize) * opt.warm_epoch  # first 5 epoch

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                for scheduler in scheduler_group:
                    scheduler.step()
                model.train(True)   # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

                # statistics
            running_loss_dict = {}
            running_corr_dict = {}
            for k in class_list:
                running_loss_dict[k] = 0.
                running_corr_dict[k] = 0.

            for data in dataloaders[phase]:
                #print("++++++Im training")
                # get the inputs
                inputs, labels, _ = data
                inputs = Variable(inputs.cuda().detach())
                label_dict = {}
                for k in class_list:
                    if k == 'upcolor':
                        a = label_finder['upblack']
                        b = label_finder['upmulticolor'] + 1
                        label_range = torch.squeeze(labels)[:, a:b]
                        label_arr = np.ones((label_range.shape[0]),dtype='long')
                        for i in range(len(label_arr)):
                            label_arr[i] = label_range[i, :].argmax()
                        label_tensor = torch.tensor(label_arr, dtype=torch.long)
                        label_dict[k] = Variable((label_tensor).cuda().detach())

                    elif k == 'downcolor':
                        c = label_finder['downblack']
                        d = label_finder['downmulticolor'] + 1
                        label_range = torch.squeeze(labels)[:, c:d]
                        label_arr = np.ones((label_range.shape[0]),dtype='long')
                        for i in range(len(label_arr)):
                            label_arr[i] = label_range[i, :].argmax()
                        label_tensor = torch.tensor(label_arr, dtype=torch.long)
                        label_dict[k] = Variable((label_tensor).cuda().detach())
                    else:
                        label_dict[k] = Variable((torch.squeeze(labels)[:,label_finder[k]] - 1).cuda().detach())

                now_batch_size, c, h, w = inputs.shape
                if now_batch_size < opt.batchsize:  # skip the last batch
                    continue

                # zero the parameter gradients
                for optiz in optimizer_group:
                    optiz.zero_grad()

                # forward
                if phase == 'val':
                    with torch.no_grad():
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)

                pred_dict = {}
                loss_dict = {}
                for i in range(len(class_list)):
                    _, preds = torch.max(outputs[class_list[i]].data, 1)
                    single_output = outputs[class_list[i]]
                    single_label = label_dict[class_list[i]]
                    single_criterion = criterion_group[i]
                    loss = single_criterion(single_output,single_label)
                    pred_dict[class_list[i]] = preds
                    loss_dict[class_list[i]] = loss

                avg_loss = 0.
                for k in loss_dict:
                    avg_loss += loss_dict[k]
                loss_dict['backbone'] = avg_loss / len(class_list)

                # backward + optimize only if in training phase
                if epoch < opt.warm_epoch and phase == 'train':
                    warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                    for k in loss_dict:
                        loss_dict[k] = loss_dict[k] * warm_up

                if phase == 'train':
                    i = 0
                    for k in loss_dict:
                        if k == 'backbone':
                            #print(k)
                            loss_dict[k].backward(retain_graph=True)
                        else:
                            #print(k)
                            loss_dict[k].backward(retain_graph=True)
                        i += 1

                    i = 0
                    for k in loss_dict:
                        if k == 'backbone':
                            optimizer_group[0].step()
                        else:
                            optimizer_group[i].step()
                        i += 1

                # statistics
                for k in running_loss_dict:
                    running_loss_dict[k] += loss_dict[k].item() * now_batch_size
                for k in running_loss_dict:
                    running_corr_dict[k] += float(torch.sum(pred_dict[k] == label_dict[k].data))

            epoch_loss_dict = {}
            epoch_acc_dict = {}
            for k in class_list:
                epoch_loss_dict[k] = running_loss_dict[k] / dataset_sizes[phase]
                epoch_acc_dict[k] = running_corr_dict[k] / dataset_sizes[phase]

            for k in class_list:
                print('Epoch [{}/{}]: {}-->{} Loss: {:.4f} Acc: {:.4f}%'.format(
                    epoch + 1, num_epochs,
                    phase, k, epoch_loss_dict[k], 100*epoch_acc_dict[k]))

                y_loss_dict[k][phase].append(epoch_loss_dict[k])
                y_err_dict[k][phase].append(1.0 - epoch_acc_dict[k])

            # deep copy the model
            if phase == 'val':
                last_model_wts = model.state_dict()
                if epoch % 10 == 9:
                    save_network(model, epoch)
                draw_curve(epoch)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(last_model_wts)
    save_network(model, 'last')
    return model


######################################################################
# Draw Curve
# ---------------------------
x_epoch = []
def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    for k in class_list:
        fig = plt.figure()
        ax0 = fig.add_subplot(121, title="loss--"+k)
        ax1 = fig.add_subplot(122, title="top1err--"+k)

        ax0.plot(x_epoch, y_loss_dict[k]['train'], 'bo-', label='train-->'+k)
        ax0.plot(x_epoch, y_loss_dict[k]['val'], 'ro-', label='val-->'+k)
        ax1.plot(x_epoch, y_err_dict[k]['train'], 'bo-', label='train-->'+k)
        ax1.plot(x_epoch, y_err_dict[k]['val'], 'ro-', label='val-->'+k)
        if current_epoch == 0:
            ax0.legend()
            ax1.legend()
        image_name = 'train_'+k+'.jpg'
        fig.savefig(os.path.join('./model', name, image_name))

######################################################################
# Save model
# ---------------------------
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join('./model', name, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda(gpu_ids[0])


######################################################################
# Finetuning the convnet
# ----------------------
if name == 'DenseNet121':
    model = DenseNet121Model(opt.droprate, returnF=False)
elif name == 'ResNet50':
    model = ResNet50Model(opt.droprate, opt.stride, returnF=False)
else:
    model = ResNet50Model(opt.droprate, opt.stride, returnF=False)

print("++++++ Model Name: [{}] ++++++".format(name))
#print(model)

ignored_params = []
ignored_params = [
    map(id, model.classifier_age.parameters()),
    map(id, model.classifier_backpack.parameters()),
    map(id, model.classifier_bag.parameters()),
    map(id, model.classifier_handbag.parameters()),
    map(id, model.classifier_clothes.parameters()),
    map(id, model.classifier_down.parameters()),
    map(id, model.classifier_up.parameters()),
    map(id, model.classifier_hair.parameters()),
    map(id, model.classifier_hat.parameters()),
    map(id, model.classifier_gender.parameters()),
    map(id, model.classifier_upcolor.parameters()),
    map(id, model.classifier_downcolor.parameters())
]

base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

# Define all components respectivelly
param_list = [
    model.classifier_age.parameters(),
    model.classifier_backpack.parameters(),
    model.classifier_bag.parameters(),
    model.classifier_handbag.parameters(),
    model.classifier_clothes.parameters(),
    model.classifier_down.parameters(),
    model.classifier_up.parameters(),
    model.classifier_hair.parameters(),
    model.classifier_hat.parameters(),
    model.classifier_gender.parameters(),
    model.classifier_upcolor.parameters(),
    model.classifier_downcolor.parameters()
]

parameter_group = []
optimizer_group = []
lr_scheduler_group = []
criterion_group = []
for i in range(len(class_list) + 1):
    if i == 0:
        param = [{'params': base_params, 'lr': 0.1 * opt.lr}]
        optimizer_base = optim.SGD(param, weight_decay=5e-4, momentum=0.9, nesterov=True)
        lr_manager = lr_scheduler.StepLR(optimizer_base, step_size=40, gamma=0.1)
        #print('pass')
        parameter_group.append(param)
        optimizer_group.append(optimizer_base)
        lr_scheduler_group.append(lr_manager)
    else:
        param = [{'params': param_list[i-1], 'lr': opt.lr}]
        optimizer_dyn = optim.SGD(param, weight_decay=5e-4, momentum=0.9, nesterov=True)
        lr_manager = lr_scheduler.StepLR(optimizer_dyn, step_size=40, gamma=0.1)

        parameter_group.append(param)
        optimizer_group.append(optimizer_dyn)
        lr_scheduler_group.append(lr_manager)
        criterion_group.append(nn.CrossEntropyLoss())


######################################################################
# Train and evaluate
# ----------------------
dir_name = os.path.join('./model', name)
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)
# record every run
copyfile('train_attributes.py', dir_name + '/train_attributes.py')
copyfile('./model.py', dir_name + '/model.py')

# save opts
with open('%s/opts.yaml' % dir_name, 'w') as fp:
    yaml.dump(vars(opt), fp, default_flow_style=False)

# model to gpu
model = model.cuda()
model = train_model(model=model, criterion_group=criterion_group, optimizer_group=optimizer_group,
                    scheduler_group=lr_scheduler_group, num_epochs=opt.num_epochs)
















