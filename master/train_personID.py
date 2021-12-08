# -*- coding: utf-8 -*-
from __future__ import print_function, division

import god
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time
import os
import pandas as pd
from ReIDmodel import ResNet50_RGA_Model
from lr_scheduler import LRScheduler
import yaml
from shutil import copyfile
from prepare import MarketDataset

version = torch.__version__
######################################
# Optional Configurations
# ----------------------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--data_dir', default='../Market', type=str, help='training dir path')
parser.add_argument('--batchsize', default=16, type=int, help='batchsize')
parser.add_argument('--stride', default=2, type=int, help='stride')
parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
parser.add_argument('--optimizer', default='SGD', type=str, help='type of optimizer')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate. for pretrained params, use 0.01')
parser.add_argument('--mom', default=0.9, type=float, help='momentum')
parser.add_argument('--wd', default=0.0005, type=float, help='weight decay')
parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
parser.add_argument('--num_epochs', default=60, type=int, help='the number of epochs')
opt = parser.parse_args()

data_dir = opt.data_dir
name = "RGA_ResNet50"
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >= 0:
        gpu_ids.append(gid)

######################################
# Load Data
# ----------------------
train_path = data_dir + "/train_id"
val_path = data_dir + "/val_id"
annotation_path = data_dir + "/annotations_train.csv"

def create_id_book(csv_path):
    annotation_df = pd.read_csv(csv_path)
    #print(len(annotation_df.id))
    #print(annotation_df.head())
    person_id_codedict = {}
    id_label = 0
    for raw_id in annotation_df.id:
        person_id_codedict[raw_id] = id_label
        id_label += 1
    #print("Total id classes = {}".format(id_label))
    return person_id_codedict

# train dataset loading with data augumentation
transform_train_list = [
    transforms.Resize((256, 128), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

transform_AUGtrain_list = [
    transforms.Resize((256, 128), interpolation=3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
]

transform_val_list = [
    transforms.Resize(size=(256, 128), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

data_transforms = {
    'train': transforms.Compose(transform_train_list),
    'AUGtrain': transforms.Compose(transform_AUGtrain_list),
    'val': transforms.Compose(transform_val_list),
}

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
print("Train total batches {}\nValidation total batches {}".format(len(dataloaders['train']),
                                                                     len(dataloaders['val'])))

######################################

# ----------------------
use_gpu = torch.cuda.is_available()

since = time.time()
inputs, classes,_ = next(iter(dataloaders['train']))
print(time.time() - since)
######################################

######################################
# Training the model
# create conatiners for training records
# ----------------------
y_loss = {'train': [], 'val': []}
y_err = {'train': [], 'val': []}

def train_model(model, criterion, optimizer, scheduler,ID_codedict,num_epochs=60):
    since = time.time()
    #warm_up = 0.01
    #warm_iteration = round(dataset_sizes['train'] / opt.batchsize) * scheduler.warmup_epoch

    for epoch in range(num_epochs):
        print('-' * 20)
        print('Epoch [{}/{}]'.format(1+epoch, num_epochs))
        print('-' * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #scheduler.step()
                lr = scheduler.update(epoch)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                model.train(True)           # Set model to training mode
            else:
                model.train(False)          # Set model to evaluate mode

            running_loss = 0.
            running_corrects = 0.
            # perform smart train & val over batches
            #print("Training data--->{}".format(len(dataloaders['train'])))
            #print("Val data--->{}".format(len(dataloaders['val'])))

            for data in dataloaders[phase]:
                # get input tensor: BS 256x128
                inputs, labels, _ = data
                labels = torch.squeeze(labels)[:, 0]
                # print("[{}] OG label:\n{}".format(phase,labels))
                for i in range(len(labels)):
                    labels[i] = ID_codedict[int(labels[i])]
                # print("[{}] Encoded label:\n{}".format(phase,labels))

                now_batch_size, c, h, w = inputs.shape

                if now_batch_size < opt.batchsize:  # skip the last batch
                    continue
                # print(inputs.shape)
                if use_gpu:
                    inputs = Variable(inputs.cuda().detach())
                    labels = Variable(labels.cuda().detach())
                else:
                    inputs = Variable(inputs)
                    labels = Variable(labels)

                # Erase gradients over all nodes
                optimizer.zero_grad()

                # Do forward-prop
                if phase == 'val':
                    # if validation, we do not compute gradients
                    with torch.no_grad():
                        feat_all, feat_dropped, outputs = model(inputs, training=True)
                else:
                    # if trainning, we compute gradients for each node
                    feat_all, feat_dropped, outputs = model(inputs, training=True)
                # print("output shape = {}".format(outputs.shape))
                # print("label shape = {}".format(labels.shape))
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()         # back-prop
                    optimizer.step()        # update weights

                # record statistics
                running_loss += loss.item() * now_batch_size
                running_corrects += float(torch.sum(preds == labels.data))
                # if phase == 'val':
                #     print(running_corrects)


            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            # print(epoch_acc)
            print('Epoch [{}/{}] Phase [{}]: Loss: {:.4f} Acc: {:.4f}%'.format(
                                                1+epoch, num_epochs,
                                                phase, epoch_loss, 100*epoch_acc))

            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0 - epoch_acc)
            # deep copy the model
            if phase == 'val':
                last_model_wts = model.state_dict()
                if epoch % 10 == 9:
                    save_network(model, epoch)
                draw_curve(epoch)

        time_elapsed = time.time() - since
        print('Epoch [{}/{}] Phase [{}] Training complete in {:.0f}m {:.0f}s'.format(
                                                1 + epoch, num_epochs, phase,
                                                time_elapsed // 60, time_elapsed % 60))
        print()

    time_elapsed = time.time() - since
    print('[End]Training complete in {:.0f}m {:.0f}s'.format(
                                                time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(last_model_wts)
    save_network(model, 'last')
    return model

######################################
# Draw Curve
# ----------------------
x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="ID-loss")
ax1 = fig.add_subplot(122, title="ID-top1err")


def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig(os.path.join('./model_id', name, 'train.jpg'))


######################################
# Define function to save model
# ----------------------
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth' % epoch_label
    save_filename = name + '_' + save_filename
    save_path = os.path.join('./model_id', name, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda(gpu_ids[0])


######################################
# Train and Evaluation
# ----------------------
personID_codedict = create_id_book(csv_path=annotation_path)

model = ResNet50_RGA_Model(pretrained=True,
                           num_feat=2048, height=256, width=128,
                           num_classes=len(personID_codedict), dropout=opt.droprate,
                           last_stride=1, scale=8, d_scale=8)
#print(model)

# Define loss function
criterion = nn.CrossEntropyLoss()

param_groups = model.parameters()
if opt.optimizer == 'Adam':
    #   Option 1: Adam
    optimizer_ft = optim.Adam(param_groups, lr=opt.lr,weight_decay=opt.wd)
else:
    #   Option 2: SGD
    optimizer_ft = optim.SGD(param_groups, lr=opt.lr,weight_decay=opt.wd, momentum=opt.mom, nesterov=True)

lr_scheduler = LRScheduler(base_lr=0.0008, step=[80, 120, 160, 200, 240, 280, 320, 360],
                           factor=0.5, warmup_epoch=20,
                           warmup_begin_lr=0.000008)

# organize record folders
dir_name = os.path.join('./model_id', name)
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)

# Record every run
copyfile('./train_personID.py', dir_name + '/train_personID.py')
copyfile('./ReIDmodel.py', dir_name + '/ReIDmodel.py')

# Save optional configurations
with open('%s/opts.yaml' % dir_name, 'w') as fp:
    yaml.dump(vars(opt), fp, default_flow_style=False)

# Send model to GPU
model = model.cuda()

# Train model
model = train_model(model=model, criterion=criterion, optimizer=optimizer_ft, scheduler=lr_scheduler,
                    ID_codedict=personID_codedict,num_epochs=opt.num_epochs)
