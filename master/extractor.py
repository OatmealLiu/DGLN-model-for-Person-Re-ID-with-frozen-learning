# extractor.py
# this is the code used to extract the global and local features for each image

# -*- coding: utf-8 -*-
from __future__ import print_function, division

import argparse
import torch
import os
import sys
import scipy.io
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from model import ResNet50Model, DenseNet121Model
from ReIDmodel import ResNet50_RGA_Model
from torchvision import transforms
from prepare import MarketDataset

######################################################################
# Options
# --------

parser = argparse.ArgumentParser(description='ReID feature extraction')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--test_dir', default='../Market', type=str, help='./test_data')
parser.add_argument('--batchsize', default=256, type=int, help='batchsize')
parser.add_argument('--name_local', default='ResNet50', type=str, help='save model path')
parser.add_argument('--name_global', default='RGA_ResNet50', type=str, help='save model path')
parser.add_argument('--which_epoch_local', default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--which_epoch_global', default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--stride', default=2, type=int, help='stride')
parser.add_argument('--train_mode', default='ON', type=str, help='ON->extract training images feature, OFF-> extract test images feature')

opt = parser.parse_args()

str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        gpu_ids.append(id)

# set gpu ids
if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

use_gpu = torch.cuda.is_available()

######################################################################
# Load Data
# ---------
def create_test_dataloader(data_path, data_transforms, workers=0, batch_size=1):
    #data_transforms = transforms.Compose(test_transform)
    image_dataset = MarketDataset(image_dir=data_path,
                                  transform=data_transforms)
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=workers)
    dataset_size = len(image_dataset)
    # print("Total To-Be-Predicted dataset: {}".format(dataset_size))
    print("'{}' is ready, size = {}".format(data_path, dataset_size))
    return dataset_size, image_dataset, dataloader

######################################################################
# Load model
# ---------------------------
def load_network(model_name, num_epoch, stride, returnF=False):
    if model_name == 'DenseNet121':
        model_structure = DenseNet121Model(returnF=returnF)
    elif model_name == 'ResNet50':
        model_structure = ResNet50Model(stride=stride, returnF=returnF)
    elif model_name == 'RGA_ResNet50':
        model_structure = ResNet50_RGA_Model(pretrained=True,
                               num_feat=2048, height=256, width=128,
                               num_classes=751, dropout=0,
                               last_stride=1, scale=8, d_scale=8)
    else:
        print("ERROR---> please enter correct model name")

    if model_name == 'DenseNet121':
        save_path = os.path.join('./model', model_name, 'net_%s.pth' % num_epoch)
        model_structure.load_state_dict(torch.load(save_path))
    elif model_name == 'ResNet50':
        save_path = os.path.join('./model', model_name, 'net_%s.pth' % num_epoch)
        model_structure.load_state_dict(torch.load(save_path))
    elif model_name == 'RGA_ResNet50':
        saved_filename = 'net_%s.pth' % num_epoch
        saved_filename = model_name + '_' + saved_filename
        save_path = os.path.join('./model_id', model_name, saved_filename)
        model_structure.load_state_dict(torch.load(save_path))
    else:
        print("ERROR---> please enter correct model name")

    return model_structure


######################################################################
# Extract feature
# ----------------------
def extract_feature(model_local, model_global, dataloaders):
    local_feat_list = ['age', 'backpack', 'bag', 'handbag', 'clothes',
                       'down', 'up', 'hair', 'hat', 'gender',
                       'upcolor', 'downcolor']

    # concatenated feat container = local + global
    features = torch.FloatTensor()
    count = 0

    for data in dataloaders:
        _, img = data
        #print("batch type = {}".format(type(data)))
        n, c, h, w = img.size()
        count += n
        print(count)

        # local feat container
        # We prefer to use the probability distribution vector as local feat
        # rather than the 12 attributes vector
        # so the feature length = 4 + 2 * 9 + 9 + 10 = 41

        # global feat container
        # global feat is the last FC layer from personID prediction model = 2048

        # Then, after we combine local and global feature as all-feature
        # the dim of feat_all is n x (41+2048)
        feat_all = torch.FloatTensor(n, 41+2048).zero_().cuda()  # all feat

        input_img = Variable(img.cuda())

        # extract local feat without training
        feat_local_dict  = model_local(input_img)
        feat_all_global, feat_dropped_global, outputs_global = model_global(input_img)
        #print("Global feature shape = {}".format(feat_all_global.shape))
        # build local + global feature combination
        # load local feature
        ff_cursor = 0
        for key in local_feat_list:
            #print(feat_all.shape)
            local_attr = feat_local_dict[key]
            #print("local attributes shape = {}".format(local_attr.shape))
            cursorL = ff_cursor
            cursorR = ff_cursor + len(local_attr[0,:])
            #print("Range = [{} : {}]".format(cursorL, cursorR))
            #print(feat_all.shape)
            feat_all[:,cursorL:cursorR] += local_attr
            ff_cursor = cursorR

        # load global feature
        feat_all[:, ff_cursor:] += feat_all_global

        fnorm = torch.norm(feat_all, p=2, dim=1, keepdim=True)
        feat_all = feat_all.div(fnorm.expand_as(feat_all))
        # normalize the feature
        features = torch.cat((features, feat_all.data.cpu()), 0)
    print("Feature extraction completed, feature shape = {}".format(features.shape))
    return features


def get_label(img_path, train=False):
    labels = []
    img_list = os.listdir(img_path)

    if train:
        for img in img_list:
            label_id = int(img.split('_')[0].lstrip("0"))
            labels.append(label_id)
    else:
        for img in img_list:
            # print(img)
            label_id = int(img.split('.')[0])
            labels.append(label_id)
    print("Got '{}' labels, size = {}".format(img_path, len(labels)))
    # return as list of int data as label id
    return labels


def main(argv=None):
    # change the way of loading data and read the labels
    training_mode = opt.train_mode

    which_epoch_local = opt.which_epoch_local
    which_epoch_global = opt.which_epoch_global
    name_localNet = opt.name_local
    name_globalNet = opt.name_global
    test_dir = opt.test_dir

    # create data loader
    data_transforms = transforms.Compose([
        transforms.Resize((256, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_dir = test_dir
    if training_mode == 'ON':
        gallery_path = data_dir + "/gallery_train"
        query_path = data_dir + "/queries_train"
    else:
        gallery_path = data_dir + "/gallery"
        query_path = data_dir + "/queries"

    path_dict = {'gallery': gallery_path,
                 'query': query_path}
    dataset_sizes = {}
    image_datasets = {}
    dataloaders = {}
    for k in ['gallery', 'query']:
        datasize, img_set, loader = create_test_dataloader(data_path=path_dict[k],
                                                           data_transforms=data_transforms,
                                                           workers=8, batch_size=opt.batchsize)
        dataset_sizes[k] = datasize
        image_datasets[k] = img_set
        dataloaders[k] = loader

    print("Gallery dataset size {}\nQuery dataset size {}".format(dataset_sizes['gallery'],
                                                                  dataset_sizes['query']))
    print("Gallery total batches {}\nQuery total batches {}".format(len(dataloaders['gallery']),
                                                                    len(dataloaders['query'])))
    if training_mode == 'ON':
        gallery_label = get_label(gallery_path,train=True)
        query_label = get_label(query_path,train=True)
    else:
        gallery_label = get_label(gallery_path,train=False)
        query_label = get_label(query_path,train=False)

    model_local = load_network(model_name=name_localNet, num_epoch=which_epoch_local,
                               stride=opt.stride, returnF=False)
    model_global = load_network(model_name=name_globalNet, num_epoch=which_epoch_global,
                               stride=opt.stride, returnF=False)

    print('-------test-----------')
    # Change to test mode
    model_local = model_local.eval()
    if use_gpu:
        model_local = model_local.cuda()

    model_global = model_global.eval()
    if use_gpu:
        model_global = model_global.cuda()

    # Extract feature
    with torch.no_grad():
        print("-"*10+"Gallery feature extraction"+"-"*10)
        gallery_feature = extract_feature(model_local=model_local,
                                          model_global=model_global,
                                          dataloaders=dataloaders['gallery'])

        query_feature = extract_feature(model_local=model_local,
                                          model_global=model_global,
                                          dataloaders=dataloaders['query'])
        print()
    # Save to Matlab for check
    result = {'gallery_f': gallery_feature.numpy(), 'gallery_label': gallery_label,
              'query_f': query_feature.numpy(),     'query_label': query_label}
    for k, v in result.items():
        print("{} shape = {}".format(k, len(v)))

    if training_mode == 'ON':
        scipy.io.savemat('./features/reID_training_result.mat', result)
    else:
        scipy.io.savemat('./features/reID_prediction_result.mat', result)

    #print(opt.name)
    result = './features/result.txt'
    if training_mode == 'ON':
        result = './features/result_training.txt'
    else:
        result = './features/result_prediction.txt'
    os.system('extractor.py | tee -a %s' % result)

if __name__ == '__main__':
    sys.exit(main())
