# -*- coding: utf-8 -*-
from __future__ import print_function, division
import numpy as np
import pandas as pd
import argparse
import torch
from torch.autograd import Variable
from torchvision import transforms
import matplotlib
matplotlib.use('agg')
import os
import sys
from model import ResNet50Model, DenseNet121Model
from prepare import MarketDataset

version = torch.__version__

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name', default='ResNet50', type=str, help='output model name')
parser.add_argument('--data_dir', default='../Market', type=str, help='training dir path')
parser.add_argument('--stride', default=2, type=int, help='stride')
parser.add_argument('--which_epoch',default='39', type=str, help='0,1,2,3...or last')
opt = parser.parse_args()


str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

use_gpu = torch.cuda.is_available()

# Load Data
def create_test_dataloader(test_path, test_transform, workers=8, batch_size=1):
    data_transforms = transforms.Compose(test_transform)
    image_dataset = MarketDataset(image_dir=test_path,
                                  transform=data_transforms)
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=workers, pin_memory=True)
    dataset_size = len(image_dataset)
    print("Total To-Be-Predicted dataset: {}".format(dataset_size))

    return dataset_size, dataloader


def load_network(name,num_epoch,stride,returnF=False):
    if name == 'DenseNet121':
        model_structure = DenseNet121Model(returnF=returnF)
    elif name == 'ResNet50':
        model_structure = ResNet50Model(stride=stride, returnF=returnF)
    else:
        model_structure = ResNet50Model(stride=stride, returnF=returnF)

    save_path = os.path.join('./model',name,'net_%s.pth'%num_epoch)
    model_structure.load_state_dict(torch.load(save_path))
    return model_structure

def make_prediction(model, dataloader, data_size, label_list, label_finder,class_list):
    len_head = len(label_list)
    result_arr = np.ones((data_size, len_head), dtype='long')

    i_bs = 0
    for _, data in dataloader:
        # print("(1):{}".format(_))
        # print("(2):{}".format(data.shape))
        model.train(False)
        # get input tensor: BS 256x128
        inputs = data

        if use_gpu:
            inputs = Variable((inputs).cuda().detach())
        else:
            inputs = Variable(inputs)

        # make prediction
        with torch.no_grad():
            outputs = model(inputs)

        # save prediction result for 12 attributes
        pred_dict = {}
        for k in class_list:
            _, preds = torch.max(outputs[k].data, 1)
            pred_dict[k] = preds
        #print(pred_dict)
        result_arr[i_bs, 0] = i_bs
        for k in class_list:
            if k == 'upcolor':
                # 'upblack' ~ 'upmulticolor':11~19
                # preds: 0~8
                result_arr[i_bs, label_finder['upblack']+pred_dict[k]] =  2
            elif k == 'downcolor':
                # 'downblack' ~ 'downmulticolor':20~29
                # preds: 0~9
                result_arr[i_bs, label_finder['downblack']+pred_dict[k]] =  2
            else:
                result_arr[i_bs, label_finder[k]] = 1 + pred_dict[k]

        i_bs += 1
        print("Test classification prediction complete [ {} / {} ]".format(i_bs, data_size))

    return result_arr

def write_prediction(preds_arr, file_name, head):
    index_name = head[0]
    index_list = []
    dummy_head = 100000000

    for row in preds_arr:
        img_id = dummy_head + row[0]
        img_id = str(img_id)[-6:] + ".jpg"
        #print(img_id)
        #print(type(img_id))
        # print(img_id)
        index_list.append(str(img_id))

    result_df = pd.DataFrame(preds_arr[:,1:],index=index_list, columns=head[1:])
    result_df.index.name=index_name

    result_name = file_name
    result_df.to_csv(result_name)
    return result_df

def main(argv=None):
    data_dir = opt.data_dir
    test_path = data_dir + "/test"

    label_list = ['id', 'age', 'backpack', 'bag', 'handbag', 'clothes',
                  'down', 'up', 'hair', 'hat', 'gender',
                  'upblack', 'upwhite', 'upred', 'uppurple', 'upyellow',
                  'upgray', 'upblue', 'upgreen', 'upmulticolor', 'downblack',
                  'downwhite', 'downpink', 'downpurple', 'downyellow', 'downgray',
                  'downblue', 'downgreen', 'downbrown', 'downmulticolor']

    class_list = ['age', 'backpack', 'bag', 'handbag', 'clothes',
                  'down', 'up', 'hair', 'hat', 'gender',
                  'upcolor', 'downcolor']

    label_finder = {label_list[i]: i for i in range(len(label_list))}


    transform_test_list = [
        transforms.Resize(size=(256, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    # create test dataloader
    dataset_size, test_dataloader = create_test_dataloader(test_path=test_path,
                                             test_transform=transform_test_list,
                                             workers=8, batch_size=1)
    # print("-->{},{}".format(type(dataset_size),type(test_dataloader)))
    # load saved model and weights
    print("Use the trained weights of the {}th epoch".format(opt.which_epoch))
    model = load_network(name=opt.name, num_epoch=opt.which_epoch, stride=opt.stride, returnF=False)
    # Send model to GPU
    model = model.cuda()

    preds_arr = make_prediction(model=model, dataloader=test_dataloader, data_size=dataset_size,
                                label_list=label_list, label_finder=label_finder, class_list=class_list)

    preds_file_name = 'classification_test.csv'
    preds_df = write_prediction(preds_arr=preds_arr, file_name=preds_file_name, head=label_list)
    print(preds_df.head())


if __name__=='__main__':
    sys.exit(main())
