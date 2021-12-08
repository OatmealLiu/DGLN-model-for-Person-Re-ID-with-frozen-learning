# visualization.py
# in this code, we will basically do the same work as in evaluator.py
# moreover, we will visualize our re-identification result here
# in order to validate our algorithm

import argparse
import scipy.io
import torch
import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from prepare import MarketDataset

#######################################################################
# Evaluate
# ----------------------
parser = argparse.ArgumentParser(description='Demo')
parser.add_argument('--query_index', default=2, type=int, help='test_image_index')
parser.add_argument('--test_dir',default='../Market',type=str, help='./test_data')
opt = parser.parse_args()

def create_demo_dataset(data_path, data_transforms):
    #data_transforms = transforms.Compose(test_transform)
    image_dataset = MarketDataset(image_dir=data_path,
                                  transform=data_transforms)

    dataset_size = len(image_dataset)
    # print("Total To-Be-Predicted dataset: {}".format(dataset_size))
    print("'{}' is ready, size = {}".format(data_path, dataset_size))
    return dataset_size, image_dataset

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
    return labels

#Show result
def imshow(path, title=None):
    """Imshow for Tensor."""
    im = plt.imread(path)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    plt.pause(0.1)  # pause a bit so that plots are updated


def sort_query_results(query_feat, gallery_feat):
    query_ff = query_feat.view(-1, 1)
    print("-"*30)
    print("Pred-->[{}]".format(query_label))
    # print(query.shape)
    # compute cosine distance
    # print("gallery_feat shape = {}".format(gallery_feat.shape))
    # print("query_ff shape = {}".format(query_ff.shape))
    # gallery_feat[:,42:] *= 0
    # query_ff[42:,:] *= 0
    # gallery_feat[:,0:42] *= 0
    # query_ff[0:42,:] *= 0

    score = torch.mm(gallery_feat, query_ff)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]        # reverse the order
    top10_result = [score[idx] for idx in index[0:10]]
    print("top 10:{}".format(top10_result))

    print("-"*30)
    # return top 10 index
    return list(index[:10])


if __name__ == "__main__":
    # change the way of loading data and read the labels
    test_dir = opt.test_dir
    data_dir = test_dir

    # create data loader
    # data_transforms = transforms.Compose([
    #     transforms.Resize((256, 128), interpolation=3),
    #     transforms.ToTensor(),
    # ])

    gallery_path = data_dir + "/gallery_train"
    query_path = data_dir + "/queries_train"


    path_dict = {'gallery': gallery_path,
                 'query': query_path}
    dataset_sizes = {}
    image_datasets = {}
    for k in ['gallery', 'query']:
        datasize, img_set = create_demo_dataset(data_path=path_dict[k], data_transforms=None)
        dataset_sizes[k] = datasize
        image_datasets[k] = img_set

    print("Gallery dataset size {}\nQuery dataset size {}".format(dataset_sizes['gallery'],
                                                                  dataset_sizes['query']))

    # gallery_label = get_label(gallery_path, train=True)
    # query_label = get_label(query_path, train=True)

    result = scipy.io.loadmat('./features/reID_training_result.mat')

    query_feature = torch.FloatTensor(result['query_f'])
    query_label = result['query_label'][0]
    print("query label len = {}".format(len(query_label)))

    gallery_feature = torch.FloatTensor(result['gallery_f'])
    gallery_label = result['gallery_label'][0]
    print("gallery label len = {}".format(len(gallery_label)))

    query_feature = query_feature.cuda()
    gallery_feature = gallery_feature.cuda()

    i = opt.query_index
    top10_index = sort_query_results(query_feature[i], gallery_feature)

    # Visualize the rank result
    query_path, _ = image_datasets['query'][i]
    print("What return: {}, {}".format(query_path,_))
    query_label = query_label[i]
    print(query_path)
    print('Top 10 images are as follow:')
    try:
        fig = plt.figure(figsize=(16, 4))
        ax = plt.subplot(1, 11, 1)
        ax.axis('off')
        imshow(query_path, 'query')
        for i in range(len(top10_index)):
            ax = plt.subplot(1, 11, i + 2)
            ax.axis('off')
            img_path, _ = image_datasets['gallery'][top10_index[i]]
            label = gallery_label[top10_index[i]]
            imshow(img_path)
            if label == query_label:
                ax.set_title('%d' % (i + 1), color='green')
            else:
                ax.set_title('%d' % (i + 1), color='red')
            print(img_path)

    except RuntimeError:
        print('RuntimeError')

    fig.savefig("show.png")