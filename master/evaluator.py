# evaluator.py
# this is the code used to match the query from the gallery for training queries
# in this code, we should compute the mAP value for training queries

import scipy.io
import torch
import numpy as np
import argparse
from map_evaluator import Evaluator

class God:
    @staticmethod
    def godBless():
        """
                               _ooOoo_
                              o8888888o
                              88" . "88
                              (| -_- |)
                               O\ = /O
                            ____/`---'\____
                         .   ' \\| |// `.
                          / \\||| : |||// \
                         / _||||| -:- |||||- \
                          | | \\\ - /// | |
                        | \_| ''\---/'' | |
                         \ .-\__ `-` ___/-. /
                     ___`. .' /--.--\ `. . __
                   ."" '< `.___\_<|>_/___.' >'"".
                  | | : `- \`.;`\ _ /`;.`/ - ` : | |
                     \ \ `-. \_ __\ /__ _/ .-` / /
            ======`-.____`-.___\_____/___.-`____.-'======
                               `=---='

            .............................................
              God Blessing            Bug-free for life
        """
        pass



parser = argparse.ArgumentParser(description='ReID feature extraction')
parser.add_argument('--make_pred', default='ON', type=str, help='ON->make re-id prediction of test data, OFF-> only evaluate the re-id mAP performance on training data')
opt = parser.parse_args()

# Evaluate
# qf, ql, qc --> query feat, query label, query camera
# gf, gl, gc --> gallery feat, gallery label, gallery camera
def evaluate(query_feat, query_label, gallery_feat, gallery_label, threshold=-1):
    query_ff = query_feat.view(-1, 1)
    print("-"*30)
    print("Pred-->[{}]".format(query_label))
    # 计算 cosine distance
    # 这一步，就直接计算了 query与所有 gallery-images feature 的距离
    score = torch.mm(gallery_feat, query_ff)
    score = score.squeeze(1).cpu()
    score = score.numpy()

    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]  # 反转排序顺序，从大到小
    top10_result = [score[idx] for idx in index[0:10]]
    print("top 10:{}".format(top10_result))

    pred = []
    gtruth = []

    # obtain prediction result
    if threshold > 0:
        for idx in index:
            if score[idx] >= threshold:
                pred.append(idx)
            else:
                break
                #continue
        print("Query result length = {}".format(len(pred)))
    else:
        # if no threshold, we will choose the top 10 images as query result
        pred.extend(list(index[:10]))
        print("Query result length = {}".format(len(pred)))

    # obtain ground truth
    gf_indices = list(np.where(np.array(gallery_label) == query_label)[0])
    gtruth.extend(gf_indices)
    print("Ground truth result length = {}".format(len(gtruth)))

    # convert to string form
    pred = [str(p) for p in pred]
    gtruth = [str(gt) for gt in gtruth]
    print("-"*30)

    return pred, set(gtruth)

def predict(query_feat, query_label, gallery_feat, gallery_label, threshold=-1):
    query_ff = query_feat.view(-1, 1)
    # print("-"*30)
    # print("Pred-->[{}]".format(query_label))
    # 计算 cosine distance
    # 这一步，就直接计算了 query与所有 gallery-images feature 的距离
    score = torch.mm(gallery_feat, query_ff)
    score = score.squeeze(1).cpu()
    score = score.numpy()

    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]  # 反转排序顺序，从大到小
    top10_result = [score[idx] for idx in index[0:10]]
    # print("top 10:{}".format(top10_result))

    pred_index = []

    # obtain prediction result
    if threshold > 0:
        for idx in index:
            if score[idx] >= threshold:
                pred_index.append(idx)
            else:
                break

        # print("Query result length = {}".format(len(pred_index)))
    else:
        # if no threshold, we will choose the top 10 images as query result
        pred_index.extend(list(index[:10]))
        # print("Query result length = {}".format(len(pred_index)))

    pred_result = [str(gallery_label[i]) for i in pred_index]
    # convert to string form
    # print("-"*30)
    return str(query_label), pred_result

def write_prediction(preds_dict, file_name):
    dummy_head = 100000000

    with open(file_name, 'w') as fbj:
        last_line = len(preds_dict) - 1
        i = 0
        for target, result in preds_dict.items():
            lineTXT = ''
            target_name = str(dummy_head + int(target))[-6:] + '.jpg'
            lineTXT = target_name + ': '
            for img in result:
                rlt_name = str(dummy_head + int(img))[-6:] + '.jpg'
                lineTXT += (rlt_name + ', ')
            if i != last_line:
                lineTXT = lineTXT[:-2] +'\n'
            else:
                lineTXT = lineTXT[:-2]
            i += 1
            fbj.write(lineTXT)

def make_prediction(result_path, threshold=-1):
    # 读取存储在 .mat 文件中的 query feature + label, gallery feature + label
    result = scipy.io.loadmat(result_path)

    query_feature = torch.FloatTensor(result['query_f'])
    query_label = result['query_label'][0]
    # print("query feat shape = {}".format(query_feature.shape))
    # print("query label len = {}".format(len(query_label)))

    gallery_feature = torch.FloatTensor(result['gallery_f'])
    gallery_label = result['gallery_label'][0]
    # print("gallery_feature feat shape = {}".format(gallery_feature.shape))
    # print("gallery label len = {}".format(len(gallery_label)))

    query_feature = query_feature.cuda()
    gallery_feature = gallery_feature.cuda()

    print(query_feature.shape)

    # 迭代查询 query 中的每个图片
    # 整个 query 的 mAP
    predictions = {}            # key = query img prefix, value = sorted result Indices in gallery images
    for i in range(len(query_label)):
        target, result = predict(query_feature[i], query_label[i],
                               gallery_feature, gallery_label, threshold=threshold)

        predictions[target] = result

    # write and store the result
    reid_filename = "reid_test.txt"
    write_prediction(preds_dict=predictions, file_name=reid_filename)
    print('ReID prediction complete :)')


######################################################################
if __name__ == "__main__":
    # 读取存储在 .mat 文件中的 query feature + label, gallery feature + label
    result = scipy.io.loadmat('./features/reID_training_result.mat')

    query_feature = torch.FloatTensor(result['query_f'])
    query_label = result['query_label'][0]
    print("query feat shape = {}".format(query_feature.shape))
    print("query label len = {}".format(len(query_label)))

    gallery_feature = torch.FloatTensor(result['gallery_f'])
    gallery_label = result['gallery_label'][0]
    print("gallery_feature feat shape = {}".format(gallery_feature.shape))
    print("gallery label len = {}".format(len(gallery_label)))


    query_feature = query_feature.cuda()
    gallery_feature = gallery_feature.cuda()

    # 迭代查询 query 中的每个图片
    # 整个 query 的 mAP
    threshold = 0.86
    predictions = {}            # key = str query img prefix, value = str sorted result Indices in gallery images
    ground_truth = {}           # key = str query img prefix, value = str ground truth Indices in gallery images
    for i in range(len(query_label)):
        pred, gtruth= evaluate(query_feature[i], query_label[i],
                               gallery_feature, gallery_label, threshold=threshold)

        predictions[str(query_label[i])] = pred
        ground_truth[str(query_label[i])] = gtruth

    # evaluate mAP value for validation
    # print("query prediction = {}".format(len(predictions)))
    # print("query ground truth = {}".format(len(ground_truth)))

    map_evaluator = Evaluator()
    mAP = map_evaluator.evaluate_map(predictions, ground_truth)
    print("ReID completed, threshold={}, mAP = {:.4}%".format(threshold, 100*mAP))
    # ReID completed, threshold=0.86, mAP = 45.01%

    # if we are satisfied by mAP result, we can finally make the ReID prediction
    # we will use the [query] dataset as query and [test] dataset as gallery
    # in the end, we write the prediction result into '.txt' file
    #
    if opt.make_pred == 'ON':
        rlt_path = './features/reID_prediction_result.mat'
        make_prediction(result_path=rlt_path, threshold=0.86)







