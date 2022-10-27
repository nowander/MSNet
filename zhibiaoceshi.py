import torch
import torch.nn.functional as F
import sys
import numpy as np
import os
import cv2
# from sunfan.SFNet6_Res_NEW_SSIM_NEW_Rotation import   SFNet
from sunfan.SFNet6_vgg_NEW_SSIM_NEW_Rotation import   SFNet
#from SFNet.SFNet5_Res import SFNet
import matplotlib.pyplot as plt
# from lr.复现的网络.BBS改 import BBSNet
from config import opt
from rgbd_dataset import test_dataset

# https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/metrics.py

import numpy as np
import torch
from PIL import Image
import os
import cv2

def getScores(conf_matrix):
    if conf_matrix.sum() == 0:
        return 0, 0, 0, 0, 0
    with np.errstate(divide='ignore',invalid='ignore'):
        globalacc = np.diag(conf_matrix).sum() / np.float(conf_matrix.sum())
        classpre = np.diag(conf_matrix) / conf_matrix.sum(0).astype(np.float)
        classrecall = np.diag(conf_matrix) / conf_matrix.sum(1).astype(np.float)
        IU = np.diag(conf_matrix) / (conf_matrix.sum(1) + conf_matrix.sum(0) - np.diag(conf_matrix)).astype(np.float)
        pre = classpre[1]
        recall = classrecall[1]
        iou = IU[1]
        F_score = 2*(recall*pre)/(recall+pre)
    return globalacc, pre, recall, F_score, iou

class runningScore(object):
    '''
        n_classes: database的类别,包括背景
        ignore_index: 需要忽略的类别id,一般为未标注id, eg. CamVid.id_unlabel
    '''

    def __init__(self, n_classes, ignore_index=None):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

        if ignore_index is None:
            self.ignore_index = None
        elif isinstance(ignore_index, int):
            self.ignore_index = (ignore_index,)
        else:
            try:
                self.ignore_index = tuple(ignore_index)
            except TypeError:
                raise ValueError("'ignore_index' must be an int or iterable")

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - pixel_acc:
            - class_acc: class mean acc
            - mIou :     mean intersection over union
            - fwIou:     frequency weighted intersection union
        """

        hist = self.confusion_matrix

        acc, pre, recall, f_score, iou = getScores(hist)


        # ignore unlabel
        # if self.ignore_index is not None:
        #     for index in self.ignore_index:
        #         hist = np.delete(hist, index, axis=0)
        #         hist = np.delete(hist, index, axis=1)

        # acc = np.diag(hist).sum() / hist.sum()
        # acc_cls = np.diag(hist) / hist.sum(axis=1)
        # acc_cls = np.nanmean(acc_cls)
        # iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        # mean_iou = np.nanmean(iu)
        # freq = hist.sum(axis=1) / hist.sum()
        # fw_iou = (freq[freq > 0] * iu[freq > 0]).sum()

        # # set unlabel as nan
        # if self.ignore_index is not None:
        #     for index in self.ignore_index:
        #         iu = np.insert(iu, index, np.nan)

        # cls_iu = dict(zip(range(self.n_classes), iu))

        return {# acc, pre, recall, f_score, iou
                "Accuracy: ": acc,
                "Precision: ": pre,
                "Recall: ": recall,
                "F_Score: ": f_score,
                "IoU: ": iou
            }

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class averageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


from datetime import datetime
from sklearn.metrics import confusion_matrix
def confusion(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amounat of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """

    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return np.array([true_positives, false_positives, true_negatives, false_negatives]).reshape(2, 2)
def getScores1(conf_matrix1):
    if conf_matrix1[0] == 0:
        pre =0
        recall = 0
        F_score = 0
        globalacc =0
        dice = 0
    else:
        pre = conf_matrix1[0] / (conf_matrix1[0] + conf_matrix1[1])
        recall = (conf_matrix1[0] ) / (conf_matrix1[0] + conf_matrix1[3])
        globalacc = (conf_matrix1[0] + conf_matrix1[2]) / (
                    conf_matrix1[0] + conf_matrix1[1] + conf_matrix1[2] + conf_matrix1[3])
        # F_score = 2 / (1 / (pre ) + 1 / ((recall)))
        F_score = 2 * pre *recall/(pre + recall)
        dice = 2 * conf_matrix1[0] / (conf_matrix1[1] + 2 * conf_matrix1[0] + conf_matrix1[3])
    return globalacc, pre, recall, F_score, dice
    # return globalacc, pre, recall, F_score,dice

def getScores(conf_matrix):
    if conf_matrix.sum() == 0:
        return 0, 0, 0, 0, 0
    with np.errstate(divide='ignore',invalid='ignore'):
        globalacc = np.diag(conf_matrix).sum() / np.float(conf_matrix.sum())
        classpre = np.diag(conf_matrix) / conf_matrix.sum(0).astype(np.float)
        classrecall = np.diag(conf_matrix) / conf_matrix.sum(1).astype(np.float)
        IU = np.diag(conf_matrix) / (conf_matrix.sum(1) + conf_matrix.sum(0) - np.diag(conf_matrix)).astype(np.float)
        pre = classpre[1]
        recall = classrecall[1]
        iou = IU[1]
        F_score = 2*(recall*pre)/(recall+pre)
    return globalacc, pre, recall, F_score, iou

dst = '/media/sunfan/date/erzhihua/TANet/NJU2K224/'
gt_path = '/media/sunfan/date/erzhihua/GT/NJU2K/'

conf = runningScore(2)

test_globalacc = []
test_pre = []
test_recall = []
test_F_score = []
test_dice = []
# test_datasets = ['NJU2K','NLPR']
# test_datasets = ['DES','SIP','LFSD','NLPR','NJU2K','STERE','DUT']
PIL = os.listdir(dst)
len1 = 0
for pil in PIL:
    pre = os.path.join(dst,pil)#f'{src}/{pil}'
    gt =  os.path.join(gt_path,pil)
    isExist = os.path.exists(gt)
    if not isExist:
        print(gt)
        print('not exit')
        # print('sd', pre)
        continue
    pre = (cv2.imread(pre,0) / 255).astype(int)
    pre_img = np.expand_dims(pre, axis=0)
    gt = (cv2.imread(gt,0) / 255).astype(int)
    gt_image = np.expand_dims(np.expand_dims(gt, axis=0), axis=0)
    # print(pre_img.s)
    # pre_img = pre_img /255.
    # gt_image = gt_image/255.
    # a = np.unique(gt_image)
    # pre_img = pre_img.reshape(1,-1)
    # gt_image = gt_image.reshape(1,-1)
    # a = np.unique(pre_img)
    # print('a',a)
    # b= np.unique(gt_image)
    # print('b',b)




    # globalacc_sum =0
    # pre_sum = 0
    # recall_sum = 0
    # F_score_sum = 0
    # dice_sum = 0

    # len1 = len1+1
    # print('gt',gt_image.shape)
    # print(pil)
    # pre_img = torch.tensor(pre_img)
    # gt_image = torch.tensor(gt_image)
    # if gt_image.shape != pre_img.shape:
    #     print(gt_image.shape)
    #     print(pre_img.shape)
    # print(np.unique(gt_image))
    # print(np.unique(pre_img))
    # gt_image = gt_image /255.
    # pre_img = pre_img / 255.
    conf_matrix = conf.update(gt_image, pre_img)
    # print(conf_matrix)
    # print(conf_matrix.shape)
    # # conf_matrix=confusion(pre_img,gt_image)
    # conf_matrix = confusion_matrix(pre_img,gt_image)
    # # print('cc',conf_matrix)
    # last = getScores1(conf_matrix)
    # globalacc =last[0]
    # pre = last[1]
    # recall = last[2]
    # F_score = last[3]
    # dice = last[4]
    # globalacc_sum = globalacc + globalacc_sum
    # pre_sum = pre + pre_sum
    # recall_sum = recall +recall_sum
    # F_score_sum = F_score + F_score_sum
    # dice_sum = dice + dice_sum
    # predict = pre_img.data.cpu().numpy().squeeze()
    # print('save img to: ', save_path + name, )
    # predict = cv2.resize(predict,(224,224),interpolation=cv2.INTER_LINEAR)
        # cv2.imwrite(save_path + name, predict*255)
        # plt.imsave(save_path + name, arr=predict, cmap='gray')
    if len1 > 10:
        break
print(len1)
metrics = conf.get_scores()
cur_time = datetime.now()
print(metrics)
# test_mae.append(mae_sum / len1)
# test_globalacc.append(globalacc_sum )
# test_pre.append(pre_sum)
# test_recall.append(recall_sum)
# test_F_score.append(F_score_sum )
# test_dice.append(dice_sum )



# print('Test_mae:', test_mae)
# print('test_globalacc:', test_globalacc)
# print('test_pre:', test_pre)
# print('test_recall:', test_recall)
# print('test_F_score:', test_F_score)
# print('test_dice:', test_dice)
# print('Test Done!')