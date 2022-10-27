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
from datetime import datetime
from torch.cuda import amp
# from SFNet.SFNet6_Res_NDEM_abstract_MFB import SFNet
# from SFFFF.SSFF6_RES import  SFNet
# from SFFFF.SSFF7_VGG import   SFNet
dataset_path = opt.test_path

#set device for test
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
print('USE GPU:', opt.gpu_id)

#load the model
model = SFNet()
print('NOW USING:SFNet4_1012')
# ICNnet uses 180 epoch
# model.load_state_dict(torch.load('/media/sunfan/date/weight_rgbd/SF6_joinloss_3datasets_NEW_Rotation_RGB/SFNet6_Res_WBCE_best_mae.pth' , map_location={'cuda:1':'cuda:0'}))
model.load_state_dict(torch.load('/media/sunfan/date/weight_rgbd/SF6_VGG16/SFNet6_Res_WBCE_best_mae.pth' , map_location={'cuda:1':'cuda:0'}))
# model.load_state_dict(torch.load('/media/sunfan/date/weight_rgbd/SF6_joinloss_3datasets_NEW_Rotation_RGB/SFNet6_Res_WBCE_best_mae.pth'))
# model.load_state_dict(torch.load('/home/sunfan/1212121212/pth/BBS_8k/SSFFF7_VGG_8k_best_mae_test.pth'))
# model.load_state_dict(torch.load('/home/sunfan/1212121212/pth/SFNet6_Res_best_mae.pth'))
model.cuda()
model.eval()

#test
from sklearn.metrics import confusion_matrix
import numpy as np

def confusion(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
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

    return true_positives, false_positives, true_negatives, false_negatives
def getScores(conf_matrix1):
    # conf_matrix= np.mat([[conf_matrix1[0],conf_matrix1[1]],[conf_matrix1[2],conf_matrix1[3]]])
    # if conf_matrix.sum() == 0:
    #     return 0, 0, 0, 0, 0
    # # print('aa',conf_matrix.type)
    # with np.errstate(divide='ignore',invalid='ignore'):
    #     globalacc = np.diag(conf_matrix).sum() / np.float(conf_matrix.sum())
    #     classpre = np.diag(conf_matrix) / conf_matrix.sum(0).astype(np.float)
    #     classrecall = np.diag(conf_matrix) / conf_matrix.sum(1).astype(np.float)
    #     IU = np.diag(conf_matrix) / (conf_matrix.sum(1) + conf_matrix.sum(0) - np.diag(conf_matrix)).astype(np.float)
    #     # print(classrecall.shape)
    #     pre = classpre[0,1]
    #     recall = classrecall[0,1]
    #     iou = IU[1]
    #     F_score = 2*(recall*pre)/(recall+pre)
    pre = conf_matrix1[0]/(conf_matrix1[0] +conf_matrix1[1])
    recall = (conf_matrix1[0]+1e-4)/(conf_matrix1[0] + conf_matrix1[3]+1e-4)
    globalacc = (conf_matrix1[0] +conf_matrix1[3])/(conf_matrix1[0]+conf_matrix1[1]+conf_matrix1[2]+conf_matrix1[3])
    F_score =2/(1/(pre+1e-4)+1/((recall+1e-4)))
    dice = 2*conf_matrix1[0]/(conf_matrix1[1]+2*conf_matrix1[0]+conf_matrix1[3])
    return globalacc, pre, recall, F_score,dice

# pred = np.array([0, 0, 1, 1, 0, 1, 0]) # 预测向量
# label = np.array([0, 0, 1, 1, 0, 1, 1]) # 真实向量
#
# # 得到混淆矩阵
# conf_matrix = confusion_matrix(label, pred)
# conf_matrix
#
# # 得到五个指标 accuracy, precision, recall, f_score, iou
# globalacc, pre, recall, F_score, iou=getScores(conf_matrix)
# print(globalacc, pre, recall, F_score, iou)
test_mae = []
test_globalacc = []
test_pre = []
test_recall = []
test_F_score = []
test_dice = []
# test_datasets = ['NJU2K','NLPR']
test_datasets = ['DES','SIP','LFSD','NLPR','NJU2K','STERE','DUT']

for dataset in test_datasets:

    mae_sum  = 0
    globalacc_sum =0
    pre_sum = 0
    recall_sum = 0
    F_score_sum = 0
    dice_sum = 0
    save_path = '/media/sunfan/date/RGBD_MAP/SF6_joinloss_3datasets_NEW_Rotation_RGB/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/RGB/'
    gt_root = dataset_path + dataset + '/GT/'
    # depth_root=dataset_path +dataset +'/parrllex/'
    depth_root = dataset_path + dataset + '/depth/'
    test_loader = test_dataset(image_root, gt_root,depth_root, opt.testsize)
    prec_time = datetime.now()
    for i in range(test_loader.size):
        image, gt, depth, name = test_loader.load_data()
        # print(image,right,name,Gabor_l,Gabor_r)
        gt = gt.cuda()
        image = image.cuda()
        depth = depth.cuda()
        n, c, h, w = image.size()
        depth = depth.view(n, h, w, 1).repeat(1, 1, 1, c)
        depth = depth.transpose(3, 1)
        depth = depth.transpose(3, 2)

        # n, c, h, w = image.size()  # batch_size, channels, height, weight
        # Gabor_l = Gabor_l.view(n, h, w, 1).repeat(1, 1, 1, c)
        # Gabor_l = Gabor_l.transpose(3, 1)
        # Gabor_l = Gabor_l.transpose(3, 2)
        #
        # Gabor_r = Gabor_r.view(n, h, w, 1).repeat(1, 1, 1, c)
        # Gabor_r = Gabor_r.transpose(3, 1)
        # Gabor_r = Gabor_r.transpose(3, 2)
        # with amp.autocast():
        res = model(image, depth)
        # res = torch.split(res, 1, 1)
        res = torch.sigmoid(res)
        # res = res[0]
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        mae_train = torch.sum((torch.abs(res - gt)) * 1.0 / (torch.numel(gt)))
        mae_sum = mae_train.item() + mae_sum
        # # print(mae_sum)
        # gt1 = gt.data.cpu().numpy().squeeze().reshape(-1)
        # res1 = res.data.cpu().numpy().squeeze().reshape(-1)
        # print(res1.shape)
        # print(gt1.shape)
        # conf_matrix = confusion_matrix(res1,gt1)
        conf_matrix=confusion(res,gt)
        last = getScores(conf_matrix)
        globalacc =last[0]
        pre = last[1]
        recall = last[2]
        F_score = last[3]
        dice = last[4]
        globalacc_sum = globalacc + globalacc_sum
        pre_sum = pre + pre_sum
        recall_sum = recall +recall_sum
        F_score_sum = F_score + F_score_sum
        dice_sum = dice + dice_sum
        predict = res.data.cpu().numpy().squeeze()
        print('save img to: ', save_path + name, )
        predict = cv2.resize(predict,(224,224),interpolation=cv2.INTER_LINEAR)
        # cv2.imwrite(save_path + name, predict*255)
        # plt.imsave(save_path + name, arr=predict, cmap='gray')

    cur_time = datetime.now()
    test_mae.append(mae_sum / len(test_loader))
    test_globalacc.append(globalacc_sum / len(test_loader))
    test_pre.append(pre_sum / len(test_loader))
    test_recall.append(recall_sum / len(test_loader))
    test_F_score.append(F_score_sum / len(test_loader))
    test_dice.append(dice_sum / len(test_loader))

h, remainder = divmod((cur_time - prec_time).seconds, 3600)
m, s = divmod(remainder, 60)
# fps = len(test_loader) / (m * 60 + s)  test_loader.size
fps = test_loader.size / (m * 60 + s)
time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
print(time_str, 'fps: ', fps)

print('Test_mae:', test_mae)
print('test_globalacc:', test_globalacc)
print('test_pre:', test_pre)
print('test_recall:', test_recall)
print('test_F_score:', test_F_score)
print('test_dice:', test_dice)
print('Test Done!')