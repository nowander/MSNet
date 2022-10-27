import torch as t
# from Dataprocess.RGBT_dataprocessing import testData1,testData2,testData3
from rgbd_dataset import get_loader, test_dataset
from torch.utils.data import DataLoader
import os
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
test_dataset1 = test_dataset(image_root='/home/sunfan/Downloads/newdata/test/NJU2K/RGB',gt_root='/home/sunfan/Downloads/newdata/test/NJU2K/GT',depth_root='/home/sunfan/Downloads/newdata/test/NJU2K/depth',testsize=224)

test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)


# 导入模型

from SFNet.SFNet6_Res_NDEM import  SFNet


net = SFNet()
# net.load_state_dict(t.load('../Pth/mobile_cx5_bestmae0917.pth'))
print('loaded')

# 导入训练好的参数

# net.load_state_dict(t.load('./weight.pth'))


import torch

with torch.no_grad():
	net.eval()
	net.cuda()
	test_mae = 0
	prec_time = datetime.now()
	for i, sample in enumerate(test_dataloader):

		image = sample['RGB'].cuda()
		depth = sample['depth'].cuda()
		label = sample['label'].cuda()
		name = sample['name']
		name = "".join(name)

		out1,_= net(image,depth)
		out= torch.sigmoid(out1)

	torch.cuda.synchronize()
	cur_time = datetime.now()
h, remainder = divmod((cur_time - prec_time).seconds, 3600)
m, s = divmod(remainder, 60)
fps = len(test_dataloader)/(m*60+s)
time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
print(time_str, 'fps: ', fps)



