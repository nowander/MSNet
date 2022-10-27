import torch as t
# from dataprocessing333 import testData
from rgbd_dataset import test_dataset
from torch.utils.data import DataLoader
import os
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
# test_datasets = ['NJU2K','STERE','DES','SIP','LFSD','NLPR']
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)


# path = './predict/DUT/'
# isExist = os.path.exists('./predict/DUT/')
# if not isExist:
# 	os.makedirs('./predict/DUT/')
# else:
# 	print('path exist')
# 导入模型
# from 不知名来源网络.SSF.model import model_VGG
# from 不知名来源网络.BBS.CPD_models_vgg_2 import CPD_VGG
from sunfan.SFNet6_vgg_NEW_SSIM_NEW_Rotation import SFNet
# from 不知名来源网络.CMWNet.model import CMW_Net
net = SFNet()

# 导入训练好的参数

# net.load_state_dict(t.load('./weight.pth'))


import torch

with torch.no_grad():
	net.eval()
	net.cuda()
	test_mae = 0
	prec_time = datetime.now()
	# for i in range(test_dataloader.size):
	# 	image, gt, depth, name = test_dataloader.load_data()
	# 	# print(image,right,name,Gabor_l,Gabor_r)
	# 	gt = gt.cuda()
	# 	image = image.cuda()
	# 	depth = depth.cuda()
	for i, sample in enumerate(test_dataloader):
		image = sample['image']
		depth = sample['depth']
		label = sample['label']
		name = sample['name']
		name = "".join(name)

		image = Variable(image).cuda()
		depth = Variable(depth).cuda()
		label = Variable(label).cuda()
		depth1 = torch.split(depth, 1, dim=1)
		# _, out, _, _ = net(imageVal, depthVal, depthVal1[0])
		out = net(image, depth)
		out = torch.sigmoid(out[0])


		# out_img = out.cpu().detach().numpy()

		# out_img = out_img.squeeze()

		# print(out_img)
		# plt.imsave(path + name + '.png', arr=out_img, cmap='gray')
	cur_time = datetime.now()
h, remainder = divmod((cur_time - prec_time).seconds, 3600)
m, s = divmod(remainder, 60)
# print('a')
fps = len(test_dataloader)/(m*60+s)
# print('b')
time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
print(time_str, 'fps: ', fps)



