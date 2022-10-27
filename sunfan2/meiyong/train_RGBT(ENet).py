import torch
from torch import nn
from Dataprocess.RGBT_dataprocessing_CNet import trainData,valData
from torch.utils.data import DataLoader
from torch import optim
from datetime import datetime
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import Loss.lovasz_losses as lovasz

# from Net.model.ENet_mobilenet.mb4_add1 import net2
from Net.model.ENet_mobilenet.mb4_add1_1 import net2
import torchvision
import time
import os
import shutil
from Dataprocess.log import get_logger

import matplotlib.pyplot as plt




def print_network(model,name):
    num_params = 0
    for p in model.parameters():
        num_params +=p.numel()
    print(name)
    print("The number of parameters:{}M".format(num_params/1000000))

# class DiceLoss(nn.Module):
#     def __init__(self):
#         super(DiceLoss, self).__init__()
#
#     def forward(self, input, target):
#         N = target.size(0)
#         smooth = 1
#
#         input_flat = input.view(N, -1)
#         target_flat = target.view(N, -1)
#
#         intersection = input_flat * target_flat
#
#         loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
#         # loss = 1 - loss.sum() / N
#         return 1 - loss

# class FocalLoss(nn.Module):
#     def __init__(self, alpha=0.25, gamma=2, logits=False, sampling='mean'):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.logits = logits
#         self.sampling = sampling
#
#     def forward(self, y_pred, y_true):
#         alpha = self.alpha
#         alpha_ = (1 - self.alpha)
#         if self.logits:
#             y_pred = torch.sigmoid(y_pred)
#
#         pt_positive = torch.where(y_true == 1, y_pred, torch.ones_like(y_pred))
#         pt_negative = torch.where(y_true == 0, y_pred, torch.zeros_like(y_pred))
#         pt_positive = torch.clamp(pt_positive, 1e-3, .999)
#         pt_negative = torch.clamp(pt_negative, 1e-3, .999)
#         pos_ = (1 - pt_positive) ** self.gamma
#         neg_ = pt_negative ** self.gamma
#
#         pos_loss = -alpha * pos_ * torch.log(pt_positive)
#         neg_loss = -alpha_ * neg_ * torch.log(1 - pt_negative)
#         loss = pos_loss + neg_loss
#
#         if self.sampling == "mean":
#             return loss.mean()
#         elif self.sampling == "sum":
#             return loss.sum()
#         elif self.sampling == None:
#             return loss




# class BinaryDiceLoss(nn.Module):
#     def __init__(self):
#         super(BinaryDiceLoss, self).__init__()
#
#     def forward(self, input, targets):
#         # 获取每个批次的大小 N
#         N = targets.size()[0]
#         # 平滑变量
#         smooth = 1
#         # 将宽高 reshape 到同一纬度
#         input_flat = input.view(N, -1)
#         targets_flat = targets.view(N, -1)
#
#         # 计算交集
#         intersection = input_flat * targets_flat
#         N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
#         # 计算一个批次中平均每张图的损失
#         loss = 1 - N_dice_eff.sum() / N
#         return loss


class DICELOSS(nn.Module):
    def __init__(self):
        super(DICELOSS, self).__init__()

    def forward(self, input_scale, taeget_scale):
        losses = []
        for inputs, targets in zip(input_scale, taeget_scale):
            N,_,_ =inputs.shape

            smooth = 1
            # 将宽高 reshape 到同一纬度

            input_flat = inputs.view(N, -1)
            targets_flat = targets.view(N, -1)

            # 计算交集
            intersection = input_flat * targets_flat
            N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
            losses.append(1-N_dice_eff )
        total_loss = sum(losses)

        return total_loss

class lovaszloss(nn.Module):
    def __init__(self):
        super(lovaszloss, self).__init__()

    def forward(self, input_scale, taeget_scale):
        losses = []
        for inputs, targets in zip(input_scale, taeget_scale):
            lossall = lovasz.lovasz_hinge(inputs, targets)
            losses.append(lossall)
        total_loss = sum(losses)
        return total_loss


class BCELOSS(nn.Module):
    def __init__(self):
        super(BCELOSS, self).__init__()
        self.nll_lose = nn.BCELoss()

    def forward(self, input_scale, taeget_scale):
        losses = []
        for inputs, targets in zip(input_scale, taeget_scale):
            lossall = self.nll_lose(inputs, targets)
            losses.append(lossall)
        total_loss = sum(losses)
        return total_loss

################################################################################################################
batchsize = 48
################################################################################################################

train_dataloader = DataLoader(trainData, batch_size=batchsize, shuffle=True, num_workers=4)
test_dataloader = DataLoader(valData,batch_size=batchsize,shuffle=True,num_workers=4)


net = net2()
net = net.cuda()

################################################################################################################
model = 'mb4_add1_1_'+ time.strftime("%Y_%m_%d_%H_%M")
print_network(net,model)
################################################################################################################
bestpath = '../../../Pth/'+ model +'_best.pth'
lastpath = '../../../Pth/'+ model +'_last.pth'
################################################################################################################


criterion1 = BCELOSS().cuda()
criterion2 = BCELOSS().cuda()
criterion3 = BCELOSS().cuda()
criterion4 = BCELOSS().cuda()
criterion5 = BCELOSS().cuda()
criterion6 = BCELOSS().cuda()
criterion7 = BCELOSS().cuda()
criterion8 = BCELOSS().cuda()

criterion_val = BCELOSS().cuda()
# criterion_lovasz1 = lovaszloss().cuda()
# criterion_lovasz2 = lovaszloss().cuda()
# criterion_lovasz3 = lovaszloss().cuda()
dice_loss1 = DICELOSS().cuda()
dice_loss2 = DICELOSS().cuda()
dice_loss3 = DICELOSS().cuda()
dice_loss4 = DICELOSS().cuda()
dice_loss5 = DICELOSS().cuda()
dice_loss6 = DICELOSS().cuda()


################################################################################################################
lr_rate = 1e-3
optimizer = optim.Adam(net.parameters(), lr=lr_rate, weight_decay=1e-3)
################################################################################################################

best = [10]
step=0

logdir = f'run/{time.strftime("%Y-%m-%d-%H-%M")}({model})'
if not os.path.exists(logdir):
    os.makedirs(logdir)

logger = get_logger(logdir)
logger.info(f'Conf | use logdir {logdir}')

################################################################################################################
epochs = 200
################################################################################################################

logger.info(f'Epochs:{epochs}  Batchsize:{batchsize}')
for epoch in range(epochs):
    trainmae = 0
    if (epoch+1) % 20 == 0 and epoch != 0:
        for group in optimizer.param_groups:
            group['lr'] = 0.85 * group['lr']
            print(group['lr'])
            lr_rate = group['lr']


    train_loss = 0
    net = net.train()
    prec_time = datetime.now()
    for i, sample in enumerate(train_dataloader):
        image = Variable(sample['RGB'].cuda())
        depth = Variable(sample['depth'].cuda())
        label = Variable(sample['label'].float().cuda())
        bound = Variable(sample['bound'].float().cuda())

        optimizer.zero_grad()

        out,out1,out2= net(image, depth)
        out = F.sigmoid(out)
        out1 = F.sigmoid(out1)
        out2 = F.sigmoid(out2)


        loss = criterion1(out,label)
        loss1 = criterion2(out1, label)
        loss2 = criterion3(out2, bound)

        iou_loss = dice_loss1(out,label)
        iou_loss1 = dice_loss2(out1, label)
        iou_loss2 = dice_loss3(out2, bound)






        loss_total = loss + loss1 + loss2 +iou_loss + iou_loss1 + iou_loss2


        time = datetime.now()

        if i % 10 == 0 :
            print('{}  epoch:{}/{}  {}/{}  total_loss:{} loss:{}  iou:{}'
                  '  '.format(time, epoch+1, epochs, i, len(train_dataloader),loss_total.item(),loss,iou_loss.item()))
        loss_total.backward()
        optimizer.step()
        train_loss = loss_total.item() + train_loss

    net = net.eval()
    eval_loss = 0
    mae = 0

    with torch.no_grad():
        for j, sampleTest in enumerate(test_dataloader):

            imageVal = Variable(sampleTest['RGB'].cuda())
            depthVal = Variable(sampleTest['depth'].cuda())
            labelVal = Variable(sampleTest['label'].float().cuda())

            out,out1,out2 = net(imageVal, depthVal)
            out = F.sigmoid(out)


            loss = criterion_val(out, labelVal)

            maeval = torch.sum(torch.abs(labelVal - out)) / (224.0*224.0)

            print('===============', j, '===============', loss.item())

            # if j==34:
            #     out=out[4].cpu().numpy()
            #     edge = edge[4].cpu().numpy()
            #     out = out.squeeze()
            #     edge = edge.squeeze()
            #     plt.imsave('/home/sunfan/代码/shiyan/Net/model/ENet_mobilenet/img/out.png', out,cmap='gray')
            #     plt.imsave('/home/sunfan/代码/shiyan/Net/model/ENet_mobilenet/img/edge1.png', edge,cmap='gray')

            eval_loss = loss.item() + eval_loss
            mae = mae + maeval.item()
    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prec_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = '{:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
    logger.info(
        f'Epoch:{epoch+1:3d}/{epochs:3d} || trainloss:{train_loss / 2500:.8f} valloss:{eval_loss / 821:.8f} || '
        f'valmae:{mae / 821:.8f} || lr_rate:{lr_rate} || spend_time:{time_str}')

    if (mae / 821) <= min(best):
        best.append(mae / 821)
        nummae = epoch+1
        torch.save(net.state_dict(), bestpath)

    torch.save(net.state_dict(), lastpath)
    print('=======best mae epoch:{},best mae:{}'.format(nummae,min(best)))
    logger.info(f'best mae epoch:{nummae:3d}  || best mae:{min(best)}')














