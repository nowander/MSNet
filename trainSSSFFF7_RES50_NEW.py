import os
import torch
import torch.nn.functional as F

import numpy as np
from datetime import datetime
from torchvision.utils import make_grid

# from SFFFF.JJNet15 import JJNet
from SFFFF.SSFF7_RES import  SFNet
from rgbd_dataset import get_loader, test_dataset
from utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from config import opt
from torch.cuda import amp
import random
# set the device for training
cudnn.benchmark = True

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
print('USE GPU:', opt.gpu_id)

# build the model

model = SFNet()
print('NOW USING:SSFFF5_NEWVMMM_RES50')
# if (opt.load is not None):
# model.load_state_dict(torch.load('/home/sunfan/1212121212/pth2_SSFFF2_RES/SSFFF2_RES_best_mae_test.pth'))
# print('load model from ', opt.load)
# device = torch.cuda.set_device(0)
# model.to(device)
model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

# set the path
train_dataset_path = opt.lr_train_root
image_root = train_dataset_path + '/RGB/'
depth_root = train_dataset_path + '/depth/'
gt_root = train_dataset_path + '/GT/'
val_dataset_path = opt.lr_val_root
val_image_root = val_dataset_path + '/RGB/'
val_depth_root = val_dataset_path + '/depth/'
val_gt_root = val_dataset_path + '/GT/'
save_path = opt.save_path

if not os.path.exists(save_path):
    os.makedirs(save_path)

# load data
print('load data...')

train_loader = get_loader(image_root, gt_root,depth_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
# print(len(train_loader))
test_loader = test_dataset(val_image_root, val_gt_root,val_depth_root, opt.trainsize)
total_step = len(train_loader)

logging.basicConfig(filename=save_path + 'log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info(save_path + "Train")
logging.info("Config")
logging.info(
    'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(
        opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate, opt.load, save_path,
        opt.decay_epoch))

# set loss function
import torch.nn as nn


class Triangle_loss(nn.Module):
    def __init__(self):
        super(Triangle_loss, self).__init__()
        self.nll_lose = nn.BCEWithLogitsLoss()
        self.mseloss = nn.MSELoss()

    def forward(self, input_scale, taeget_scale):
        b,_,_,_ = input_scale.size()
        loss = []
        for inputs, targets in zip(input_scale, taeget_scale):
            pred = torch.sigmoid(inputs)
            # print(pred.shape)
            inter = (pred * targets).sum(dim=(1, 2))
            union = (pred + targets).sum(dim=(1, 2))
            IOU = (inter + 1) / (union - inter + 1)
            BCE = self.nll_lose(inputs, targets)
            MSE = self.mseloss(pred, targets)
            losses = ((1 + MSE) * BCE) / (IOU + 1)
            loss.append(losses)
        total_loss = sum(loss)
        return total_loss / b


class Depth_loss(nn.Module):
    def __init__(self):
        super(Depth_loss, self).__init__()

    def forward(self, predict, actual_depth):
        loss = []
        b, _, _, _ = predict.size()
        for image, depth in zip(predict, actual_depth):
            n_pixels = depth.shape[1] * depth.shape[2]
            # image = (image * 0.225) + 0.45
            # image = image * 255
            # image[image <= 0] = 0.00001
            # depth[depth == 0] = 0.00001
            # depth.unsqueeze_(dim=1)

            e = torch.abs(image - depth)
            # print(e)
            term_1 = torch.pow(e, 2).mean(dim=1).sum()
            term_2 = (torch.pow(e.sum(dim=1), 2) / (2 * (n_pixels ** 2)))
            loss.append(term_1-term_2)
        total_loss = sum(loss)
        return total_loss



class BCELOSS(nn.Module):
    def __init__(self):
        super(BCELOSS, self).__init__()
        self.nll_lose = nn.BCELoss()

    def forward(self, input_scale, taeget_scale):
        losses = []


        for inputs, targets in zip(input_scale, taeget_scale):
            weight = torch.mean(inputs,dim=(1,2))
            lossall = (1-weight)*self.nll_lose(inputs, targets)
            # print(targets.shape)
            # lossall = sigmoid_cross_entropy_loss(inputs,targets)
            losses.append(lossall)
        total_loss = sum(losses)
        return total_loss


# criterion = torch.nn.BCELoss()
criterion = torch.nn.BCEWithLogitsLoss()
# Trianglecriterion = Triangle_loss().cuda()
# L1criterion = nn.SmoothL1Loss().cuda()
# Depthcriterion = Depth_loss().cuda()


# 超参数
step = 0
writer = SummaryWriter(save_path + 'summary')
best_mae = 1
best_epoch = 0
Sacler = amp.GradScaler()

# train function
def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, gts,depths) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            images = images.cuda()
            # print(images.shape)
            depths = depths.cuda()
            gts = gts.cuda()
            _,_,w_t,h_t = gts.size()

            gts2 = F.upsample(gts, (w_t // 2, h_t // 2), mode='bilinear')
            gts3 = F.upsample(gts, (w_t // 4, h_t // 4), mode='bilinear')
            gts4 = F.upsample(gts, (w_t // 8, h_t // 8), mode='bilinear')
            gts5 = F.upsample(gts, (w_t // 16, h_t // 16), mode='bilinear')
            # Gabor_ls变为三通道,Gabor_rs变为三通道
            # n, c, h, w = images.size()  # batch_size, channels, height, weight
            # Gabor_ls = Gabor_ls.view(n, h, w, 1).repeat(1, 1, 1, c)
            # Gabor_ls = Gabor_ls.transpose(3, 1)
            # Gabor_ls = Gabor_ls.transpose(3, 2)
            #
            # Gabor_rs = Gabor_rs.view(n, h, w, 1).repeat(1, 1, 1, c)
            # Gabor_rs = Gabor_rs.transpose(3, 1)
            # Gabor_rs = Gabor_rs.transpose(3, 2)

            # with amp.autocast():
            n, c, h, w = images.size()
            depths = depths.view(n, h, w, 1).repeat(1, 1, 1, c)
            depths = depths.transpose(3, 1)
            depths = depths.transpose(3, 2)
            out = model(images, depths)
            # print(out)
            # out = torch.sigmoid(out)
            # print(out)
            # out0 = torch.sigmoid(out[0])
            # out1 = torch.sigmoid(out[1])
            # out2 = torch.sigmoid(out[2])
            # out3 = torch.sigmoid(out[3])
            # out4 = torch.sigmoid(out[4])
            # out5 = torch.sigmoid(out[5])
            # out6 = torch.sigmoid(out[6])
            # out7 = torch.sigmoid(out[7])
            # out8 = torch.sigmoid(out[8])
            # out9 = torch.sigmoid(out[9])
            #
            out0 = out[0]
            out1 = out[1]
            out2 = out[2]
            out3 = out[3]
            # out4 = out[4]
            # out5 = out[5]
            # out6 = out[6]
            # out7 = out[7]
            # out8 = out[8]
            # out9 = out[9]
            # loss = criterion(out, gts)
            # print(loss)
            loss2 = criterion(out0, gts)
            loss3 = criterion(out1, gts)
            loss4 = criterion(out2, gts)
            loss5 = criterion(out3, gts)
            # loss6 = criterion(out4, gts)
            # loss7 = criterion(out5, gts)
            # loss8 = criterion(out6, gts)
            # loss9 = criterion(out7, gts)
            # loss10 = criterion(out8, gts)
            # loss11 = criterion(out9, gts)
            # loss1 = criterion(out[0], gts)
            # loss2 = criterion(out[1], gts)
            # loss3 = criterion(out[2], gts)
            # loss4 = criterion(out[3], gts)
            # loss5 = criterion(out[4], gts)
            # loss6 = criterion(out[5], gts)


            loss = loss2 + loss3 + loss4 + loss5 #+ loss6 #+ loss7 +loss8 + loss9 + loss10 +loss11
            # print(loss.data[0])
            loss.backward()
            # Sacler.scale(loss).backward()
            # clip_gradient(optimizer, opt.clip)
            # Sacler.step(optimizer)
            # Sacler.update()
            optimizer.step()
            step += 1
            epoch_step += 1
            loss_all += loss.item()
            if i % 1 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}],W*H [{:03d}*{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
                      format(datetime.now(), epoch+1, opt.epoch, w_t, h_t, i, total_step, loss.item()))
                # print('{} ,W*H [{:03d}*{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
                #       format(datetime.now(), w_t, h_t, i, total_step, loss.item()))
                logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
                             format(epoch+1, opt.epoch, i, total_step, loss.item()))
                writer.add_scalar('Loss/total_loss', loss, global_step=step)

                # res = out[0][0].clone().sigmoid().data.cpu().numpy().squeeze()
                # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                # writer.add_image('SOD_contrast/last_out', torch.tensor(res), step, dataformats='HW')
                #
                # res = out[1][0].clone().sigmoid().data.cpu().numpy().squeeze()
                # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                # writer.add_image('SOD_contrast/second_step', torch.tensor(res), step, dataformats='HW')
                #
                # res = out[6][0].clone().sigmoid().data.cpu().numpy().squeeze()
                # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                # writer.add_image('SOD_contrast/first_step', torch.tensor(res), step, dataformats='HW')
                #
                # res = out[7][0].clone().data.cpu().numpy().squeeze()
                # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                # writer.add_image('Depth_contrast/depth_predict', torch.tensor(res), step, dataformats='HW')


        loss_all /= epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch+1, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if (epoch+1) % 20 == 0 or (epoch+1) == opt.epoch:
            torch.save(model.state_dict(), save_path + 'SSFFF7_RES50_epoch_{}_test.pth'.format(epoch+1))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'SSFFF7_RES50_epoch_{}_test.pth'.format(epoch + 1))
        print('save checkpoints successfully!')
        raise

#

# test function
def test(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, depth, name = test_loader.load_data()
            # print(image,right,name,Gabor_l,Gabor_r)
            gt = gt.cuda()
            image = image.cuda()
            depth = depth.cuda()
            n ,c ,h ,w =image.size()
            depth = depth.view(n ,h, w, 1).repeat(1 , 1, 1,c)
            depth = depth.transpose(3,1)
            depth = depth.transpose(3,2)

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
            res = torch.sigmoid(res)
            res = (res-res.min())/(res.max()-res.min()+1e-8)
            mae_train =torch.sum(torch.abs(res-gt))*1.0/(torch.numel(gt))
            mae_sum = mae_train.item()+mae_sum
            #mae_sum += torch.sum(torch.abs(res - gt)) / torch.numel(gt)
                # print(torch.numel(gt))
                # print(mae_sum)
        mae = mae_sum / test_loader.size
        # print(mae,test_loader.size)
        #writer.add_scalar('MAE', torch.as_tensor(mae), global_step=epoch)
        # res = res[0].clone().data.cpu().numpy().squeeze()
        # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        # writer.add_image('SOD_contrast/test_predict', torch.tensor(res), step, dataformats='HW')
        # print(' MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(mae, best_mae, best_epoch))
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'SSFFF7_RES50_best_mae_test.pth')
                print('best epoch:{}'.format(epoch))
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))



if __name__ == '__main__':
    print("Start train...")
    start_time = datetime.now()
    for epoch in range(opt.epoch):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        train(train_loader, model, optimizer, epoch, save_path)
        test(test_loader, model, epoch, save_path)
    finish_time = datetime.now()
    h, remainder = divmod((finish_time - start_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
    print(time)