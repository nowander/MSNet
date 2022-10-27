import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import math
from torch.autograd import Variable
import numpy as np
from torch.nn import Softmax
from backbone.ResNet import resnet_left,resnet_right
from scipy import stats
from torchvision import models
from skimage.measure import compare_ssim
from sunfan import pytorch_ssim
# from SFNet.ResNet import resnet101
# res =resnet101(True)
# from backbone.vgg import vgg_right
# torch.set_default_tensor_type(torch.DoubleTensor)
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


class DEM(nn.Module):
    def __init__(self,inchannel):
        super(DEM, self).__init__()
        self.ChannelA = ChannelAttention(inchannel)
        self.ChannelS = SpatialAttention()

    def forward(self, x):
        # print("x",x.shape)
        temp = x.mul(self.ChannelA(x))
        temp = temp.mul(self.ChannelS(temp))
        x = x + temp
        # temp1 = x.mul(self.ChannelA(x))
        # temp2 = x.mul(self.ChannelS(x))
        # x = temp2 + temp1
        # print('xxxxxxxxxx')
        # print("x_a",x.shape)
        return x

class PAM(nn.Module):
    def __init__(self, channel, ratio=16):
        super(PAM, self).__init__()
        self.inter_channel = channel // ratio
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.gmma = nn.Parameter(torch.zeros(1))
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.channel = channel
    def forward(self, x):
        # [N, C, H, W]
        # print('x',x.shape)
        # print('self.inter_channel',self.inter_channel)
        b, c, h, w = x.size()
        c = c // 16
        # [N, C // ratio, H*W]
        # print('b',b)
        # print('c',c)
        # print('self.conv_phi(x)',self.conv_phi(x).shape)
        x_phi = self.conv_phi(x).view(b, c, -1)
        # [N, H * W, C // ratio]
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        # [N, H * W, H * W]M
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        # [N, H * W, C // ratio]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # [N, C // ratio, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0, 2, 1).contiguous().view(b, self.inter_channel, h, w)
        # [N, C, H, W]
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask * self.gmma + x
        return out



class CAM(nn.Module):
    def __init__(self, channel):
        super(CAM, self).__init__()
        self.channel = channel
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.channel, kernel_size=1, stride=1,
                                  padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.channel, kernel_size=1, stride=1,
                                    padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.channel, kernel_size=1, stride=1,
                                padding=0, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.conv_mask = nn.Conv2d(in_channels=self.channel, out_channels=channel, kernel_size=1, stride=1,
                                   padding=0, bias=False)
        # self.gmma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # [N, C, H, W]
        b, c, h, w = x.size()
        # [N, C, H*W]
        x_phi = self.conv_phi(x).view(b, c, -1)
        # [N, H*W, C]
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        # [N, C, H*w]
        x_g = self.conv_g(x).view(b, c, -1)
        # [N, C, C]
        mul_theta_phi = torch.matmul(x_phi, x_theta)
        mul_theta_phi = self.softmax(mul_theta_phi)
        # [N, C, H * W]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # [N, C, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0, 2, 1).contiguous().view(b, self.channel, h, w)
        # [N, C, H, W]
        mask = self.conv_mask(mul_theta_phi_g)
        out = torch.sigmoid(mask  + x)
        return out


class SPM(nn.Module):
    def __init__(self, channel, ratio=8, is_more=1):
        super(SPM, self).__init__()
        self.inter_channel = channel // ratio
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        # self.softmax = torch.sigmoid()
        # self.gmma = nn.Parameter(torch.zeros(1))
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.channel = channel
        self.is_more = is_more
    def forward(self, *x):
        x1=x[0]
        x2 = x[1]
        b, c, h, w = x1.size()
        c = c // 16
        x11 = self.conv_phi(x1).view(h * w, -1)
        x22 = self.conv_theta(x2).view(-1, h * w)
        y1 = torch.sigmoid(torch.matmul(x11, x22))
        if self.is_more:
            x3 = x[2]
            x33 = self.conv_g(x3).view(h * w, -1)
            y2 = torch.matmul(y1, x33)
            x_theta = y2.view(b, self.inter_channel, h, -1)
        else:
            # x3=0
            # print(x11.shape)
            # print(x22.shape)
            x_theta1 = torch.matmul(y1, self.conv_phi(x1).view(h * w, -1))
            x_theta2 = torch.matmul(y1, self.conv_theta(x2).view(h * w, -1)) + x_theta1

            x_theta = x_theta2.view(b, self.inter_channel, h, -1)
        # [N, C // ratio, H*W]
        # print('b',b)
        # print('c',c)
        # print('self.conv_phi(x)',self.conv_phi(x).shape)




        # [N, H * W, C // ratio]
        # print(y2.shape)
        #.permute(0, 2, 1,3).contiguous()
        # print(x_theta.shape)
        x_out = torch.sigmoid(self.conv_mask(x_theta))
        # x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        # # [N, H * W, H * W]M
        # mul_theta_phi = torch.matmul(x_theta, x_phi)
        # mul_theta_phi = self.softmax(mul_theta_phi)
        # # [N, H * W, C // ratio]
        # mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # # [N, C // ratio, H, W]
        # mul_theta_phi_g = mul_theta_phi_g.permute(0, 2, 1).contiguous().view(b, self.inter_channel, h, w)
        # # [N, C, H, W]
        # mask = self.conv_mask(mul_theta_phi_g)
        # out = mask * self.gmma + x
        return x_out

class RowAttention(nn.Module):

    def __init__(self, in_dim, q_k_dim):
        '''
        Parameters
        ----------
        in_dim : int
            channel of input img tensor
        q_k_dim: int
            channel of Q, K vector
        device : torch.device
        '''
        super(RowAttention, self).__init__()
        self.in_dim = in_dim
        self.q_k_dim = q_k_dim
        # self.device = device

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.q_k_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.q_k_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.in_dim, kernel_size=1)
        self.softmax = Softmax(dim=2)
        # self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        '''
        Parameters
        ----------
        x : Tensor
            4-D , (batch, in_dims, height, width) -- (b,c1,h,w)
        '''

        ## c1 = in_dims; c2 = q_k_dim
        b, _, h, w = x.size()

        Q = self.query_conv(x)  # size = (b,c2, h,w)
        K = self.key_conv(x)  # size = (b, c2, h, w)
        V = self.value_conv(x)  # size = (b, c1,h,w)

        Q = Q.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w).permute(0, 2, 1)  # size = (b*h,w,c2)
        K = K.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)  # size = (b*h,c2,w)
        V = V.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)  # size = (b*h, c1,w)

        # size = (b*h,w,w) [:,i,j] 表示Q的所有h的第 Wi行位置上所有通道值与 K的所有h的第 Wj列位置上的所有通道值的乘积，
        # 即(1,c2) * (c2,1) = (1,1)
        row_attn = torch.bmm(Q, K)
        ########
        # 此时的 row_atten的[:,i,0:w] 表示Q的所有h的第 Wi行位置上所有通道值与 K的所有行的 所有列(0:w)的逐个位置上的所有通道值的乘积
        # 此操作即为 Q的某个（i,j）与 K的（i,0:w）逐个位置的值的乘积，得到行attn
        ########

        # 对row_attn进行softmax
        row_attn = self.softmax(row_attn)  # 对列进行softmax，即[k,i,0:w] ，某一行的所有列加起来等于1，

        # size = (b*h,c1,w) 这里先需要对row_atten进行 行列置换，使得某一列的所有行加起来等于1
        # [:,i,j]即为V的所有行的某个通道上，所有列的值 与 row_attn的行的乘积，即求权重和
        out = torch.bmm(V, row_attn.permute(0, 2, 1))
        #(B*H,W,C)
        # size = (b,c1,h,2)
        out = out.view(b, h, -1, w).permute(0, 2, 1, 3)

        row_attn2 = torch.bmm(Q, V)
        row_attn2 = self.softmax(row_attn2)
        # print('Q',Q.shape)
        # print('row_attn2.permute(0, 2, 1)', row_attn2.permute(0, 2, 1).shape)
        out2 = torch.bmm(row_attn2.permute(0, 2, 1),Q)
        out2 = out2.view(b, h, -1, w).permute(0, 2, 1, 3)
        out = out + out2
        # out = self.gamma * out + x

        return out


# class RowAttention(nn.Module):
#
#     def __init__(self, in_dim, q_k_dim):
#         '''
#         Parameters
#         ----------
#         in_dim : int
#             channel of input img tensor
#         q_k_dim: int
#             channel of Q, K vector
#         device : torch.device
#         '''
#         super(RowAttention, self).__init__()
#         self.in_dim = in_dim
#         self.q_k_dim = q_k_dim
#         # self.device = device
#
#         self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.q_k_dim, kernel_size=1)
#         self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.q_k_dim, kernel_size=1)
#         self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.in_dim, kernel_size=1)
#         self.softmax = Softmax(dim=2)
#         # self.gamma = nn.Parameter(torch.zeros(1))
#
#     def forward(self, x):
#         '''
#         Parameters
#         ----------
#         x : Tensor
#             4-D , (batch, in_dims, height, width) -- (b,c1,h,w)
#         '''
#
#         ## c1 = in_dims; c2 = q_k_dim
#         b, _, h, w = x.size()
#
#         Q = self.query_conv(x)  # size = (b,c2, h,w)
#         K = self.key_conv(x)  # size = (b, c2, h, w)
#         V = self.value_conv(x)  # size = (b, c1,h,w)
#
#         Q = Q.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w).permute(0, 2, 1)  # size = (b*h,w,c2)
#         K = K.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)  # size = (b*h,c2,w)
#         V = V.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)  # size = (b*h, c1,w)
#
#         # size = (b*h,w,w) [:,i,j] 表示Q的所有h的第 Wi行位置上所有通道值与 K的所有h的第 Wj列位置上的所有通道值的乘积，
#         # 即(1,c2) * (c2,1) = (1,1)
#         row_attn = torch.bmm(Q, K)
#         ########
#         # 此时的 row_atten的[:,i,0:w] 表示Q的所有h的第 Wi行位置上所有通道值与 K的所有行的 所有列(0:w)的逐个位置上的所有通道值的乘积
#         # 此操作即为 Q的某个（i,j）与 K的（i,0:w）逐个位置的值的乘积，得到行attn
#         ########
#
#         # 对row_attn进行softmax
#         row_attn = self.softmax(row_attn)  # 对列进行softmax，即[k,i,0:w] ，某一行的所有列加起来等于1，
#
#         # size = (b*h,c1,w) 这里先需要对row_atten进行 行列置换，使得某一列的所有行加起来等于1
#         # [:,i,j]即为V的所有行的某个通道上，所有列的值 与 row_attn的行的乘积，即求权重和
#         out = torch.bmm(V, row_attn.permute(0, 2, 1))
#         #(B*H,W,C)
#         # size = (b,c1,h,2)
#         out = out.view(b, h, -1, w).permute(0, 2, 1, 3)
#         out = out + x
#         # out = self.gamma * out + x
#
#         return out


# def calc_corr(a, b):
#     import numpy as np
#     import time as t
#     # print(Kendallta2)
#     # a = np.random.normal(3, 2.5, size=(2, 1000))
#     # time1 = t.time()
#     # from dtaidistance import dtw
#     # # print(a.shape)
#     # # print(a[0].shape)
#     # # print(a[1].shape)
#     # dist_dtai = dtw.distance_fast(a[0], a[1])
#     # # print('dd',dist_dtai)
#     # time2 = t.time()
#     # cost_dtai = time2 - time1
#
#     #
#     # time1 = t.time()
#     # # from contextlib import ContextDecorator
#     # from dtaidistance import dtw
#     # a = a.detach().cpu().numpy().reshape(1,-1).astype('double')
#     # b = b.detach().cpu().numpy().reshape(1,-1).astype('double')
#     # dist_dtai = dtw.distance_fast(a[0], b[0])
#     # time2 = t.time()
#     # cost_dtai = time2 - time1
#     # print('cost_dtai',cost_dtai)
#     x1 = a.detach().cpu().numpy().reshape(1, -1)
#     y1 = b.detach().cpu().numpy().reshape(1, -1)
#     # x1 = a.view(1, -1)
#     # y1 = b.view(1, -1)
#     corr_factor = stats.kendalltau(x1, y1)
#     return corr_factor


#
#
# class BFF(nn.Module):
#     def __init__(self,inchannel=64):
#         super(BFF, self).__init__()
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.conv_3 = nn.Sequential(
#             nn.Conv2d(inchannel, inchannel, kernel_size=1, padding=0))#, nn.BatchNorm2d(inchannel), nn.PReLU())
#         # self.conv_4 = nn.Sequential(
#         #     nn.AdaptiveAvgPool2d((1,1)),
#         #     nn.Conv2d(inchannel, inchannel//4, kernel_size=1, padding=0),
#         #     nn.ReLU(),
#         #     nn.Conv2d(inchannel//4, inchannel, kernel_size=1, padding=0)
#         #     )
#         self.conv_4 = nn.Sequential(
#
#             nn.Conv2d(2*inchannel, inchannel, kernel_size=3, padding=1),
#             nn.ReLU()
#
#             )
#
#
#     def forward(self, x, y):
#         # print("x",x.shape)
#         # print("y", y.shape)
#
#         x_1 = torch.cat((x,y),dim=1)
#         x_2 = self.conv_4(x_1)
#         scp = x+y+x_2
#         scp_1 = self.conv_3(scp)
#         scp_2 = scp + scp_1
#         scp_3 = scp_2 * scp
#         # scp_1 = self.conv_4(scp)
#         # scp_1 = torch.sigmoid(scp)
#         # print("scp_1",scp_1.shape)
#         # print("scp",scp.shape)
#         # scp_2 = torch.multiply(scp,scp_1)
#         # scp_2 = scp * scp_1
#         # scp_3 = torch.add(scp_2,scp)
#         # scp_4 = torch.subtract(scp_3,scp)
#         # print("scp_4",scp_4.shape)
#         # print("scp", scp.shape)
#
#         # scp_5 = torch.cat((scp_4,scp),dim=1)
#         # scp_6 = self.conv_4(scp_5)
#         return scp_3

class BFF(nn.Module):
    def __init__(self,inchannel=64):
        super(BFF, self).__init__()
        #self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_3 = nn.Sequential(
            nn.Conv2d(inchannel, inchannel, kernel_size=3, padding=1))#, nn.BatchNorm2d(inchannel), nn.PReLU())

        self.conv_4 = nn.Sequential(

            nn.Conv2d(2*inchannel, inchannel, kernel_size=1, padding=0),
            nn.ReLU()

            )


    def forward(self, x, y):
        x_1 = torch.add(x,y)
        x_2 = self.conv_3(x_1)
        scp = x+y+x_2
        # scp_1 = self.conv_4(scp)
        scp_1 = torch.sigmoid(scp)
        # print("scp_1",scp_1.shape)
        # print("scp",scp.shape)
        scp_2 = torch.multiply(scp,scp_1)
        # scp_2 = scp * scp_1
        scp_3 = torch.add(scp_2,scp)
        scp_4 = torch.subtract(scp_3,scp)
        # print("scp_4",scp_4.shape)
        # print("scp", scp.shape)

        scp_5 = torch.cat((scp_4,scp),dim=1)
        scp_6 = self.conv_4(scp_5)
        return scp_6


def calc_corr(a, b):
    a_avg = a.mean()
    b_avg = b.mean()

    # 计算分子，协方差————按照协方差公式，本来要除以n的，由于在相关系数中上下同时约去了n，于是可以不除以n
    cov_ab = ((a-a_avg) * (b - b_avg)).sum()
    # cov_ab = sum([(x - a_avg) * (y - b_avg) for x, y in zip(a, b)])

    # 计算分母，方差乘积————方差本来也要除以n，在相关系数中上下同时约去了n，于是可以不除以n
    sq = math.sqrt(((a-a_avg) ** 2).sum() * ((b-b_avg) ** 2).sum())
    # sq = math.sqrt(sum([(x - a_avg) ** 2 for x in a]) * sum([(x - b_avg) ** 2 for x in b]))

    corr_factor = cov_ab / sq
    return corr_factor

class Kendalltau_Gate(nn.Module):
    def __init__(self,is_more,inchannel=64):
        super(Kendalltau_Gate, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(128,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.inchannel = inchannel
        self.is_more = is_more
        # self.PAM = PAM(inchannel)
        self.SPM = SPM(inchannel, is_more=0)
        # self.SPM2 = SPM(inchannel,is_more=0)
        self.CAM = CAM(inchannel)
        self.row_attention1 = RowAttention(inchannel, inchannel)
        self.row_attention2 = RowAttention(inchannel, inchannel)
        # self.up2 = nn.Upsample(scale_factor=2,mode='bilinear', align_corners=True)
        # self.is_up = is_up


    def forward(self, *x):

        z=x[0]
        if self.is_more:
            y=torch.cat((x[1],x[2]),dim=1)
            y=self.conv1(y)
        else:
            y=x[1]
        # z = self.row_attention1(z)
        # y = self.row_attention2(y)
        out = z + y
        return out

class MFB(nn.Module):
    def __init__(self,inchannel):
        super(MFB, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.PAM=PAM(inchannel)
        # self.SPM=SPM(inchannel,is_more=0)
        # # self.SPM2 = SPM(inchannel,is_more=0)
        # self.CAM = CAM(inchannel)
        # self.row_attention1 = RowAttention(inchannel,inchannel)
        # self.row_attention2 = RowAttention(inchannel , inchannel)
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(inchannel*2,inchannel,kernel_size=3,padding=1),
        #     nn.BatchNorm2d(inchannel),
        #     nn.ReLU(inplace=True)
        # )
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(inchannel * 3, inchannel, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(inchannel),
        #     nn.ReLU(inplace=True)
        # )
        self.conv4 = nn.Sequential(
            nn.Conv2d(inchannel , inchannel, kernel_size=(1,7), padding=(0,3)),
            nn.Conv2d(inchannel , inchannel, kernel_size=(7, 1), padding=(3, 0)),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(inchannel , inchannel, kernel_size=(1, 7), padding=(0, 3)),
            nn.Conv2d(inchannel , inchannel, kernel_size=(7, 1), padding=(3, 0)),
            # nn.Conv2d(inchannel, inchannel, kernel_size=(7, 1), padding=(2, 1)),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True)
        )
    def forward(self, *x):
        if len(x)==2:
            x1=x[0]
            x2=x[1]
            x3=0
        else:
            x1 = x[0]
            x2 = x[1]
            x3 = x[2]
            x3 = self.upsample(x3)
            # print("x3", x3.shape)
            # print("x2", x2.shape)
            # print("x1", x1.shape)
            # print("x32", x3_1.shape)
            # x1 = self.conv_33(x1)
            # x2 = self.conv_33(x2)
            # x3 = self.conv_33(x3_1)
            # print("x3",x3.shape)
            # print("x3_1",x3_1.shape)
            # print("x2", x2.shape)
            # print("x1", x1.shape)
        D1 = torch.add(x1,x2)
        D2 = torch.multiply(x1, x2)
        D3 = torch.max(x1, x2)
        D4 = (x1+x2)/2
        D5 = x1 - x2
        # SPM1 = self.SPM(D1,D2,D3)
        # SPM2 = self.SPM2(D4, D5)
        D6 = D1+ D2+ D3
        D7 = D4 + D5
        corr = pytorch_ssim.ssim(D6, D7)
        z_new1 = torch.multiply(D6 , corr)
        y_new1 = torch.sigmoid(D7)
        y_new2 = torch.tanh(D7)

        y_new3 = torch.multiply(y_new1, y_new2)
        y_new4 = torch.multiply(y_new3, (1-corr))
        out = y_new4 +z_new1
        M_out = out + x3


        return M_out
        # else:
        #     x1 = x[0]
        #     x2 = x[1]
        #     # x1 = self.conv_33(x1)
        #     # x2 = self.conv_33(x2)
        #     # print("x22", x2.shape)
        #     # print("x11", x1.shape)
        #     D1 = torch.add(x1, x2)
        #     D2 = torch.multiply(x1, x2)
        #     D3 = torch.max(x1, x2)
        #     D4 = (x1 + x2) / 2
        #     D5 = x1 - x2
        #     D6 = torch.cat((D1, D2, D3, D4, D5), dim=1)
        #     #D5_1 = self.conv_3(D6)
        #     #D5 = torch.cat((D1, D2, D3), dim=1)
        #     # print("D5",D5.shape)
        #     # print("inchannel",self.inchannel)
        #     M_out = self.conv_3(D6)
        #     # M_out = self.upsample(M_out)
        #     # M_out = D5_1 + x3
        #     # print(M_out.shape)
        #     return M_out






class SFNet(nn.Module):
    def __init__(self, pretrained=True, num_class=1,bias = False):
        super(SFNet, self).__init__()


        self.resnet = resnet_left()
        self.resnet_depth = resnet_right()



        self.dem1 = DEM(64)
        self.dem2 = DEM(256)
        self.dem3 = DEM(512)
        self.dem4 = DEM(1024)
        self.dem5 = DEM(2048)

        self.MFB1 = MFB(64)
        self.MFB2 = MFB(256)
        self.MFB3 = MFB(512)
        self.MFB4 = MFB(1024)
        self.MFB5 = MFB(2048)

        self.BFF1 = BFF()
        self.BFF2 = BFF()
        self.BFF3 = BFF()
        self.BFF4 = BFF()
        self.BFF5 = BFF()
        self.KGM1 = Kendalltau_Gate(0,64)
        self.KGM2 = Kendalltau_Gate(1,64)
        self.KGM3 = Kendalltau_Gate(1,64)
        self.KGM4 = Kendalltau_Gate(1,64)

        self.conv_2048_1024 = nn.Sequential(nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1, bias=bias),
                                         nn.BatchNorm2d(1024),
                                         nn.ReLU(inplace=True)
                                         )
        self.conv_1024_512 = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, bias=bias),
                                            nn.BatchNorm2d(512),
                                            nn.ReLU(inplace=True)
                                            )
        self.conv_512_256 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=bias),
                                           nn.BatchNorm2d(256),
                                           nn.ReLU(inplace=True)
                                           )
        self.conv_256_64 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1, bias=bias),
                                          nn.BatchNorm2d(64),
                                          nn.ReLU(inplace=True)
                                          )

        self.conv_256_64 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1, bias=bias),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU(inplace=True)
                                         )


        self.conv_512_64 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=3, stride=1, padding=1, bias=bias),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU(inplace=True)
                                         )
        self.conv_1024_64 = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=3, stride=1, padding=1, bias=bias),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU(inplace=True)
                                         )
        self.conv_2048_64 = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=3, stride=1, padding=1, bias=bias),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU(inplace=True)
                                         )



        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample32 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
        self.conv_1x1_output = nn.Conv2d(64, 1, 1, 1)
        self.conv_256_1_output = nn.Conv2d(256, 1, 1, 1)
        self.conv_512_1_output = nn.Conv2d(512, 1, 1, 1)
        self.conv_1024_1_output = nn.Conv2d(1024, 1, 1, 1)
        self.conv_2048_1_output = nn.Conv2d(2048, 1, 1, 1)
    def forward(self, left, depth):

        x = self.resnet.conv1(left)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x_l = self.resnet.maxpool(x)
        # print("Res50",self.resnet_depth)
        x_depth = self.resnet_depth.conv1(depth)
        x_depth = self.resnet_depth.bn1(x_depth)
        x_depth = self.resnet_depth.relu(x_depth)
        x_depth_l = self.resnet_depth.maxpool(x_depth)
        # print("x_l",x_l.shape)
        x_depth = self.dem1(x_depth)
        # print("x_depth2", x_depth.shape)
        # x_depth_l = self.dem1(x_depth_l)
        x_l= x
        lf1 = self.resnet.layer1(x_l)
        # lf1 = self.resnet.layer1(x_l)  # 256 x 64 x 64
        rf1 = self.resnet_depth.layer1(x_depth)
        # print("lf1",lf1.shape)
        rf1 = self.dem2(rf1)
        # print("rf1", rf1.shape)
        lf2 = self.resnet.layer2(lf1)  # 256 x 56 x 56
        # print("lf2",lf2.shape)
        rf2 = self.resnet_depth.layer2(rf1)

        rf2 = self.dem3(rf2)
        # print("lf2",lf2.shape)
        # print("rf2",rf2.shape)
        lf3 = self.resnet.layer3(lf2)  # 256 x 64 x 64
        rf3 = self.resnet_depth.layer3(rf2)
        # print("lf3", lf3.shape)
        rf3 = self.dem4(rf3)

        # print("lf3", lf3.shape)
        lf4 = self.resnet.layer4(lf3)  # 256 x 64 x 64
        # print("lf4",lf4.shape)
        rf4 = self.resnet_depth.layer4(rf3)
        # F1 = rf4
        # print("rf41",rf4.shape)
        rf4 = self.dem5(rf4)
        # print("rf42", rf4.shape)
        # F1 = rf4
        # #
        # print("rf4",rf4.shape)
        # print("rf3", rf3.shape)
        # print("rf2", rf2.shape)
        # print("rf1", rf1.shape)
        # # print("x", x.shape)
        # print("lf4",lf4.shape)
        # print("rf4", rf4.shape)
        MFB_5_r = lf4 + rf4
        #csm_5_r = self.upsample2(csm_5_r)
        # print("MFB_5_r",MFB_5_r.shape)
        MFB_5_r_2 = self.conv_2048_1024(MFB_5_r)
        # print("csm_5_r",csm_5_r_2.shape)
        # MFB_5_r_2 = self.upsample2(MFB_5_r_2)
        # print("csm_5_r_2",MFB_5_r_2.shape)
        # print("rf3", rf3.shape)
        # print("lf3", lf3.shape)
        # MFB_4_r = self.MFB4(lf3, rf3)
        # print("lf3",lf3.shape)
        # print("rf3",rf3.shape)
        # print("MFB_5_r_2",MFB_5_r_2.shape)
        MFB_4_r = self.MFB4(lf3, rf3, MFB_5_r_2)
        # print("csm_4_r_2",csm_4_r.shape)
        MFB_4_r_2 = self.conv_1024_512(MFB_4_r)
        # MFB_4_r_2 = self.upsample2(MFB_4_r_2)

        # print(lf2.shape)
        # print(rf2.shape)
        # print("csm_4_r_2",csm_4_r_2.shape)

        MFB_3_r = self.MFB3(lf2, rf2, MFB_4_r_2)
        MFB_3_r_2 = self.conv_512_256(MFB_3_r)
        #MFB_3_r_2 = self.upsample2(MFB_3_r_2)
        # print("csm_3_r_2",csm_3_r_2.shape)
        # print("lf2",lf2.shape)
        # print("rf2", rf2.shape)
        MFB_2_r = self.MFB2(lf1, rf1, MFB_3_r_2)
        # print("MFB_2_r",MFB_2_r.shape)
        MFB_2_r_2 = self.conv_256_64(MFB_2_r)
        # MFB_2_r_2 = self.upsample2(MFB_2_r_2)
        # MFB_2_r_2 = MFB_2_r
        # print("x", x.shape)
        # print("x_depth", x_depth.shape)
        # print("MFB_2_r_2", MFB_2_r_2.shape)
        MFB_1_r = self.MFB1(x,x_depth,MFB_2_r_2)
        # print("csm_1_r",csm_1_r.shape)
        # print("csm_1_r",csm_1_r.shape)

        # print("csm_2_r", csm_2_r.shape)
        # print("lf3",lf3.shape)
        # print("rf3",rf3.shape)

        # print("lf4", lf4.shape)
        # print("rf4", rf4.shape)

        # print("csm_4_r", csm_4_r.shape)
        # print("lf5", lf5.shape)
        # print("rf5", rf5.shape)

        #print(csm_1_r.shape)
        # M_1 =MFB_1_r
       # M_1 = self.conv_M64_128(M_1)
        # print("M_1", M_1.shape)
        M_2 = MFB_2_r
        # print("M2",M_2.shape)
        #print("M_2",M_2.shape)
        M_2 = self.conv_256_64(M_2)
        # print("M_2_after", M_2.shape)
        M_2 = self.upsample2(M_2)
        # print("csm_3_r",csm_3_r.shape)
        M_3 = self.upsample4(MFB_3_r)
        M_3 = self.conv_512_64(M_3)
        # M_3 = self.conv_512_64(M_3)
        # print("M_3", M_3.shape)
        #print("M_3.shape",M_3.shape)
        # M_4 = self.upsample8(MFB_4_r)
        #
        # print(csm_4_r.shape)
        M_4 = self.conv_1024_64(MFB_4_r)
        M_4 = self.upsample8(M_4)
        # M_4 = self.conv_1024_64(M_4)
        # print("M_4",M_4.shape)

        # print("M_5",M_5.shape)
        M_5 = self.conv_2048_64(MFB_5_r)
        M_5 = self.upsample16(M_5)
        # M_5 = self.conv_2048_64(M_5)
        # print("M_5",M_5.shape)
        #M_Edge_1 = MFB_1_r
        #MFB_1_r = self.upsample2(MFB_1_r)
        M_Edge_2 = MFB_1_r
        M_Edge_3 = MFB_1_r
        M_Edge_4 = MFB_1_r
        M_Edge_5 = MFB_1_r

        #print(csm_1_r.shape)
        # print("M_Edge_5",M_Edge_5.shape)
        # print("M_5",M_5.shape)
        # print("M_Edge_5",M_Edge_5.shape)
        # print("M_5",M_5.shape)
        # print(csm_2_r_2.shape)
        # S_5 = M_Edge_5 + M_5
        S_5 = self.BFF1(M_Edge_5, M_5)
        # print("M_Edge_4",M_Edge_4.shape)
        # print("M_4",M_4.shape)

        S_4 = self.BFF2(M_Edge_4,M_4)
        # S_4 = M_Edge_4 + M_4
        S_3 = self.BFF3(M_Edge_3, M_3)
        # S_3 = M_Edge_3 + M_3
        # print(M_Edge_2.shape)
        # print(M_2.shape)
        S_2 = self.BFF4(M_Edge_2, M_2)




        # print('S_5',S_5.shape)
        KGM1 = self.KGM1(S_5,M_5)
        KGM2 = self.KGM2(KGM1,S_4,M_5)
        KGM3 = self.KGM3(KGM2,S_3,M_5)
        KGM4 = self.KGM4(KGM3,S_2,M_5)
        # out3 = out1 + S_2
        # print("out3",out3.shape)
        out = self.upsample2(KGM4)
        out = self.conv_1x1_output(out)
        out4 = self.conv_1x1_output (MFB_1_r)
        out4 = self.upsample2(out4)
        out5 = self.conv_256_1_output(MFB_2_r)
        out5 = self.upsample4(out5)
        # print("MFB_3_r",MFB_3_r.shape)
        out6 = self.conv_512_1_output(MFB_3_r)
        # print("out6",out6.shape)
        out6 = self.upsample8(out6)
        out7 = self.conv_1024_1_output(MFB_4_r)
        out7 = self.upsample16(out7)
        out8 = self.conv_2048_1_output(MFB_5_r)
        out8 = self.upsample32(out8)
        # out = torch.sigmoid(out)
        # print("MFB_1_r", MFB_1_r.shape)
        # print("MFB_2_r", MFB_2_r.shape)
        # print("MFB_3_r", MFB_3_r.shape)
        # print("MFB_4_r", MFB_4_r.shape)
        # print("MFB_5_r",MFB_5_r.shape)

        if self.training == True:
            return out, out4, out5, out6, out7, out8
        else:
            return out





        # print("lf5",lf5.shape)

        # if self.training:
        #     return out, out1, out2, out3, out4, out5
        # return out




if __name__ == '__main__':
    image = torch.randn(1, 3, 224, 224)
    ndsm = torch.randn(1, 3, 224, 224)
    ndsm1 = torch.randn(1, 3, 224, 224)
    net = SFNet()
    # from FLOP import CalParams
    # CalParams(net,image,ndsm)
    # FLOPs: 18.307
    # G
    # Params: 74.751
    # M
    out = net(image, ndsm)
    for i in out:
        print(i.shape)
    # print(vgg_right)
    # for i in out:
    #     print(i.shape)
    # net = CSM(1)
    # out = net(image, ndsm, ndsm1)
    # print(out.shape)
    # image = torch.randn(1, 128, 224, 224)
    # ndsm = torch.randn(1, 128, 224, 224)
    # net = BFF()
    # out = net(image, ndsm)
    # print(out.shape)






