import torch
import torch.nn as nn
import torch.nn.functional as F
# from toolbox.models.JJNet13.EANet import External_attention

class Gate(nn.Module):
    def __init__(self, in_channels):
        super(Gate, self).__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.gate(x)

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

    def forward(self, x):
        # [N, C, H, W]
        b, c, h, w = x.size()
        # [N, C // ratio, H*W]
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
        self.gmma = nn.Parameter(torch.zeros(1))

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
        out = mask * self.gmma + x
        return out


class GRU(nn.Module):
    def __init__(self, in_channels):
        super(GRU, self).__init__()
        self.gate = Gate(128)
        # self.CAM = CAM(256*2)
        # self.PAM = PAM(256*2)
        self.ea = External_attention(128*2)
        self.conv = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels, 128, kernel_size=1, stride=1, padding=0)
        self.tanh = nn.Sequential(
            nn.Conv2d(128*2, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.Tanh()
        )

    def forward(self, irrg, dsm):
        # print(irrg.shape)
        # print(dsm.shape)
        b, c, h, w = irrg.size()
        irrg = self.conv1(irrg)
        dsm = self.conv2(dsm)
        Xcom = torch.cat([irrg, dsm], dim=1)
        G = dsm * self.gate(dsm)
        print(G.shape)
        print(irrg.shape)
        out1 = self.conv(G * irrg) + irrg
        T = self.tanh(self.ea(Xcom))
        # one = torch.ones([b, 128, h, w]).cuda()
        one = torch.ones([b, 128, h, w])
        out2 = T - (one - G)
        out = torch.add(out1, out2)
        return out


if __name__ == "__main__":
    image = torch.randn(4, 64, 8, 8)
    depth = torch.randn(4, 64, 1, 1)
    # model = GRU(64)
    # out = model(image, depth)
    out = image * depth
    print(out.shape)