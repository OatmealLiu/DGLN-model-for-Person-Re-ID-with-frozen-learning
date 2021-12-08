import torch
from torch import nn
from torch.nn import functional as F


class RGA_Module(nn.Module):
    def __init__(self, in_channel, in_spatial, cha_ratio=8, spa_ratio=8, down_ratio=8):
        super(RGA_Module, self).__init__()

        self.in_channel = in_channel
        self.in_spatial = in_spatial

        self.inter_channel = in_channel // cha_ratio
        self.inter_spatial = in_spatial // spa_ratio

        # Embedding functions for original features
        self.gx_spatial = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_channel),
            nn.ReLU())
        self.gx_channel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_spatial),
            nn.ReLU())

        # Embedding functions for relation features
        self.gg_spatial = nn.Sequential(
            nn.Conv2d(in_channels=self.in_spatial * 2, out_channels=self.inter_spatial,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_spatial),
            nn.ReLU())
        self.gg_channel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel * 2, out_channels=self.inter_channel,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_channel),
            nn.ReLU())

        # Networks for learning attention weights
        num_channel_s = 1 + self.inter_spatial
        self.W_spatial = nn.Sequential(
            nn.Conv2d(in_channels=num_channel_s, out_channels=num_channel_s // down_ratio,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_channel_s // down_ratio),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_channel_s // down_ratio, out_channels=1,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1))
        num_channel_c = 1 + self.inter_channel
        self.W_channel = nn.Sequential(
            nn.Conv2d(in_channels=num_channel_c, out_channels=num_channel_c // down_ratio,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_channel_c // down_ratio),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_channel_c // down_ratio, out_channels=1,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1))

        # Embedding functions for modeling relations
        self.theta_spatial = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_channel),
            nn.ReLU())
        self.phi_spatial = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_channel),
            nn.ReLU())

        self.theta_channel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_spatial),
            nn.ReLU())
        self.phi_channel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_spatial),
            nn.ReLU())

    def forward(self, x):
        b, c, h, w = x.size()

        # ---Spatial attention---
        # print(x.shape)
        theta_xs = self.theta_spatial(x)  # 8 32 64 32
        # print(theta_xs.shape)
        phi_xs = self.phi_spatial(x)  # 8 32 64 32
        # print(phi_xs.shape)
        theta_xs = theta_xs.view(b, self.inter_channel, -1)  # 8 32 64*32
        # print(theta_xs.shape)
        theta_xs = theta_xs.permute(0, 2, 1)  # 8 64*32 32
        # print(theta_xs.shape)
        phi_xs = phi_xs.view(b, self.inter_channel, -1)  # 8 32 64*32
        # print(phi_xs.shape)
        Gs = torch.matmul(theta_xs, phi_xs)  # 8 2048 2048
        # print(Gs.shape)
        Gs_in = Gs.permute(0, 2, 1).view(b, h * w, h, w)  # 8 2048 64 32 调换下顺序
        # print(Gs_in.shape)
        Gs_out = Gs.view(b, h * w, h, w)  # 8 2048 64 32
        # print(Gs_out.shape)
        Gs_joint = torch.cat((Gs_in, Gs_out), 1)  # 8 4096 64 32
        # print(Gs_joint.shape)
        Gs_joint = self.gg_spatial(Gs_joint)  # 8 256 64 32
        # print(Gs_joint.shape)

        g_xs = self.gx_spatial(x)  # 8 32 64 32
        # print(g_xs.shape)
        g_xs = torch.mean(g_xs, dim=1, keepdim=True)  # 8 1 64 32
        # print(g_xs.shape)
        ys = torch.cat((g_xs, Gs_joint), 1)  # 8 257 64 32
        # print(ys.shape)

        W_ys = self.W_spatial(ys)  # 8 1 64 32
        x = F.sigmoid(W_ys.expand_as(x)) * x
        # print(x.shape) # 8 256 64 32

        # ---Channel attention---
        xc = x.view(b, c, -1).permute(0, 2, 1).unsqueeze(-1)  # 8 2048 256 1
        # print(xc.shape)
        theta_xc = self.theta_channel(xc).squeeze(-1).permute(0, 2, 1)  # 8 256 256
        # print(theta_xc.shape)
        phi_xc = self.phi_channel(xc).squeeze(-1)  # 8 256 256
        # print(phi_xc.shape)
        Gc = torch.matmul(theta_xc, phi_xc)  # 8 256 256
        # print(Gc.shape)
        Gc_in = Gc.permute(0, 2, 1).unsqueeze(-1)  # 8 256 256 1
        # print(Gc_in.shape)
        Gc_out = Gc.unsqueeze(-1)  # 8 256 256 1
        # print(Gc_out.shape)
        Gc_joint = torch.cat((Gc_in, Gc_out), 1)  # 8 512 256 1
        # print(Gc_joint.shape)
        Gc_joint = self.gg_channel(Gc_joint)  # 8 32 256 1
        # print(Gc_joint.shape)

        g_xc = self.gx_channel(xc)  # 8 256 256 1
        # print(g_xc.shape)
        g_xc = torch.mean(g_xc, dim=1, keepdim=True)  # 8 1 256 1
        # print(g_xc.shape)
        yc = torch.cat((g_xc, Gc_joint), 1)  # 8 33 256 1
        # print(yc.shape)
        W_yc = self.W_channel(yc).transpose(1, 2)  # 8 256 1 1 得到权重分配
        # print(W_yc.shape)
        out = F.sigmoid(W_yc) * x
        # print(out.shape)

        return out
