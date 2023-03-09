import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.nn.functional import interpolate

resnet = models.resnet50(pretrained=False)


class ResBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(ResBlock, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=(3, 3), stride=(stride, stride),
                               padding=(1, 1), bias=False)
        self.bn0 = nn.BatchNorm2d(planes)

        self.conv1 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=(3, 3), stride=(stride, stride),
                               padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.short_cut = nn.Sequential(nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=(3, 3),
                                                 stride=(stride, stride), padding=(1, 1), bias=False),
                                       nn.BatchNorm2d(planes))

    def forward(self, x):
        out = F.relu(self.bn0(self.conv0(x)))
        out = self.bn1(self.conv1(out))
        out += self.short_cut(x)
        out = F.relu(out)

        return out


class Linnet(nn.Module):
    def __init__(self, resnet=resnet):
        super(Linnet, self).__init__()
        self.resnet = resnet
        self.conv0 = nn.Conv2d(in_channels=8, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                               bias=False)
        self.bn0 = nn.BatchNorm2d(num_features=64)
        self.relu0 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.layer0 = nn.Sequential(self.conv0, self.bn0, self.relu0, self.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4  # w=1/16, h=1/16, channel=2048

        # self.upx2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upx2 = interpolate

        self.cat_residual_block_1 = ResBlock(2048 + 1024, 1024)
        self.cat_residual_block_2 = ResBlock(1024 + 512, 512)
        self.cat_residual_block_3 = ResBlock(512 + 256, 256)
        self.cat_residual_block_4 = ResBlock(256 + 64, 128)

        self.cat_residual_block_5 = ResBlock(128, 64)
        self.conv_out = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                  bias=False)
        self.bn_out = nn.BatchNorm2d(num_features=1)

    def get_out_map(self, x):
        # print(type(x))
        # print('-x in net', x.shape)
        # N, C ,W, H
        fb_map = x[:, 8, :, :]
        out_map = fb_map

        return out_map

    def forward(self, x):  # [b, h, w, 9]
        out_map = self.get_out_map(x)
        x = x[:, :-1, :, :]

        x_0 = self.layer0(x)
        x_1 = self.layer1(x_0)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)

        # print(x.shape)      # [4, 8, 1224, 370]
        # print(x_0.shape)    # [4, 64, 305, 92]
        # print(x_1.shape)    # [4, 256, 305, 92]
        # print(x_2.shape)    # [4, 512, 153, 45]
        # print(x_3.shape)    # [4, 1024, 77, 23]
        # print(x_4.shape)    # [4, 2048, 39, 12]
        # print(x_5.shape)

        x_5 = self.upx2(x_4, x_3.shape[2:4])
        x_cat1 = torch.cat((x_5, x_3), 1)
        x_5 = self.cat_residual_block_1(x_cat1)

        x_6 = self.upx2(x_5, x_2.shape[2:4])
        x_cat2 = torch.cat((x_6, x_2), 1)
        x_6 = self.cat_residual_block_2(x_cat2)

        x_7 = self.upx2(x_6, x_1.shape[2:4])
        x_cat3 = torch.cat((x_7, x_1), 1)
        x_7 = self.cat_residual_block_3(x_cat3)

        x_8 = self.upx2(x_7, x_0.shape[2:4])
        x_cat4 = torch.cat((x_8, x_0), 1)
        x_8 = self.cat_residual_block_4(x_cat4)

        # print(x_8.shape)
        out = self.cat_residual_block_5(x_8)
        # print(out.shape)
        # print(out_map.shape)
        out = self.upx2(out, out_map.shape[1:3])
        # print(out.shape)
        out = self.conv_out(out)
        out = self.bn_out(out)
        out = out_map * out

        out = 2 * torch.sigmoid(out) - 1
        res = out.detach().cpu().numpy()
        # np.save('a.npy', res)
        # exit()
        # print(out.shape)
        return out


class Linnet2(nn.Module):
    def __init__(self, resnet=resnet):
        super(Linnet2, self).__init__()
        self.resnet = resnet
        self.conv0 = nn.Conv2d(in_channels=21, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                               bias=False)
        self.bn0 = nn.BatchNorm2d(num_features=64)
        self.relu0 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.layer0 = nn.Sequential(self.conv0, self.bn0, self.relu0, self.maxpool)

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4  # w=1/16, h=1/16, channel=2048

        # self.upx2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upx2 = interpolate

        self.cat_residual_block_1 = ResBlock(2048 + 1024, 1024)
        self.cat_residual_block_2 = ResBlock(1024 + 512, 512)
        self.cat_residual_block_3 = ResBlock(512 + 256, 256)
        self.cat_residual_block_4 = ResBlock(256 + 64, 128)

        self.cls_residual_block = ResBlock(64, 64)
        self.reg_residual_block = ResBlock(64, 64)

        self.cat_residual_block_5 = ResBlock(128, 64)
        ##############################################################################################################
        self.cls_conv_out = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                      bias=False)
        self.cls_bn_out = nn.BatchNorm2d(num_features=2)
        ##############################################################################################################
        self.reg_conv_out = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                      bias=False)
        self.reg_bn_out = nn.BatchNorm2d(num_features=1)
        ##############################################################################################################
        self.ref_conv_out = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                      bias=False)
        self.ref_bn_out = nn.BatchNorm2d(num_features=1)

    def get_out_map(self, x):
        # N, C ,W, H
        uv_map = x[:, 18, :, :]
        out_map = uv_map.unsqueeze(1)

        return out_map

    def forward(self, x):  # [b, h, w, 21]
        out_map = self.get_out_map(x)
        # x = x[:, :-1, :, :]

        x_0 = self.layer0(x)
        x_1 = self.layer1(x_0)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)

        # print(x.shape)      # [4, 8, 1224, 370]
        # print(x_0.shape)    # [4, 64, 305, 92]
        # print(x_1.shape)    # [4, 256, 305, 92]
        # print(x_2.shape)    # [4, 512, 153, 45]
        # print(x_3.shape)    # [4, 1024, 77, 23]
        # print(x_4.shape)    # [4, 2048, 39, 12]
        # print(x_5.shape)

        x_5 = self.upx2(x_4, x_3.shape[2:4])
        x_cat1 = torch.cat((x_5, x_3), 1)
        x_5 = self.cat_residual_block_1(x_cat1)

        x_6 = self.upx2(x_5, x_2.shape[2:4])
        x_cat2 = torch.cat((x_6, x_2), 1)
        x_6 = self.cat_residual_block_2(x_cat2)

        x_7 = self.upx2(x_6, x_1.shape[2:4])
        x_cat3 = torch.cat((x_7, x_1), 1)
        x_7 = self.cat_residual_block_3(x_cat3)

        x_8 = self.upx2(x_7, x_0.shape[2:4])
        x_cat4 = torch.cat((x_8, x_0), 1)
        x_8 = self.cat_residual_block_4(x_cat4)

        # print(x_8.shape)
        out_feature = self.cat_residual_block_5(x_8)
        out_feature = self.upx2(out_feature, out_map.shape[2:])  # B, 64, W, H
        # print(out.shape)

        # branch 1: cls
        cls_feature = self.cls_residual_block(out_feature)
        cls_out = self.cls_conv_out(cls_feature)
        # cls_out = self.cls_bn_out(cls_out)
        cls_out = torch.sigmoid(cls_out)
        cls_out = cls_out * out_map

        # branch 2: dl regression
        reg_feature = self.reg_residual_block(out_feature)
        reg_out = self.reg_conv_out(reg_feature)
        # reg_out = self.reg_bn_out(reg_out)
        # reg_out = 2 * torch.sigmoid(reg_out) - 1    #
        reg_out = reg_out * out_map
        # print(torch.unique(reg_out))
        # exit()

        # # branch 3: reflection regression
        # ref_feature = self.reg_residual_block(out_feature)
        # ref_out = self.ref_conv_out(ref_feature)
        # ref_out = self.reg_bn_out(ref_out)
        # ref_out = torch.sigmoid(ref_out)
        # ref_out = ref_out * out_map

        output = torch.cat([cls_out, reg_out], dim=1)  # B C W H
        # print(cls_map.shape, reg_map.shape)
        # print(output.shape)

        return output

