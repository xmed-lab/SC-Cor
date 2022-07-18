import torch
import torch.nn.functional as F
from torch import nn
# import torch.nn as nn
from resnext.resnext101_regular import ResNeXt101
loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)


class DenseCLNeck(nn.Module):
    '''The non-linear neck in DenseCL.
        Single and dense in parallel: fc-relu-fc, conv-relu-conv
    '''
    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 num_grid=7):
        super(DenseCLNeck, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hid_channels), nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels))

        self.with_pool = num_grid != None
        if self.with_pool:
            self.pool = nn.AdaptiveAvgPool2d((num_grid, num_grid))
        self.mlp2 = nn.Sequential(
            nn.Conv2d(in_channels, hid_channels, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hid_channels, out_channels, 1))
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x):
        # print(len(x))
        # assert len(x) == 1
        # x = x[0]

        avgpooled_x = self.avgpool(x)
        avgpooled_x = self.mlp(avgpooled_x.view(avgpooled_x.size(0), -1))

        if self.with_pool:
            x = self.pool(x) # sxs
        x = self.mlp2(x) # sxs: bxdxsxs
        avgpooled_x2 = self.avgpool2(x) # 1x1: bxdx1x1
        x = x.view(x.size(0), x.size(1), -1) # bxdxs^2
        avgpooled_x2 = avgpooled_x2.view(avgpooled_x2.size(0), -1) # bxd
        return [avgpooled_x, x, avgpooled_x2]

class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(64, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 1, bias=False), nn.BatchNorm2d(64)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 1, bias=False), nn.BatchNorm2d(32)
        )

        
    def forward(self, x):
        block1 = F.relu(self.block1(x) + x, True)
        block2 = self.block2(block1)

        return block2


class AttentionModule(nn.Module):
    def __init__(self):
        super(AttentionModule, self).__init__()
        self.att = nn.Sequential(
            nn.Conv2d(64, 1, 3, bias=False, padding=1), nn.BatchNorm2d(1), nn.Sigmoid()
        )

    def forward(self, x):
        block1 = self.att(x)
        block2 = block1.repeat(1, 32, 1, 1)

        return block2
        
class PlainDSD(nn.Module):
    def __init__(self,fixedBN,freeze_bn_affine):
        super(PlainDSD, self).__init__()
        resnext = ResNeXt101()
        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4
        self.fixedBN = fixedBN
        self.freeze_bn_affine = freeze_bn_affine
        self.down4 = nn.Sequential(
            nn.Conv2d(2048, 512, 3, bias=False, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, bias=False, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(512, 256, 3, bias=False, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, bias=False, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU()
        )

        self.down0 = nn.Sequential(
            nn.Conv2d(64, 64, 3, bias=False, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU()
        )

        self.shad_att = nn.Sequential(
            nn.Conv2d(32, 32, 3, bias=False, padding=1), nn.BatchNorm2d(32), nn.ReLU()
        )

        self.dst1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, bias=False, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, bias=False, padding=1), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.dst2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, bias=False, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, bias=False, padding=1), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.refine4_hl = ConvBlock()
        self.refine3_hl = ConvBlock()
        self.refine2_hl = ConvBlock()
        self.refine1_hl = ConvBlock()

        self.refine0_hl = ConvBlock()

        self.attention4_hl = AttentionModule()
        self.attention3_hl = AttentionModule()
        self.attention2_hl = AttentionModule()
        self.attention1_hl = AttentionModule()
        self.attention0_hl = AttentionModule()
        self.conv1x1_ReLU_down4 = nn.Sequential(
            nn.Conv2d(32, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 1, 1, bias=False)
        )

        self.conv1x1_ReLU_down3 = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 1, 1, bias=False)
        )

        self.conv1x1_ReLU_down2 = nn.Sequential(
            nn.Conv2d(96, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 1, 1, bias=False)
        )

        self.conv1x1_ReLU_down1 = nn.Sequential(
            nn.Conv2d(128, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 1, 1, bias=False)
        )

        self.conv1x1_ReLU_down0 = nn.Sequential(
            nn.Conv2d(160, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 1, 1, bias=False)
        )

        self.fuse_predict = nn.Sequential(
            nn.Conv2d(5, 1, 1, bias=False)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d([17,17])
        # self.head = 
        # self.head = DenseCLNeck(304,all_channel,all_channel,7)
   

    def forward(self, x, val=False):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        down4 = self.down4(layer4)
        down3 = self.down3(layer3)
        down2 = self.down2(layer2)
        down1 = self.down1(layer1)
        down0 = self.down0(layer0)

        down4_dst1 = self.dst1(down4)
        down4_dst1_3 = F.upsample(down4_dst1,size=down3.size()[2:], mode='bilinear')
        # down4_dst1_2 = F.upsample(down4_dst1,size=down2.size()[2:], mode='bilinear')
        # down4_dst1_1 = F.upsample(down4_dst1,size=down1.size()[2:], mode='bilinear')
        # down4_dst1_0 = F.upsample(down4_dst1,size=down0.size()[2:], mode='bilinear')


        down4_dst2 = self.dst2(down4)
        down4_dst2_3 = F.upsample(down4_dst2,size=down3.size()[2:], mode='bilinear')
        # down4_dst2_2 = F.upsample(down4_dst2,size=down2.size()[2:], mode='bilinear')
        # down4_dst2_1 = F.upsample(down4_dst2,size=down1.size()[2:], mode='bilinear')
        # down4_dst2_0 = F.upsample(down4_dst2,size=down0.size()[2:], mode='bilinear')
        down4_shad = down4

        down4_shad = (1 + self.attention4_hl(torch.cat((down4_shad, down4_dst2), 1))) * down4_shad
        down4_shad = F.relu(-self.refine4_hl(torch.cat((down4_shad, down4_dst1), 1)) + down4_shad, True)


        down4_shad_3 = F.upsample(down4_shad,size=down3.size()[2:], mode='bilinear')
        down4_shad_2 = F.upsample(down4_shad,size=down2.size()[2:], mode='bilinear')
        down4_shad_1 = F.upsample(down4_shad,size=down1.size()[2:], mode='bilinear')
        down4_shad_0 = F.upsample(down4_shad,size=down0.size()[2:], mode='bilinear')
        # up_down4_dst1 = self.conv1x1_ReLU_down4(down4_dst1)
        # up_down4_dst2 = self.conv1x1_ReLU_down4(down4_dst2)
        ### feat
        up_down4_feat = self.conv1x1_ReLU_down4[:1](down4_shad)
        up_down4_shad = self.conv1x1_ReLU_down4[1:](up_down4_feat)
        up_down4_feat = down4_shad
        #
        # pred_down4_dst1 = F.upsample(up_down4_dst1,size=x.size()[2:], mode='bilinear')
        # pred_down4_dst2 = F.upsample(up_down4_dst2,size=x.size()[2:], mode='bilinear')
        pred_down4_shad = F.upsample(up_down4_shad,size=x.size()[2:], mode='bilinear')

        down3_dst1 = self.dst1(down3)
        down3_dst2 = self.dst2(down3)
        down3_shad = down3

        down3_shad = (1 + self.attention3_hl(torch.cat((down3_shad, down3_dst2), 1))) * down3_shad
        down3_shad = F.relu(-self.refine3_hl(torch.cat((down3_shad, down3_dst1), 1)) + down3_shad, True)

        # down3_dst1_2 = F.upsample(down3_dst1,size=down2.size()[2:], mode='bilinear')
        # down3_dst1_1 = F.upsample(down3_dst1,size=down1.size()[2:], mode='bilinear')
        # down3_dst1_0 = F.upsample(down3_dst1,size=down0.size()[2:], mode='bilinear')
        # down3_dst2_2 = F.upsample(down3_dst2,size=down2.size()[2:], mode='bilinear')
        # down3_dst2_1 = F.upsample(down3_dst2,size=down1.size()[2:], mode='bilinear')
        # down3_dst2_0 = F.upsample(down3_dst2,size=down0.size()[2:], mode='bilinear')
        down3_shad_2 = F.upsample(down3_shad,size=down2.size()[2:], mode='bilinear')
        down3_shad_1 = F.upsample(down3_shad,size=down1.size()[2:], mode='bilinear')
        down3_shad_0 = F.upsample(down3_shad,size=down0.size()[2:], mode='bilinear')
        
        up_down3_dst1 = self.conv1x1_ReLU_down3(torch.cat((down3_dst1,down4_dst1_3),1))
        up_down3_dst2 = self.conv1x1_ReLU_down3(torch.cat((down3_dst2,down4_dst2_3),1))
        up_down3_shad = self.conv1x1_ReLU_down3(torch.cat((down3_shad,down4_shad_3),1))
        ### feat
        up_down3_feat = self.conv1x1_ReLU_down3[:1](torch.cat((down3_shad,down4_shad_3),1))
        up_down3_shad = self.conv1x1_ReLU_down3[1:](up_down3_feat)
        #
        # pred_down3_dst1 = F.upsample(up_down3_dst1,size=x.size()[2:], mode='bilinear')
        # pred_down3_dst2 = F.upsample(up_down3_dst2,size=x.size()[2:], mode='bilinear')
        pred_down3_shad = F.upsample(up_down3_shad,size=x.size()[2:], mode='bilinear')


        down2_dst1 = self.dst1(down2)
        down2_dst2 = self.dst2(down2)
        down2_shad = down2
        down2_shad = (1 + self.attention2_hl(torch.cat((down2_shad, down2_dst2), 1))) * down2_shad
        down2_shad = F.relu(-self.refine2_hl(torch.cat((down2_shad, down2_dst1), 1)) + down2_shad, True)

        # down2_dst1_1 = F.upsample(down2_dst1,size=down1.size()[2:], mode='bilinear')
        # down2_dst1_0 = F.upsample(down2_dst1,size=down0.size()[2:], mode='bilinear')
        # down2_dst2_1 = F.upsample(down2_dst2,size=down1.size()[2:], mode='bilinear')
        # down2_dst2_0 = F.upsample(down2_dst2,size=down0.size()[2:], mode='bilinear')
        down2_shad_1 = F.upsample(down2_shad,size=down1.size()[2:], mode='bilinear')
        down2_shad_0 = F.upsample(down2_shad,size=down0.size()[2:], mode='bilinear')
        # up_down2_dst1 = self.conv1x1_ReLU_down2(torch.cat((down2_dst1,down3_dst1_2,down4_dst1_2),1))
        # up_down2_dst2 = self.conv1x1_ReLU_down2(torch.cat((down2_dst2,down3_dst2_2,down4_dst2_2),1))
        ### feat
        up_down2_feat = self.conv1x1_ReLU_down2[:1](torch.cat((down2_shad,down3_shad_2,down4_shad_2),1))
        up_down2_shad = self.conv1x1_ReLU_down2[1:](up_down2_feat)
        #
        # pred_down2_dst1 = F.upsample(up_down2_dst1,size=x.size()[2:], mode='bilinear')
        # pred_down2_dst2 = F.upsample(up_down2_dst2,size=x.size()[2:], mode='bilinear')
        pred_down2_shad = F.upsample(up_down2_shad,size=x.size()[2:], mode='bilinear')

        down1_dst1 = self.dst1(down1)
        down1_dst2 = self.dst2(down1)
        down1_shad = down1

        down1_shad = (1 + self.attention1_hl(torch.cat((down1_shad, down1_dst2), 1))) * down1_shad
        down1_shad = F.relu(-self.refine1_hl(torch.cat((down1_shad, down1_dst1), 1)) + down1_shad, True)

        # down1_dst1_0 = F.upsample(down1_dst1, size=down0.size()[2:], mode='bilinear')
        # down1_dst2_0 = F.upsample(down1_dst2, size=down0.size()[2:], mode='bilinear')
        down1_shad_0 = F.upsample(down1_shad, size=down0.size()[2:], mode='bilinear')
        # up_down1_dst1 = self.conv1x1_ReLU_down1(torch.cat((down1_dst1,down2_dst1_1,down3_dst1_1,down4_dst1_1),1))
        # up_down1_dst2 = self.conv1x1_ReLU_down1(torch.cat((down1_dst2,down2_dst2_1,down3_dst2_1,down4_dst2_1),1))
        up_down1_shad = self.conv1x1_ReLU_down1(torch.cat((down1_shad,down2_shad_1,down3_shad_1,down4_shad_1),1))
        ### feat
        up_down1_feat = self.conv1x1_ReLU_down1[:1](torch.cat((down1_shad,down2_shad_1,down3_shad_1,down4_shad_1),1))
        up_down1_shad = self.conv1x1_ReLU_down1[1:](up_down1_feat)
        #
        # pred_down1_dst1 = F.upsample(up_down1_dst1,size=x.size()[2:], mode='bilinear')
        # pred_down1_dst2 = F.upsample(up_down1_dst2,size=x.size()[2:], mode='bilinear')
        pred_down1_shad = F.upsample(up_down1_shad,size=x.size()[2:], mode='bilinear')


        down0_dst1 = self.dst1(down0)
        down0_dst2 = self.dst2(down0)
        down0_shad = down0

        down0_shad = (1 + self.attention0_hl(torch.cat((down0_shad, down0_dst2), 1))) * down0_shad
        down0_shad = F.relu(-self.refine0_hl(torch.cat((down0_shad, down0_dst1), 1)) + down0_shad, True)


        # up_down0_dst1 =self.conv1x1_ReLU_down0(torch.cat((down0_dst1,down1_dst1_0,down2_dst1_0,down3_dst1_0,down4_dst1_0),1))
        # up_down0_dst2 = self.conv1x1_ReLU_down0(torch.cat((down0_dst2,down1_dst2_0,down2_dst2_0,down3_dst2_0,down4_dst2_0),1))
        ### feat
        up_down0_feat = self.conv1x1_ReLU_down0[:1](torch.cat((down0_shad,down1_shad_0,down2_shad_0,down3_shad_0,down4_shad_0),1))
        up_down0_shad = self.conv1x1_ReLU_down0[1:](up_down0_feat)
        #
        # pred_down0_dst1 = F.upsample(up_down0_dst1,size=x.size()[2:], mode='bilinear')
        # pred_down0_dst2 = F.upsample(up_down0_dst2,size=x.size()[2:], mode='bilinear')
        pred_down0_shad = F.upsample(up_down0_shad,size=x.size()[2:], mode='bilinear')


        fuse_pred_shad = self.fuse_predict(torch.cat((pred_down0_shad,pred_down1_shad,pred_down2_shad,pred_down3_shad,pred_down4_shad),1))
        # fuse_pred_dst1 = self.fuse_predict(torch.cat((pred_down0_dst1,pred_down1_dst1,pred_down2_dst1,pred_down3_dst1,pred_down4_dst1),1))
        # fuse_pred_dst2 = self.fuse_predict(torch.cat((pred_down0_dst2,pred_down1_dst2,pred_down2_dst2,pred_down3_dst2,pred_down4_dst2),1))
            # sadasdasdasd
        if self.training or val:
            return fuse_pred_shad, pred_down1_shad, pred_down2_shad, pred_down3_shad, pred_down4_shad, self.avg_pool(up_down0_feat)
        #     return fuse_pred_shad, pred_down1_shad, pred_down2_shad, pred_down3_shad, pred_down4_shad, \
        #     fuse_pred_dst1, pred_down1_dst1, pred_down2_dst1, pred_down3_dst1, pred_down4_dst1,\
        #     fuse_pred_dst2, pred_down1_dst2, pred_down2_dst2, pred_down3_dst2, pred_down4_dst2, \
        #            pred_down0_dst1, pred_down0_dst2, pred_down0_shad
        return  fuse_pred_shad
        return  F.sigmoid(fuse_pred_shad)
        return F.sigmoid(fuse_pred_shad), up_down4_feat, up_down3_feat, up_down2_feat, up_down1_feat, up_down0_feat


    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super(PlainDSD, self).train(mode)
        if self.fixedBN:
            print("Freezing Mean/Var of BatchNorm2D.")
            if self.freeze_bn_affine:
                print("Freezing Weight/Bias of BatchNorm2D.")
        if self.fixedBN:
            for m in self.modules():
                
                if isinstance(m, nn.BatchNorm2d):
                    # print(m)
                    m.eval()
                    
                    if self.freeze_bn_affine:
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
                    # print(m.parameters)

   

class DSDNet(nn.Module):
    
    def __init__(self,args):
        
        super(DSDNet, self).__init__()
        resnext = ResNeXt101()
        self.interation = 0 
        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4
        
        self.fixedBN = args['fixedBN']
        self.freeze_bn_affine = args['freeze_bn_affine']
        self.args = args
        self.down4 = nn.Sequential(
            nn.Conv2d(2048, 512, 3, bias=False, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, bias=False, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(512, 256, 3, bias=False, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, bias=False, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU()
        )

        self.down0 = nn.Sequential(
            nn.Conv2d(64, 64, 3, bias=False, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU()
        )

        self.shad_att = nn.Sequential(
            nn.Conv2d(32, 32, 3, bias=False, padding=1), nn.BatchNorm2d(32), nn.ReLU()
        )

        self.dst1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, bias=False, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, bias=False, padding=1), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.dst2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, bias=False, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, bias=False, padding=1), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.refine4_hl = ConvBlock()
        self.refine3_hl = ConvBlock()
        self.refine2_hl = ConvBlock()
        self.refine1_hl = ConvBlock()

        self.refine0_hl = ConvBlock()

        self.attention4_hl = AttentionModule()
        self.attention3_hl = AttentionModule()
        self.attention2_hl = AttentionModule()
        self.attention1_hl = AttentionModule()
        self.attention0_hl = AttentionModule()
        self.conv1x1_ReLU_down4 = nn.Sequential(
            nn.Conv2d(32, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 1, 1, bias=False)
        )

        self.conv1x1_ReLU_down3 = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 1, 1, bias=False)
        )

        self.conv1x1_ReLU_down2 = nn.Sequential(
            nn.Conv2d(96, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 1, 1, bias=False)
        )

        self.conv1x1_ReLU_down1 = nn.Sequential(
            nn.Conv2d(128, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 1, 1, bias=False)
        )

        self.conv1x1_ReLU_down0 = nn.Sequential(
            nn.Conv2d(160, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 1, 1, bias=False)
        )

        self.fuse_predict = nn.Sequential(
            nn.Conv2d(5, 1, 1, bias=False)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d([args['grid'],args['grid']])

    def dssd_forward(self, x, val=False):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        down4 = self.down4(layer4)
        down3 = self.down3(layer3)
        down2 = self.down2(layer2)
        down1 = self.down1(layer1)
        down0 = self.down0(layer0)

        down4_dst1 = self.dst1(down4)
        down4_dst1_3 = F.upsample(down4_dst1,size=down3.size()[2:], mode='bilinear')
        down4_dst1_2 = F.upsample(down4_dst1,size=down2.size()[2:], mode='bilinear')
        down4_dst1_1 = F.upsample(down4_dst1,size=down1.size()[2:], mode='bilinear')
        down4_dst1_0 = F.upsample(down4_dst1,size=down0.size()[2:], mode='bilinear')


        down4_dst2 = self.dst2(down4)
        down4_dst2_3 = F.upsample(down4_dst2,size=down3.size()[2:], mode='bilinear')
        down4_dst2_2 = F.upsample(down4_dst2,size=down2.size()[2:], mode='bilinear')
        down4_dst2_1 = F.upsample(down4_dst2,size=down1.size()[2:], mode='bilinear')
        down4_dst2_0 = F.upsample(down4_dst2,size=down0.size()[2:], mode='bilinear')
        down4_shad = down4

        down4_shad = (1 + self.attention4_hl(torch.cat((down4_shad, down4_dst2), 1))) * down4_shad
        down4_shad = F.relu(-self.refine4_hl(torch.cat((down4_shad, down4_dst1), 1)) + down4_shad, True)


        down4_shad_3 = F.upsample(down4_shad,size=down3.size()[2:], mode='bilinear')
        down4_shad_2 = F.upsample(down4_shad,size=down2.size()[2:], mode='bilinear')
        down4_shad_1 = F.upsample(down4_shad,size=down1.size()[2:], mode='bilinear')
        down4_shad_0 = F.upsample(down4_shad,size=down0.size()[2:], mode='bilinear')
        up_down4_dst1 = self.conv1x1_ReLU_down4(down4_dst1)
        up_down4_dst2 = self.conv1x1_ReLU_down4(down4_dst2)
        ### feat
        up_down4_feat = self.conv1x1_ReLU_down4[:1](down4_shad)
        up_down4_shad = self.conv1x1_ReLU_down4[1:](up_down4_feat)
        up_down4_feat = down4_shad
        #
        pred_down4_dst1 = F.upsample(up_down4_dst1,size=x.size()[2:], mode='bilinear')
        pred_down4_dst2 = F.upsample(up_down4_dst2,size=x.size()[2:], mode='bilinear')
        pred_down4_shad = F.upsample(up_down4_shad,size=x.size()[2:], mode='bilinear')

        down3_dst1 = self.dst1(down3)
        down3_dst2 = self.dst2(down3)
        down3_shad = down3

        down3_shad = (1 + self.attention3_hl(torch.cat((down3_shad, down3_dst2), 1))) * down3_shad
        down3_shad = F.relu(-self.refine3_hl(torch.cat((down3_shad, down3_dst1), 1)) + down3_shad, True)

        # down3_dst1_2 = F.upsample(down3_dst1,size=down2.size()[2:], mode='bilinear')
        # down3_dst1_1 = F.upsample(down3_dst1,size=down1.size()[2:], mode='bilinear')
        # down3_dst1_0 = F.upsample(down3_dst1,size=down0.size()[2:], mode='bilinear')
        # down3_dst2_2 = F.upsample(down3_dst2,size=down2.size()[2:], mode='bilinear')
        # down3_dst2_1 = F.upsample(down3_dst2,size=down1.size()[2:], mode='bilinear')
        # down3_dst2_0 = F.upsample(down3_dst2,size=down0.size()[2:], mode='bilinear')
        down3_shad_2 = F.upsample(down3_shad,size=down2.size()[2:], mode='bilinear')
        down3_shad_1 = F.upsample(down3_shad,size=down1.size()[2:], mode='bilinear')
        down3_shad_0 = F.upsample(down3_shad,size=down0.size()[2:], mode='bilinear')
        
        up_down3_dst1 = self.conv1x1_ReLU_down3(torch.cat((down3_dst1,down4_dst1_3),1))
        up_down3_dst2 = self.conv1x1_ReLU_down3(torch.cat((down3_dst2,down4_dst2_3),1))
        up_down3_shad = self.conv1x1_ReLU_down3(torch.cat((down3_shad,down4_shad_3),1))
        ### feat
        up_down3_feat = self.conv1x1_ReLU_down3[:1](torch.cat((down3_shad,down4_shad_3),1))
        up_down3_shad = self.conv1x1_ReLU_down3[1:](up_down3_feat)
        #
        # pred_down3_dst1 = F.upsample(up_down3_dst1,size=x.size()[2:], mode='bilinear')
        # pred_down3_dst2 = F.upsample(up_down3_dst2,size=x.size()[2:], mode='bilinear')
        pred_down3_shad = F.upsample(up_down3_shad,size=x.size()[2:], mode='bilinear')


        down2_dst1 = self.dst1(down2)
        down2_dst2 = self.dst2(down2)
        down2_shad = down2
        down2_shad = (1 + self.attention2_hl(torch.cat((down2_shad, down2_dst2), 1))) * down2_shad
        down2_shad = F.relu(-self.refine2_hl(torch.cat((down2_shad, down2_dst1), 1)) + down2_shad, True)

        # down2_dst1_1 = F.upsample(down2_dst1,size=down1.size()[2:], mode='bilinear')
        # down2_dst1_0 = F.upsample(down2_dst1,size=down0.size()[2:], mode='bilinear')
        # down2_dst2_1 = F.upsample(down2_dst2,size=down1.size()[2:], mode='bilinear')
        # down2_dst2_0 = F.upsample(down2_dst2,size=down0.size()[2:], mode='bilinear')
        down2_shad_1 = F.upsample(down2_shad,size=down1.size()[2:], mode='bilinear')
        down2_shad_0 = F.upsample(down2_shad,size=down0.size()[2:], mode='bilinear')
        # up_down2_dst1 = self.conv1x1_ReLU_down2(torch.cat((down2_dst1,down3_dst1_2,down4_dst1_2),1))
        # up_down2_dst2 = self.conv1x1_ReLU_down2(torch.cat((down2_dst2,down3_dst2_2,down4_dst2_2),1))
        ### feat
        up_down2_feat = self.conv1x1_ReLU_down2[:1](torch.cat((down2_shad,down3_shad_2,down4_shad_2),1))
        up_down2_shad = self.conv1x1_ReLU_down2[1:](up_down2_feat)
        #
        # pred_down2_dst1 = F.upsample(up_down2_dst1,size=x.size()[2:], mode='bilinear')
        # pred_down2_dst2 = F.upsample(up_down2_dst2,size=x.size()[2:], mode='bilinear')
        pred_down2_shad = F.upsample(up_down2_shad,size=x.size()[2:], mode='bilinear')

        down1_dst1 = self.dst1(down1)
        down1_dst2 = self.dst2(down1)
        down1_shad = down1

        down1_shad = (1 + self.attention1_hl(torch.cat((down1_shad, down1_dst2), 1))) * down1_shad
        down1_shad = F.relu(-self.refine1_hl(torch.cat((down1_shad, down1_dst1), 1)) + down1_shad, True)

        # down1_dst1_0 = F.upsample(down1_dst1, size=down0.size()[2:], mode='bilinear')
        # down1_dst2_0 = F.upsample(down1_dst2, size=down0.size()[2:], mode='bilinear')
        down1_shad_0 = F.upsample(down1_shad, size=down0.size()[2:], mode='bilinear')
        # up_down1_dst1 = self.conv1x1_ReLU_down1(torch.cat((down1_dst1,down2_dst1_1,down3_dst1_1,down4_dst1_1),1))
        # up_down1_dst2 = self.conv1x1_ReLU_down1(torch.cat((down1_dst2,down2_dst2_1,down3_dst2_1,down4_dst2_1),1))
        up_down1_shad = self.conv1x1_ReLU_down1(torch.cat((down1_shad,down2_shad_1,down3_shad_1,down4_shad_1),1))
        ### feat
        up_down1_feat = self.conv1x1_ReLU_down1[:1](torch.cat((down1_shad,down2_shad_1,down3_shad_1,down4_shad_1),1))
        up_down1_shad = self.conv1x1_ReLU_down1[1:](up_down1_feat)
        #
        # pred_down1_dst1 = F.upsample(up_down1_dst1,size=x.size()[2:], mode='bilinear')
        # pred_down1_dst2 = F.upsample(up_down1_dst2,size=x.size()[2:], mode='bilinear')
        pred_down1_shad = F.upsample(up_down1_shad,size=x.size()[2:], mode='bilinear')


        down0_dst1 = self.dst1(down0)
        down0_dst2 = self.dst2(down0)
        down0_shad = down0

        down0_shad = (1 + self.attention0_hl(torch.cat((down0_shad, down0_dst2), 1))) * down0_shad
        down0_shad = F.relu(-self.refine0_hl(torch.cat((down0_shad, down0_dst1), 1)) + down0_shad, True)


        # up_down0_dst1 =self.conv1x1_ReLU_down0(torch.cat((down0_dst1,down1_dst1_0,down2_dst1_0,down3_dst1_0,down4_dst1_0),1))
        # up_down0_dst2 = self.conv1x1_ReLU_down0(torch.cat((down0_dst2,down1_dst2_0,down2_dst2_0,down3_dst2_0,down4_dst2_0),1))
        ### feat
        up_down0_feat = self.conv1x1_ReLU_down0[:1](torch.cat((down0_shad,down1_shad_0,down2_shad_0,down3_shad_0,down4_shad_0),1))
        up_down0_shad = self.conv1x1_ReLU_down0[1:](up_down0_feat)
        #
        # pred_down0_dst1 = F.upsample(up_down0_dst1,size=x.size()[2:], mode='bilinear')
        # pred_down0_dst2 = F.upsample(up_down0_dst2,size=x.size()[2:], mode='bilinear')
        pred_down0_shad = F.upsample(up_down0_shad,size=x.size()[2:], mode='bilinear')


        fuse_pred_shad = self.fuse_predict(torch.cat((pred_down0_shad,pred_down1_shad,pred_down2_shad,pred_down3_shad,pred_down4_shad),1))
        # fuse_pred_dst1 = self.fuse_predict(torch.cat((pred_down0_dst1,pred_down1_dst1,pred_down2_dst1,pred_down3_dst1,pred_down4_dst1),1))
        # fuse_pred_dst2 = self.fuse_predict(torch.cat((pred_down0_dst2,pred_down1_dst2,pred_down2_dst2,pred_down3_dst2,pred_down4_dst2),1))

        if self.training or val:
            return fuse_pred_shad, pred_down1_shad, pred_down2_shad, pred_down3_shad, pred_down4_shad, self.avg_pool(up_down0_feat),up_down0_feat
        #     return fuse_pred_shad, pred_down1_shad, pred_down2_shad, pred_down3_shad, pred_down4_shad, \
        #     fuse_pred_dst1, pred_down1_dst1, pred_down2_dst1, pred_down3_dst1, pred_down4_dst1,\
        #     fuse_pred_dst2, pred_down1_dst2, pred_down2_dst2, pred_down3_dst2, pred_down4_dst2, \
        #            pred_down0_dst1, pred_down0_dst2, pred_down0_shad
        return  fuse_pred_shad
        return  F.sigmoid(fuse_pred_shad)
        return F.sigmoid(fuse_pred_shad), up_down4_feat, up_down3_feat, up_down2_feat, up_down1_feat, up_down0_feat
    
    def get_sim(self, feat_0,query_feat_0):
        feat_0 = feat_0.flatten(start_dim=2)
        query_feat_0 = query_feat_0.flatten(start_dim=2)
        feat_0 = nn.functional.normalize(feat_0, dim=1)
        query_feat_0 = nn.functional.normalize(query_feat_0, dim=1)

        # print(feat_0.size())
        sim_matrix = torch.matmul(feat_0.transpose(1,2),query_feat_0)
      
        # print(sim_matrix.size())
        sim_matrix = torch.max(sim_matrix,dim=1)[0] - torch.mean(sim_matrix,dim=1)[0]
        idex_matrix = torch.max(sim_matrix,dim=1)[1]
        # print(idex_matrix)
        return sim_matrix

    def get_shadow_map(self,img_predict,target,bs):
        avg_img = self.avg_pool(img_predict)
        dim = avg_img.size(1)

        target = self.avg_pool(target)
        target1 = torch.where(target>0.5,1.0,0.0)
        target1 = target1.repeat(1,dim,1,1)
        # print(target.size(),avg_img.size())
        # print(target.dtype, avg_img.dtype)

        non_shadow_index = (target1.flatten(start_dim=0)==0.0).nonzero()
        shadow_index = (target1.flatten(start_dim=0)==1.0).nonzero()
        # print(non_shadow_index.size())
        avg_img = avg_img.flatten(start_dim=0)
        # print(avg_img.size())
        shadow_feat = avg_img.index_fill(0,non_shadow_index.squeeze(),0.0)
        shadow_feat = shadow_feat.reshape(bs,dim,-1)

        non_shadow_feat = avg_img.index_fill(0,shadow_index.squeeze(),0.0)
        non_shadow_feat = non_shadow_feat.reshape(bs,dim,-1)
        # ssss
        return shadow_feat, non_shadow_feat

    def get_sim_and_guided(self, feat_shadow, query_feat_0,  query_feat_shadow, query_feat_non_shadow):

       
        shadow_matrix = self.get_sim(feat_shadow, query_feat_0)
        query_shadow_matrix = self.get_sim(feat_shadow, query_feat_shadow)
        query_non_shadow_matrix = self.get_sim(feat_shadow, query_feat_non_shadow)
        # predict_sim_matrix = self.get_sim(query_feat_shadow, avg_query_img)

        return shadow_matrix, query_shadow_matrix, query_non_shadow_matrix
    
        return loss_fn(sim_matrix,predict_sim_matrix) 

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super(DSDNet, self).train(mode)
       
        # if self.interation>5000:
        if self.fixedBN:
            print("Freezing Mean/Var of BatchNorm2D.")
            if self.freeze_bn_affine:
                print("Freezing Weight/Bias of BatchNorm2D.")
        if self.fixedBN:
            for m in self.modules():
                
                if isinstance(m, nn.BatchNorm2d):
                    # print(m)
                    m.eval()
                    
                    # if self.freeze_bn_affine:
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
                    # print(m.parameters)
    
    def forward(self, inputs, curr_iter=0, val=False):
        self.interation = curr_iter
        # print('inner:',self.interation)
        
        img = inputs['img']
        bs = img.size(0)
        outputs = dict()
           
        # img_predict, feat_4, feat_3, feat_2, feat_1, feat_0 = self.dssd_forward(img)
        # query_img_predict, query_feat_4, query_feat_3, query_feat_2, query_feat_1, query_feat_0 = self.dssd_forward(query_img)
        if self.training or val:
            query_img = inputs['query_img']
            target = inputs['target']
            query_target = inputs['query_target']
            if 'e_query_img' in inputs:
                e_query_img = inputs['e_query_img']
                e_query_img_predict,  e_query_img_predict1, e_query_img_predict2, e_query_img_predict3, e_query_img_predict4, e_query_feat_0,_ = self.dssd_forward(e_query_img,val)
            # e_query_target = inputs['e_query_target']

            img_predict, img_predict1, img_predict2, img_predict3, img_predict4, feat_0,final_feat = self.dssd_forward(img,val)
            query_img_predict,  query_img_predict1, query_img_predict2, query_img_predict3, query_img_predict4, query_feat_0,final_query_feat = self.dssd_forward(query_img,val)
           
            outputs['feat_0'] = final_feat
            outputs['query_feat_0'] = final_query_feat
            
      
            # print(sim_matrix.size())
            # refere = (img_predict.data > 0).to(torch.float32)
            # query_refere = (query_img_predict.data > 0).to(torch.float32)
            if 'e_query_img' in inputs:
                print('e_query_img')
                query_feat_0 = e_query_feat_0

            feat_shadow, feat_non_shadow = self.get_shadow_map(feat_0, target, bs)

            query_feat_shadow, query_feat_non_shadow= self.get_shadow_map(query_feat_0, query_target, bs)
           
            shadow_matrix1, shadow_matrix2, shadow_matrix3 = self.get_sim_and_guided( feat_shadow, query_feat_0,  query_feat_shadow, query_feat_non_shadow)
            # non_shadow_matrix1, non_shadow_matrix2 = self.get_sim_and_guided( feat_non_shadow, query_feat_0,  query_feat_non_shadow)
            query_shadow_matrix1, query_shadow_matrix2, query_shadow_matrix3  = self.get_sim_and_guided( query_feat_shadow, feat_0,  feat_shadow, feat_non_shadow)
            # query_non_shadow_matrix1, query_non_shadow_matrix2  = self.get_sim_and_guided( query_feat_non_shadow, feat_0,  feat_non_shadow)


            # if 'e_query_img' in inputs:
            #     e_query_feat_shadow = self.get_shadow_map(e_query_feat_0, query_target, bs)
            #     e_shadow_matrix1, e_shadow_matrix2 = self.get_sim_and_guided(feat_shadow, e_query_feat_0, e_query_feat_shadow) 
            #     e_query_shadow_matrix1, e_query_shadow_matrix2 = self.get_sim_and_guided(e_query_feat_shadow, feat_0, feat_shadow) 
                
                # bright_sim_matrix = self.get_sim(feat_0,query_feat_0)
                # bright_e_sim_matrix = self.get_sim(feat_0,e_query_feat_0)
           

            # pc_loss1 = self.pc_loss(avg_img, avg_query_img, feat_0, query_feat_0) 
            # pc_loss2 = self.pc_loss(avg_query_img, avg_img, query_feat_0, feat_0) 
            
           
            
           
            
            
            # pc_c = pc_loss1 + pc_loss2 + e_pc_loss1 + e_pc_loss2

            # pc_c = pc_loss1 + pc_loss2 
           
            # print(sim_matrix.size(),predict_sim_matrix.size())

            # ssss
            # print(query_feat_0.size(),feat_0.size())
            # print(img_predict.size())
            # print(img_predict.size(), feat_4.size(), feat_3.size(), feat_2.size(), feat_1.size(), feat_0.size())
           
            outputs['query'] = [
                [query_img_predict,  query_img_predict1, query_img_predict2, query_img_predict3, query_img_predict4],\
                                # [e_query_img_predict,  e_query_img_predict1, e_query_img_predict2, e_query_img_predict3, e_query_img_predict4]
                                ]
            outputs['sim_matrix'] = [[shadow_matrix1,shadow_matrix2,shadow_matrix3], [query_shadow_matrix1, query_shadow_matrix2,query_shadow_matrix3],
                                    # [non_shadow_matrix1, non_shadow_matrix2], [query_non_shadow_matrix1, query_non_shadow_matrix2]
                                         ]
                                    #   [ e_sim_matrix, e_predict_sim_matrix], [ e_query_sim_matrix, e_query_predict_sim_matrix],
                                                             
            # print(shadow_matrix1.size())                                                
            # if self.args['e_query'] and 'e_query_img' in inputs:
            #     outputs['sim_matrix'].append([ e_shadow_matrix1, e_shadow_matrix2])
            #     outputs['sim_matrix'].append([ e_query_shadow_matrix1, e_query_shadow_matrix2])
            #     outputs['query'].append([e_query_img_predict,  e_query_img_predict1, e_query_img_predict2, e_query_img_predict3, e_query_img_predict4])
            # if self.args['consis'] and 'e_query_img' in inputs:
            #     outputs['sim_matrix'].append([bright_sim_matrix, bright_e_sim_matrix])
            # if self.args['non_shadow']:
            #     outputs['sim_matrix'].append([non_shadow_matrix1, non_shadow_matrix2])
            #     outputs['sim_matrix'].append([query_non_shadow_matrix1, query_non_shadow_matrix2])
           
            return  img_predict, img_predict1, img_predict2, img_predict3, img_predict4,\
                     outputs
            # else:
            #     return img_predict, img_predict1, img_predict2, img_predict3, img_predict4, \
            #         query_img_predict,  query_img_predict1, query_img_predict2, query_img_predict3, query_img_predict4, pc_c

                
        return self.dssd_forward(img)
        
