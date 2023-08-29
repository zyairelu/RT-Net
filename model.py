import torch
from torch import nn


class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(F_l, 1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        # import matplotlib.pyplot as plt
        # import matplotlib.colors as mcolors
        # att = psi[0,0].cpu().detach().numpy()
        # colors = [(1, 1, 1), (1, 0, 0)] # W -> R
        # cmap = mcolors.LinearSegmentedColormap.from_list('my_cmap', colors)

       
        att_map = self.conv(x*psi)

        return att_map


class Our_Net(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(Our_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=in_channels, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        # self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=512, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        # self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        # self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=128, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        # self.Up_conv2 = conv_block(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, out_channels, kernel_size=1, stride=1, padding=0)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x, rough_turbo=False):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4) # attention_map
        # d5 = torch.cat((x4, d5), dim=1)
        d5 = torch.mul(x4, d5)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        # d4 = torch.cat((x3, d4), dim=1)
        d4 = torch.mul(x3, d4)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        # d3 = torch.cat((x2, d3), dim=1)
        d3 = torch.mul(x2, d3)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        # d2 = torch.cat((x1, d2), dim=1)
        d2 = torch.mul(x1, d2)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        # d1 = self.sigmoid(d1)

        if rough_turbo:
            trans1 = torch.sigmoid(x1)
            trans2 = torch.sigmoid(x2)
            trans3 = torch.sigmoid(x3)
            trans4 = torch.sigmoid(x4)
            return trans1, trans2, trans3, trans4
        else:
            return d1

if __name__ == "__main__":
    import os
    import numpy as np
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Our_Net(in_channels=1, out_channels=1)
    model.to(device=device)
    for name, param in model.named_parameters():
        if "Att" in name:
            print(name, param.shape)
    input = torch.randn(8, 1, 256, 256).to(device=device)
    output = model(input)
    output = torch.sigmoid(output)

    # print(np.unique(output.cpu().detach().numpy()))
    # print(output.shape)