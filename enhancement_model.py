import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class PPM1(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM1, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.PReLU()
            ))
        self.features = nn.ModuleList(self.features)
        self.fuse = nn.Sequential(
                nn.Conv2d(in_dim+reduction_dim*4, in_dim, kernel_size=3, padding=1, bias=False),
                nn.PReLU())

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        out_feat = self.fuse(torch.cat(out, 1))
        return out_feat       
          

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.downsample = downsample
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.lrelu(out)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.lrelu(out)
        return out


class UNet_emb_oneBranch_symmetry(nn.Module):
    
    def __init__(self, in_channels=3, out_channels=3, bias=False, padding_mode='zeros'):
        super(UNet_emb_oneBranch_symmetry, self).__init__()

        self.cond1 = nn.Conv2d(in_channels, 32, 3, 1, 1, bias=True, padding_mode=padding_mode) 
        self.cond_add1 = nn.Conv2d(32, out_channels, 3, 1, 1, bias=True, padding_mode=padding_mode)           

        self.condx = nn.Conv2d(32, 64, 3, 1, 1, bias=True, padding_mode=padding_mode) 
        self.condy = nn.Conv2d(64, 32, 3, 1, 1, bias=True, padding_mode=padding_mode) 

        self.relu = nn.ReLU(inplace=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.ResidualBlock1=ResidualBlock(32, 32)
        self.ResidualBlock2=ResidualBlock(32, 32)
        self.ResidualBlock3=ResidualBlock(64, 64)
        self.ResidualBlock4=ResidualBlock(64, 64)
        self.ResidualBlock5=ResidualBlock(32, 32)
        self.ResidualBlock6=ResidualBlock(32, 32)

        self.PPM1 = PPM1(32, 8, bins=(1,2,3,6))


    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                #nn.init.kaiming_normal_(m.weight.data)
                m.weight.data.normal_(0.0, 0.02)
                #nn.init.zeros_(m.bias.data)


    def forward(self, x):
        
        light_conv1=self.lrelu(self.cond1(x))
        res1=self.ResidualBlock1(light_conv1)
        
        res2=self.ResidualBlock2(res1)
        res2=self.PPM1(res2)
        res2=self.condx(res2)
        
        res3=self.ResidualBlock3(res2)
        res4=self.ResidualBlock4(res3)
        res4=self.condy(res4)
        
        res5=self.ResidualBlock5(res4)
        res6=self.ResidualBlock6(res5)

        light_map=self.relu(self.cond_add1(res6))

        return light_map


# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False, padding_mode='reflect')


def weights_init(m):
    classname = m.__class__.__name__ 
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def load_enhancement_model(config, padding_mode='zeros'):
    # load enhancement model, it's same for any mode
    U_net=UNet_emb_oneBranch_symmetry(3, 1, padding_mode=padding_mode).cuda()
    U_net.apply(weights_init)

    if config.unet_model.load_pretrain:
        # create new OrderedDict that does not contain `module.`
        state_dict = torch.load(config.unet_model.pretrain_dir)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        U_net.load_state_dict(new_state_dict)
    U_net= torch.nn.DataParallel(U_net)

    return U_net
