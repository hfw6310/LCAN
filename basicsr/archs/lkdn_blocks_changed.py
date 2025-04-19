import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.archs.arch_util import default_init_weights
from basicsr.archs.triplet_attention import *
import torch.nn.init as init
import math

# from basicsr.archs.deformable_LKA import deformable_LKA
# import DCNv4

# class BSConvU(nn.Module):

#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  kernel_size=3,
#                  stride=1,
#                  padding=1,
#                  dilation=1,
#                  bias=True,
#                  padding_mode="zeros"):
#         super().__init__()

#         # pointwise
#         self.pw = nn.Conv2d(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             kernel_size=(1, 1),
#             stride=1,
#             padding=0,
#             dilation=1,
#             groups=1,
#             bias=False,
#         )
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         if in_channels == out_channels:
#             self.depthwise_1k = nn.Conv2d(out_channels, out_channels, (1,5), padding=(0,2), groups=out_channels)
#             self.depthwise_k1 = nn.Conv2d(out_channels, out_channels, (5,1), padding=(2,0), groups=out_channels)
#         else:
#         # depthwise
#             self.dw = nn.Conv2d(
#                 in_channels=out_channels,
#                 out_channels=out_channels,
#                 kernel_size=kernel_size,
#                 stride=stride,
#                 padding=padding,
#                 dilation=dilation,
#                 groups=out_channels,
#                 bias=bias,
#                 padding_mode=padding_mode,
#             )

#     def forward(self, x):
#         u = x.clone()
        
#         #print("u.shape:",u.shape)
        
#         fea = self.pw(x)
#         if self.in_channels == self.out_channels:
#             fea = self.depthwise_1k(fea)
#             fea = self.depthwise_k1(fea)
#             return u * fea
#         #print("fea.shape:",fea.shape)
#         else:
#             fea = self.dw(fea)
#             return fea

class SEKG(nn.Module):
    def __init__(self, in_channels=64, kernel_size=3):
        super().__init__()
        self.conv_sa = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=in_channels)
        # channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_ca = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2) 

    def forward(self, input_x):
        b, c, h, w = input_x.size()
        # spatial attention
        sa_x = self.conv_sa(input_x)  
        # channel attention
        y = self.avg_pool(input_x)
        ca_x = self.conv_ca(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out  = sa_x + ca_x
        return out

# Adaptice Filter Generation 
class AFG(nn.Module):
    def __init__(self, in_channels=64, kernel_size=3):
        super(AFG, self).__init__()
        self.kernel_size = kernel_size
        self.sekg = SEKG(in_channels, kernel_size)
        self.conv = nn.Conv2d(in_channels, in_channels*kernel_size*kernel_size, 1, 1, 0)

    def forward(self, input_x):
        b, c, h, w = input_x.size()
        x = self.sekg(input_x)
        x = self.conv(x)
        filter_x = x.reshape([b, c, self.kernel_size*self.kernel_size, h, w])

        return filter_x

# Dynamic convolution
class DyConv(nn.Module):
    def __init__(self, in_channels=64, kernel_size=3):
        super(DyConv, self).__init__()
        self.kernel_size = kernel_size
        self.afg = AFG(in_channels, kernel_size)
        self.unfold = nn.Unfold(kernel_size=3, dilation=1, padding=1, stride=1)
        
    def forward(self, input_x):
        b, c, h, w = input_x.size()
        filter_x = self.afg(input_x)
        unfold_x = self.unfold(input_x).reshape(b, c, -1, h, w)
        out = (unfold_x * filter_x).sum(2)
        
        return out

class BSConvU(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 bias=True,
                 padding_mode="zeros"):
        super().__init__()

        # pointwise
        self.pw = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        # depthwise
        self.dw = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=out_channels,
            bias=bias,
            padding_mode=padding_mode,
        )

    def forward(self, fea):
        fea = self.pw(fea)
        fea = self.dw(fea)
        return fea

class BSConvU111(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 bias=True,
                 padding_mode="zeros"):
        super().__init__()

        # pointwise
        self.pw = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        # depthwise
        self.dw1 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=out_channels,
            bias=bias,
            padding_mode=padding_mode,
        )
                # depthwise
                #   
        self.dw2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=2,
            dilation=2,
            groups=out_channels,
            bias=bias,
            padding_mode=padding_mode,
        )

    def forward(self, fea):
        fea = self.pw(fea)
        fea = self.dw1(fea)
        fea = self.dw2(fea)
        return fea

class BSConvU_ks(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=[1,3,5], stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_ln=False, bn_kwargs=None):
        super().__init__()
        self.with_ln = with_ln
        # check arguments
        if bn_kwargs is None:
            bn_kwargs = {}

        # pointwise
        self.pw=torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
        )

        # depthwise
        kernel_size=[1,3,5]
        self.blocks = len(kernel_size)
        padding0 = kernel_size[0]//2
        self.dw0 = torch.nn.Conv2d(
                in_channels=out_channels//self.blocks,
                out_channels=out_channels//self.blocks,
                kernel_size=kernel_size[0],
                stride=stride,
                padding=padding0,
                dilation=dilation,
                groups=out_channels//self.blocks,
                bias=bias,
                padding_mode=padding_mode,
        )

        padding1 = kernel_size[1]//2
        self.dw1 = torch.nn.Conv2d(
                in_channels=out_channels//self.blocks,
                out_channels=out_channels//self.blocks,
                kernel_size=kernel_size[1],
                stride=stride,
                padding=padding1,
                dilation=dilation,
                groups=out_channels//self.blocks,
                bias=bias,
                padding_mode=padding_mode,
        )

        padding2 = kernel_size[2]//2
        self.dw2 = torch.nn.Conv2d(
                in_channels=out_channels-out_channels//self.blocks*2,
                out_channels=out_channels-out_channels//self.blocks*2,
                kernel_size=kernel_size[2],
                stride=stride,
                padding=padding2,
                dilation=dilation,
                groups=out_channels-out_channels//self.blocks*2,
                bias=bias,
                padding_mode=padding_mode,
        )
        # self.dw2 = torch.nn.Sequential(torch.nn.Conv2d(
        #         in_channels=out_channels-2*out_channels//self.blocks,
        #         out_channels=out_channels-2*out_channels//self.blocks,
        #         kernel_size=(1,kernel_size[2]),
        #         stride=stride,
        #         padding=padding2,
        #         dilation=dilation,
        #         groups=out_channels-2*out_channels//self.blocks,
        #         bias=bias,
        #         padding_mode=padding_mode,
        # ),
        # torch.nn.Conv2d(
        #         in_channels=out_channels-2*out_channels//self.blocks,
        #         out_channels=out_channels-2*out_channels//self.blocks,
        #         kernel_size=(kernel_size[2],1),
        #         stride=stride,
        #         padding=0,
        #         dilation=dilation,
        #         groups=out_channels-2*out_channels//self.blocks,
        #         bias=bias,
        #         padding_mode=padding_mode,
        # )
        # )
        
        # self.InceptionDWConv2d = InceptionDWConv2d(out_channels)

    def forward(self, fea):
        fea = self.pw(fea)
        # print('self.blocks:',self.blocks)
        self.split_channels = [fea.shape[1] // self.blocks, fea.shape[1] // self.blocks,fea.shape[1]-fea.shape[1] // self.blocks*2]
        # print('!!!!!!!!!!!!!!!!!!!!!!!!!self.split_channels:',self.split_channels)
        fea0,fea1,fea2= torch.split(fea, self.split_channels, dim=1)

        fea0 = self.dw0(fea0)
        fea1 = self.dw1(fea1)
        fea2 = self.dw2(fea2)  
        fea = torch.cat((fea0,fea1,fea2),1)
        
        # fea = self.dw(fea)
        # fea = self.InceptionDWConv2d(fea)

        # fea = self.channel_shuffle(fea, 2)
        return fea

# class BSConvU_ks(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=[1,3,5], stride=1, padding=1,
#                  dilation=1, bias=True, padding_mode="zeros", with_ln=False, bn_kwargs=None):
#         super().__init__()
#         self.with_ln = with_ln
#         # check arguments
#         if bn_kwargs is None:
#             bn_kwargs = {}

#         # pointwise
#         self.pw=torch.nn.Conv2d(
#                 in_channels=in_channels,
#                 out_channels=out_channels,
#                 kernel_size=(1, 1),
#                 stride=1,
#                 padding=0,
#                 dilation=1,
#                 groups=1,
#                 bias=False,
#         )

#         # depthwise
#         kernel_size=[3,3,3]
#         self.blocks = len(kernel_size)
#         padding0 = kernel_size[0]//2
#         self.dw0 = torch.nn.Conv2d(
#                 in_channels=out_channels//self.blocks,
#                 out_channels=out_channels//self.blocks,
#                 kernel_size=kernel_size[0],
#                 stride=stride,
#                 padding=padding0,
#                 dilation=dilation,
#                 groups=out_channels//self.blocks,
#                 bias=bias,
#                 padding_mode=padding_mode,
#         )

#         dilation1 = (kernel_size[1]+1)//2
#         padding1 = dilation1*(kernel_size[1]-1)//2 
#         self.dw1 = torch.nn.Conv2d(
#                 in_channels=out_channels//self.blocks,
#                 out_channels=out_channels//self.blocks,
#                 kernel_size=kernel_size[1],
#                 stride=stride,
#                 padding=padding1,
#                 dilation=dilation1,
#                 groups=out_channels//self.blocks,
#                 bias=bias,
#                 padding_mode=padding_mode,
#         )

#         dilation2 = (kernel_size[2]+1)//2+1
#         padding2 = dilation2*(kernel_size[2]-1)//2
#         self.dw2 = torch.nn.Conv2d(
#                 in_channels=out_channels-out_channels//self.blocks*2,
#                 out_channels=out_channels-out_channels//self.blocks*2,
#                 kernel_size=kernel_size[2],
#                 stride=stride,
#                 padding=padding2,
#                 dilation=dilation2,
#                 groups=out_channels-out_channels//self.blocks*2,
#                 bias=bias,
#                 padding_mode=padding_mode,
#         )
#         # self.dw2 = torch.nn.Sequential(torch.nn.Conv2d(
#         #         in_channels=out_channels-2*out_channels//self.blocks,
#         #         out_channels=out_channels-2*out_channels//self.blocks,
#         #         kernel_size=(1,kernel_size[2]),
#         #         stride=stride,
#         #         padding=padding2,
#         #         dilation=dilation,
#         #         groups=out_channels-2*out_channels//self.blocks,
#         #         bias=bias,
#         #         padding_mode=padding_mode,
#         # ),
#         # torch.nn.Conv2d(
#         #         in_channels=out_channels-2*out_channels//self.blocks,
#         #         out_channels=out_channels-2*out_channels//self.blocks,
#         #         kernel_size=(kernel_size[2],1),
#         #         stride=stride,
#         #         padding=0,
#         #         dilation=dilation,
#         #         groups=out_channels-2*out_channels//self.blocks,
#         #         bias=bias,
#         #         padding_mode=padding_mode,
#         # )
#         # )
        
#         # self.InceptionDWConv2d = InceptionDWConv2d(out_channels)

#     def forward(self, fea):
#         fea = self.pw(fea)
#         # print('self.blocks:',self.blocks)
#         self.split_channels = [fea.shape[1] // self.blocks, fea.shape[1] // self.blocks,fea.shape[1]-fea.shape[1] // self.blocks*2]
#         # print('!!!!!!!!!!!!!!!!!!!!!!!!!self.split_channels:',self.split_channels)
#         fea0,fea1,fea2= torch.split(fea, self.split_channels, dim=1)

#         fea0 = self.dw0(fea0)
#         fea1 = self.dw1(fea1)
#         fea2 = self.dw2(fea2)  
#         fea = torch.cat((fea0,fea1,fea2),1)
#         # fea = self.dw(fea)
#         # fea = self.InceptionDWConv2d(fea)

#         # fea = self.channel_shuffle(fea, 2)
#         return fea

class BSConvU_idt(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 bias=True,
                 padding_mode="zeros"):
        super().__init__()

        # pointwise
        self.pw = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        # depthwise
        self.dw = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=out_channels,
            bias=bias,
            padding_mode=padding_mode,
        )

    def forward(self, x):
        fea = self.pw(x)
        fea = self.dw(fea)
        return fea + x


class BSConvU_rep(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 bias=True,
                 padding_mode="zeros"):
        super().__init__()

        # pointwise
        self.pw = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        # depthwise
        self.dw = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=out_channels,
            bias=bias,
            padding_mode=padding_mode,
        )

        self.rep1x1 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=dilation,
            groups=out_channels,
            bias=bias,
            padding_mode=padding_mode,
        )

    def forward(self, fea):
        fea = self.pw(fea) + fea
        fea = self.dw(fea) + fea + self.rep1x1(fea)
        return fea
class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        
        # Two different branches of ECA module
        y = self.conv1(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class KBAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, att, selfk, selfg, selfb, selfw):
        B, nset, H, W = att.shape
        KK = selfk ** 2
        selfc = x.shape[1]

        att = att.reshape(B, nset, H * W).transpose(-2, -1)

        ctx.selfk, ctx.selfg, ctx.selfc, ctx.KK, ctx.nset = selfk, selfg, selfc, KK, nset
        ctx.x, ctx.att, ctx.selfb, ctx.selfw = x, att, selfb, selfw

        bias = att @ selfb
        attk = att @ selfw

        uf = torch.nn.functional.unfold(x, kernel_size=selfk, padding=selfk // 2)

        # for unfold att / less memory cost
        uf = uf.reshape(B, selfg, selfc // selfg * KK, H * W).permute(0, 3, 1, 2)
        attk = attk.reshape(B, H * W, selfg, selfc // selfg, selfc // selfg * KK)

        x = attk @ uf.unsqueeze(-1)  #
        del attk, uf
        x = x.squeeze(-1).reshape(B, H * W, selfc) + bias
        x = x.transpose(-1, -2).reshape(B, selfc, H, W)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, att, selfb, selfw = ctx.x, ctx.att, ctx.selfb, ctx.selfw
        selfk, selfg, selfc, KK, nset = ctx.selfk, ctx.selfg, ctx.selfc, ctx.KK, ctx.nset

        B, selfc, H, W = grad_output.size()

        dbias = grad_output.reshape(B, selfc, H * W).transpose(-1, -2)

        dselfb = att.transpose(-2, -1) @ dbias
        datt = dbias @ selfb.transpose(-2, -1)

        attk = att @ selfw
        uf = F.unfold(x, kernel_size=selfk, padding=selfk // 2)
        # for unfold att / less memory cost
        uf = uf.reshape(B, selfg, selfc // selfg * KK, H * W).permute(0, 3, 1, 2)
        attk = attk.reshape(B, H * W, selfg, selfc // selfg, selfc // selfg * KK)

        dx = dbias.view(B, H * W, selfg, selfc // selfg, 1)

        dattk = dx @ uf.view(B, H * W, selfg, 1, selfc // selfg * KK)
        duf = attk.transpose(-2, -1) @ dx
        del attk, uf

        dattk = dattk.view(B, H * W, -1)
        datt += dattk @ selfw.transpose(-2, -1)
        dselfw = att.transpose(-2, -1) @ dattk

        duf = duf.permute(0, 2, 3, 4, 1).view(B, -1, H * W)
        dx = F.fold(duf, output_size=(H, W), kernel_size=selfk, padding=selfk // 2)

        datt = datt.transpose(-1, -2).view(B, nset, H, W)

        return dx, datt, None, None, dselfb, dselfw
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
        
class MFF(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, act=True, gc=2, nset=32, k=3):
        super(MFF, self).__init__()
        self.act = act
        self.gc = gc

        hidden_features = int(dim * ffn_expansion_factor)     # ffn_expansion_factor=1.5

        # self.dwconv = nn.Sequential(
        #     nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias),
        #     nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
        #               groups=hidden_features, bias=bias),
        # )
        # self.sca = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(in_channels=dim, out_channels=hidden_features, kernel_size=1, padding=0, stride=1,
        #               groups=1, bias=True),
        # )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)


        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias),
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                      groups=hidden_features, bias=bias),
        )

        c = hidden_features
        self.k, self.c = k, c
        self.nset = nset

        self.g = c // gc
        self.w = nn.Parameter(torch.zeros(1, nset, c * c // self.g * self.k ** 2))
        self.b = nn.Parameter(torch.zeros(1, nset, c))
        self.init_p(self.w, self.b)
        interc = min(dim, 24)
        # print("!!!:",dim,  c, interc)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=interc, kernel_size=3, padding=1, stride=1, groups=interc,
                      bias=True),
            SimpleGate(),
            nn.Conv2d(interc // 2, self.nset, 1, padding=0, stride=1),
        )
        self.conv211 = nn.Conv2d(in_channels=dim, out_channels=self.nset, kernel_size=1)
        self.attgamma = nn.Parameter(torch.zeros((1, self.nset, 1, 1)) + 1e-2, requires_grad=True)
        self.ga1 = nn.Parameter(torch.zeros((1, hidden_features, 1, 1)) + 1e-2, requires_grad=True)
        

    def forward(self, x):
        # sca = self.sca(x)
        # x1 = self.dwconv(x)

        att = self.conv2(x) * self.attgamma + self.conv211(x)
        uf = self.conv1(x)
        x = self.KBA(uf, att, self.k, self.g, self.b, self.w) * self.ga1 + uf

        # x = F.gelu(x1) * x2 if self.act else x1 * x2
        # x = x * sca

        x = self.project_out(x)
        return x

    def init_p(self, weight, bias=None):
        init.kaiming_uniform_(weight, a=math.sqrt(5))
        if bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(bias, -bound, bound)

    def KBA(self, x, att, selfk, selfg, selfb, selfw):
        return KBAFunction.apply(x, att, selfk, selfg, selfb, selfw)

class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C,height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

# class Attention(nn.Module):

#     def __init__(self, dim):
#         super().__init__()
#         self.pointwise = nn.Conv2d(dim, dim, 1)
#         # self.depthwise = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
#         # self.depthwise_dilated = nn.Conv2d(dim, dim, 5, stride=1, padding=6, groups=dim, dilation=3)
        
#         self.dcn = DCNv4.DCNv4(
#             channels=dim,
#             kernel_size=3,
#             stride=1,
#             group=4,
#             pad=1,
#             offset_scale=1.0,
#             dw_kernel_size=None,
#             output_bias=True,  
#         )
#         # self.dcn_dilated = DCNv4.DCNv4(
#         #     channels=dim,
#         #     kernel_size=5,
#         #     stride=1,
#         #     group=dim,
#         #     pad=6,
#         #     dilation=3,
#         #     offset_scale=1.0,
#         #     dw_kernel_size=None,
#         #     output_bias=True,
#         # )
#     def forward(self, x):
#         u = x.clone()
        
#         attn = self.dcn(x)
#         # attn = self.dcn_dilated(attn)
#         attn = self.pointwise(attn)
#         return u * attn

# class Attention(nn.Module):

#     def __init__(self, dim):
#         super().__init__()
#         self.pointwise = nn.Conv2d(dim, dim, 1)
#         self.depthwise = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
#         self.depthwise_dilated = nn.Conv2d(dim, dim, 5, stride=1, padding=6, groups=dim, dilation=3)

#     def forward(self, x):
#         u = x.clone()
        
#         attn = self.depthwise(x)
#         attn = self.depthwise_dilated(attn)
#         attn = self.pointwise(attn)
#         return u * attn

# class Attention(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.pointwise = nn.Conv2d(dim, dim, 1)
#         self.depthwise = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
#         self.depthwise_dilated = DeformableConv2d(dim, dim, 5, stride=1, padding=6, groups=dim, dilation=3)

#     def forward(self, x):
#         u = x.clone()
#         attn = self.pointwise(x)
#         attn = self.depthwise(attn)
#         # print("attn.shape:",attn.shape)
#         attn = self.depthwise_dilated(attn)
#         return u * attn


# class Attention(nn.Module):

#     def __init__(self, dim):
#         super().__init__()
#         self.pointwise = nn.Conv2d(dim, dim, 1)
#         self.depthwise = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
#         self.depthwise_dilated = nn.Conv2d(dim, dim, 3, stride=1, padding=2, groups=dim, dilation=2)

#     def forward(self, x):
#         # attn = self.pointwise(x)
#         attn = x
#         attn = self.depthwise(attn)
#         # print("attn.shape:",attn.shape)
#         attn = self.depthwise_dilated(attn)
#         return attn

# class Attention(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.pointwise = nn.Conv2d(dim, dim, 1)
#         self.depthwise = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
#         self.depthwise_dilated = nn.Conv2d(dim, dim, 5, stride=1, padding=6, groups=dim, dilation=3)

#     def forward(self, x):
#         u = x.clone()
#         attn1 = self.pointwise(x)
#         attn1 = u * attn1
#         u1 = attn1.clone()
#         attn2 = self.depthwise(attn1)

#         # attn = attn * u
#         # print("attn.shape:",attn.shape)
#         attn2 = u1 * attn2
#         u2 = attn2.clone()
#         attn3 = self.depthwise_dilated(attn2)

#         return u2 * attn3

# class Attention(nn.Module):

#     def __init__(self, dim):
#         super().__init__()
#         self.pointwise = nn.Conv2d(dim, dim, 1)
#         self.depthwise = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
#         self.depthwise_dilated = nn.Conv2d(dim, dim, 3, stride=1, padding=2, groups=dim, dilation=2)
#         self.depthwise_dilated1 = nn.Conv2d(dim, dim, 3, stride=1, padding=3, groups=dim, dilation=3)
#         self.depthwise_dilated2 = nn.Conv2d(dim, dim, 3, stride=1, padding=4, groups=dim, dilation=4)
#     def forward(self, x):
#         u = x.clone()
#         attn = self.pointwise(x)
#         attn = self.depthwise(attn)
#         # print("attn.shape:",attn.shape)
#         attn = self.depthwise_dilated(attn)
#         attn = self.depthwise_dilated1(attn)
#         attn = self.depthwise_dilated2(attn)
#         return u * attn


# class Attention(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.pointwise = nn.Conv2d(dim, dim, 1)
#         self.depthwise = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
#         self.depthwise_dilated = nn.Conv2d(dim, dim, 5, stride=1, padding=6, groups=dim, dilation=3)

#     def forward(self, x):
#         u = x.clone()
#         attn1 = self.pointwise(x)
#         u1 = attn1.clone()
#         attn2 = self.depthwise(attn1)

#         # attn = attn * u
#         # print("attn.shape:",attn.shape)
#         u2 = attn2.clone()

#         attn3 = self.depthwise_dilated(u * attn2)

#         return u1 * attn3

# class Attention(nn.Module):

#     def __init__(self, dim):
#         super().__init__()
#         self.pointwise1 = nn.Conv2d(dim, dim//2, 1)
#         self.depthwise = nn.Conv2d(dim//2, dim//2, 7, padding=3, groups=dim//2)
#         self.depthwise_dilated = nn.Conv2d(dim//2, dim//2, 7, stride=1, padding=12, groups=dim//2, dilation=4)
#         self.pointwise2 = nn.Conv2d(dim//2, dim, 1)

#     def forward(self, x):
#         u = x.clone()
#         attn = self.pointwise1(x)
#         attn = self.depthwise(attn)
#         # print("attn.shape:",attn.shape)
#         attn = self.depthwise_dilated(attn)
#         attn = self.pointwise2(attn)

#         return u * attn

# class Attention(nn.Module):

#     def __init__(self, dim):
#         super().__init__()
#         self.eca_layer1 = eca_layer1(dim)

#         self.pointwise1 = nn.Conv2d(dim, dim//2, 1)
#         self.depthwise1 = nn.Conv2d(dim//2, dim//2, 7, padding=3, groups=dim//2)
#         self.depthwise_dilated1 = nn.Conv2d(dim//2, 1, 7, stride=1, padding=12, groups=1, dilation=4)

#         # self.eca_layer2 = eca_layer2(dim)       
#         # self.pointwise2 = nn.Conv2d(dim, 1, 1)
#         # self.depthwise2 = nn.Conv2d(1, 1, 5, padding=2, groups=1)
#         # self.depthwise_dilated2 = nn.Conv2d(1, 1, 5, stride=1, padding=6, groups=1, dilation=3)

#     def forward(self, x):
        
#         attn1 =  self.eca_layer1(x)
#         u1 = attn1.clone()
#         attn = self.pointwise1(attn1)
#         attn = self.depthwise1(attn)
#         # print("attn.shape:",attn.shape)
#         attn = self.depthwise_dilated1(attn)

#         # attn2 =  self.eca_layer2(x)
#         # u2 = attn2.clone()
#         # attn_ = self.pointwise2(attn2)
#         # attn_ = self.depthwise2(attn_)
#         # # print("attn.shape:",attn.shape)
#         # attn_ = self.depthwise_dilated2(attn_)
#         # return u1 * attn + u2 * attn_

#         return u1 * attn 
      
import torchvision
class DeformConv(nn.Module):

    def __init__(self, in_channels, groups, kernel_size=(3,3), padding=1, stride=1, dilation=1, bias=True):
        super(DeformConv, self).__init__()
        self.offset_net = BSConvU(in_channels=in_channels, # nn.Conv2d
                                    out_channels=2 * kernel_size[0] * kernel_size[1],
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    stride=stride,
                                    dilation=dilation,
                                    bias=True)

        self.deform_conv = torchvision.ops.DeformConv2d(in_channels=in_channels,
                                                        out_channels=in_channels,
                                                        kernel_size=kernel_size,
                                                        padding=padding,
                                                        groups=groups,
                                                        stride=stride,
                                                        dilation=dilation,
                                                        bias=False)

    def forward(self, x):
        offsets = self.offset_net(x)
        out = self.deform_conv(x, offsets)
        return out

class deformable_LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = DeformConv(dim, kernel_size=(5,5), padding=2, groups=dim)
        self.conv_spatial = DeformConv(dim, kernel_size=(5,5), stride=1, padding=6, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)


    def forward(self, x):
        u = x.clone()        
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn

# class Attention(nn.Module):

#     def __init__(self, dim):
#         super().__init__()
#         self.pointwise = nn.Conv2d(dim, dim, 1)
        
#         # self.depthwise_1k = nn.Conv2d(dim, dim, (1,11), padding=5, groups=dim)
#         # self.depthwise_k1 = nn.Conv2d(dim, dim, (11,1), padding=0, groups=dim)
#         # self.depthwise_dilated_1k = nn.Conv2d(dim, dim, (1,11), stride=1, padding=30, groups=dim, dilation=6)
#         # self.depthwise_dilated_k1 = nn.Conv2d(dim, dim, (11,1), stride=1, padding=0, groups=dim, dilation=6)

        
#         # self.depthwise_1k = nn.Conv2d(dim, dim, (1,11), padding=5, groups=dim)
#         # self.depthwise_k1 = nn.Conv2d(dim, dim, (11,1), padding=0, groups=dim)
#         # self.depthwise_dilated_1k = nn.Conv2d(dim, dim, (1,11), stride=1, padding=30, groups=dim, dilation=6)
#         # self.depthwise_dilated_k1 = nn.Conv2d(dim, dim, (11,1), stride=1, padding=0, groups=dim, dilation=6)
        
#         # self.depthwise_1k = nn.Conv2d(dim, dim, (1,11), padding=(0,5), groups=dim)
#         # self.depthwise_k1 = nn.Conv2d(dim, dim, (11,1), padding=(5,0), groups=dim)
#         # self.depthwise_dilated_1k = nn.Conv2d(dim, dim, (1,11), stride=1, padding=(0,30), groups=dim, dilation=6)
#         # self.depthwise_dilated_k1 = nn.Conv2d(dim, dim, (11,1), stride=1, padding=(30,0), groups=dim, dilation=6)
        
#         #self.depthwise_1k = nn.Conv2d(dim, dim, (1,7), padding=(0,3), groups=dim)
#         #self.depthwise_k1 = nn.Conv2d(dim, dim, (7,1), padding=(3,0), groups=dim)
#         #self.depthwise_dilated_1k = nn.Conv2d(dim, dim, (1,13), stride=1, padding=(0,24), groups=dim, dilation=4)
#         #self.depthwise_dilated_k1 = nn.Conv2d(dim, dim, (13,1), stride=1, padding=(24,0), groups=dim, dilation=4)
                
#         self.depthwise_1k = nn.Conv2d(dim, dim, (1,9), padding=(0,4), groups=dim)
#         self.depthwise_k1 = nn.Conv2d(dim, dim, (9,1), padding=(4,0), groups=dim)
#         self.depthwise_dilated_1k = nn.Conv2d(dim, dim, (1,9), stride=1, padding=(0,20), groups=dim, dilation=5)
#         self.depthwise_dilated_k1 = nn.Conv2d(dim, dim, (9,1), stride=1, padding=(20,0), groups=dim, dilation=5)
        
#         # self.depthwise_1k = nn.Conv2d(dim, dim, (1,7), padding=3, groups=dim)
#         # self.depthwise_k1 = nn.Conv2d(dim, dim, (7,1), padding=0, groups=dim)
#         # self.depthwise_dilated_1k = nn.Conv2d(dim, dim, (1,7), stride=1, padding=12, groups=dim, dilation=4)
#         # self.depthwise_dilated_k1 = nn.Conv2d(dim, dim, (7,1), stride=1, padding=0, groups=dim, dilation=4)

#         # self.depthwise_1k = nn.Conv2d(dim, dim, (1,5), padding=2, groups=dim)
#         # self.depthwise_k1 = nn.Conv2d(dim, dim, (5,1), padding=0, groups=dim)
#         # self.depthwise_dilated_1k = nn.Conv2d(dim, dim, (1,5), stride=1, padding=6, groups=dim, dilation=3)
#         # self.depthwise_dilated_k1 = nn.Conv2d(dim, dim, (5,1), stride=1, padding=0, groups=dim, dilation=3)
        
#     def forward(self, x):
#         u = x.clone()

#         # attn = self.pointwise(x)
#         # attn = self.depthwise_1k(attn)
#         # attn = self.depthwise_dilated_1k(attn)     

#         # attn = self.depthwise_k1(attn)
#         # attn = self.depthwise_dilated_k1(attn)
        

#         attn = self.depthwise_1k(x)
        
#         attn = self.depthwise_dilated_1k(attn)     

#         attn = self.depthwise_k1(attn)
#         attn = self.depthwise_dilated_k1(attn)
        
#         attn = self.pointwise(attn)

#         return u * attn

# class Attention(nn.Module):
#     """Constructs a ECA module.

#     Args:
#         channel: Number of channels of the input feature map
#         k_size: Adaptive selection of kernel size
#     """
#     def __init__(self, channel, k_size=3):
#         super(Attention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv1 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         y = self.avg_pool(x)
#         y = self.conv1(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
#         y = self.sigmoid(y)

#         return x * y.expand_as(x)

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        # print('mu, var', mu.mean(), var.mean())
        # d.append([mu.mean(), var.mean()])
        y = (x - mu) / (var + eps).sqrt()
        weight, bias, y = weight.contiguous(), bias.contiguous(), y.contiguous()  # avoid cuda error
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        # y, var, weight = ctx.saved_variables
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6, requires_grad=True):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones((channels)), requires_grad=requires_grad))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels), requires_grad=requires_grad))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

### residual Dynamic convolution blockresidule
# class LKDB(nn.Module):
#     def __init__(self, in_channels, out_channels, atten_channels=None, conv=nn.Conv2d):
#         super().__init__()

#         if (atten_channels is None):
#             self.atten_channels = in_channels
#         else:
#             self.atten_channels = atten_channels
#         self.act = nn.GELU()

#         self.conv1 = DyConv(in_channels)
#         self.conv2 = conv(in_channels, in_channels, kernel_size=3, padding=1)

#         self.atten = Attention(self.atten_channels)
#         # self.atten = deformable_LKA(self.atten_channels)
        
#         # self.atten = deformable_LKA(self.atten_channels) 
#         # self.atten = TripletAttention(self.atten_channels)
#         # self.atten = eca_layer(self.atten_channels)
#         # self.atten = MFF(self.atten_channels,1.5,False)
#         # self.norm1 = LayerNorm2d(self.atten_channels)
#         # self.atten = PAM_Module(self.atten_channels)

#         self.c6 = nn.Conv2d(self.atten_channels, out_channels, 1)
#         self.pixel_norm = nn.LayerNorm(out_channels)  # channel-wise
#         default_init_weights([self.pixel_norm], 0.1)

#     def forward(self, input):

#         out = self.conv1(input)
#         out = self.act(out)
#         out = self.conv2(out)

#         # out_fused = self.atten(self.norm1(out))
#         out_fused = self.atten(out)
#         #out_fused = out

#         out_fused = self.c6(out_fused)
#         out_fused = out_fused.permute(0, 2, 3, 1)  # (B, H, W, C)
#         out_fused = self.pixel_norm(out_fused)
#         out_fused = out_fused.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        
#         return out_fused + input
        
#         # return out # + input

class LKDB(nn.Module):

    def __init__(self, in_channels, out_channels, atten_channels=None, conv=nn.Conv2d):
        super().__init__()

        self.dc = self.distilled_channels = in_channels // 2
        self.rc = self.remaining_channels = in_channels
        if (atten_channels is None):
            self.atten_channels = in_channels
        else:
            self.atten_channels = atten_channels

        self.c1_d = nn.Conv2d(in_channels, self.dc, 1)
        self.c1_r = conv(in_channels, self.rc, kernel_size=3, padding=1)
        self.c2_d = nn.Conv2d(self.rc, self.dc, 1)
        self.c2_r = conv(self.rc, self.rc, kernel_size=3, padding=1)
        self.c3_d = nn.Conv2d(self.rc, self.dc, 1)
        self.c3_r = conv(self.rc, self.rc, kernel_size=3, padding=1)

        self.c4 = conv(self.rc, self.dc, kernel_size=3, padding=1)
        self.act = nn.GELU()

        self.c5 = nn.Conv2d(self.dc * 4, self.atten_channels, 1)

        self.atten = Attention(self.atten_channels)
        # self.atten = deformable_LKA(self.atten_channels)
        
        # self.atten = deformable_LKA(self.atten_channels) 
        # self.atten = TripletAttention(self.atten_channels)
        # self.atten = eca_layer(self.atten_channels)
        # self.atten = MFF(self.atten_channels,1.5,False)
        # self.norm1 = LayerNorm2d(self.atten_channels)
        # self.atten = PAM_Module(self.atten_channels)

        self.c6 = nn.Conv2d(self.atten_channels, out_channels, 1)
        self.pixel_norm = nn.LayerNorm(out_channels)  # channel-wise
        default_init_weights([self.pixel_norm], 0.1)

    def forward(self, input):

        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out = self.c5(out)

        # out_fused = self.atten(self.norm1(out))
        out_fused = self.atten(out)
        #out_fused = out

        out_fused = self.c6(out_fused)
        out_fused = out_fused.permute(0, 2, 3, 1)  # (B, H, W, C)
        out_fused = self.pixel_norm(out_fused)
        out_fused = out_fused.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        return out_fused + input
        # return out_fused * input

# class LKDB(nn.Module):

#     def __init__(self, in_channels, out_channels, atten_channels=None, conv=nn.Conv2d):
#         super().__init__()

#         self.dc = self.distilled_channels = in_channels
#         self.rc = self.remaining_channels = in_channels
#         if (atten_channels is None):
#             self.atten_channels = in_channels
#         else:
#             self.atten_channels = atten_channels

#         self.c1_d = nn.Conv2d(in_channels, self.dc, 1)
#         self.c1_r = conv(in_channels, self.rc, kernel_size=3, padding=1)
#         self.c2_d = nn.Conv2d(self.rc, self.dc, 1)
#         self.c2_r = conv(self.rc, self.rc, kernel_size=3, padding=1)
#         self.c3_d = nn.Conv2d(self.rc, self.dc, 1)
#         self.c3_r = conv(self.rc, self.rc, kernel_size=3, padding=1)

#         self.c4 = BSConvU(self.rc, self.dc, kernel_size=3, padding=1)
#         self.act = nn.GELU()

#         # self.c5 = nn.Conv2d(self.dc * 4, self.atten_channels, 1)
#         # self.atten = Attention(self.atten_channels)

#         self.c6 = nn.Conv2d(self.atten_channels, out_channels, 1)
#         self.pixel_norm = nn.LayerNorm(out_channels)  # channel-wise
#         default_init_weights([self.pixel_norm], 0.1)
        
#         self.Attention1 = Attention(self.atten_channels)
#         self.Attention2 = Attention(self.atten_channels)
#         self.Attention3 = Attention(self.atten_channels)
#         self.Attention4 = Attention(self.atten_channels)
#         self.Attention_conv = nn.Conv2d(self.rc, self.dc, 1)
#     def forward(self, input):

#         distilled_c1 = self.act(self.c1_d(input))
#         r_c1 = (self.c1_r(input))
#         r_c1 = self.act(r_c1)

#         distilled_c2 = self.act(self.c2_d(r_c1))
#         r_c2 = (self.c2_r(r_c1))
#         r_c2 = self.act(r_c2)

#         distilled_c3 = self.act(self.c3_d(r_c2))
#         r_c3 = (self.c3_r(r_c2))
#         r_c3 = self.act(r_c3)

#         r_c4 = self.act(self.c4(r_c3))

#         # out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
#         # out = self.c5(out)
        
#         Attention1 = self.Attention1(distilled_c1)
#         Attention2 = self.Attention2(distilled_c2)
#         Attention3 = self.Attention3(distilled_c3)
#         Attention4 = self.Attention4(r_c4)
#         Attention = Attention1 + Attention2 + Attention3 + Attention4
#         Attention = self.Attention_conv(Attention)
#         out_fused = r_c4 * Attention

#         out_fused = self.c6(out_fused)
#         out_fused = out_fused.permute(0, 2, 3, 1)  # (B, H, W, C)
#         out_fused = self.pixel_norm(out_fused)
#         out_fused = out_fused.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

#         return out_fused + input
#         # return out_fused * input

def UpsampleOneStep(in_channels, out_channels, upscale_factor=4):
    """
    Upsample features according to `upscale_factor`.
    """
    conv = nn.Conv2d(in_channels, out_channels * (upscale_factor**2), 3, 1, 1)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return nn.Sequential(*[conv, pixel_shuffle])


class Upsampler_rep(nn.Module):

    def __init__(self, in_channels, out_channels, upscale_factor=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels * (upscale_factor**2), 1)
        self.conv3 = nn.Conv2d(in_channels, out_channels * (upscale_factor**2), 3, 1, 1)
        self.conv1x1 = nn.Conv2d(in_channels, in_channels * 2, 1)
        self.conv3x3 = nn.Conv2d(in_channels * 2, out_channels * (upscale_factor**2), 3)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        v1 = F.conv2d(x, self.conv1x1.weight, self.conv1x1.bias, padding=0)
        v1 = F.pad(v1, (1, 1, 1, 1), 'constant', 0)
        b0_pad = self.conv1x1.bias.view(1, -1, 1, 1)
        v1[:, :, 0:1, :] = b0_pad
        v1[:, :, -1:, :] = b0_pad
        v1[:, :, :, 0:1] = b0_pad
        v1[:, :, :, -1:] = b0_pad
        v2 = F.conv2d(v1, self.conv3x3.weight, self.conv3x3.bias, padding=0)
        out = self.conv1(x) + self.conv3(x) + v2
        return self.pixel_shuffle(out)
