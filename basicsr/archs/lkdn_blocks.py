import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.archs.arch_util import default_init_weights


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
#         kernel_size=[1,3,5,7]
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
#         padding1 = kernel_size[1]//2
#         self.dw1 = torch.nn.Conv2d(
#                 in_channels=out_channels//self.blocks,
#                 out_channels=out_channels//self.blocks,
#                 kernel_size=kernel_size[1],
#                 stride=stride,
#                 padding=padding1,
#                 dilation=dilation,
#                 groups=out_channels//self.blocks,
#                 bias=bias,
#                 padding_mode=padding_mode,
#         )
#         padding2 = kernel_size[2]//2
#         self.dw2 = torch.nn.Conv2d(
#                 in_channels=out_channels//self.blocks,
#                 out_channels=out_channels//self.blocks,
#                 kernel_size=kernel_size[2],
#                 stride=stride,
#                 padding=padding2,
#                 dilation=dilation,
#                 groups=out_channels//self.blocks,
#                 bias=bias,
#                 padding_mode=padding_mode,
#         )
#         padding3 = kernel_size[3]//2
#         self.dw3 = torch.nn.Conv2d(
#                 in_channels=out_channels-3*(out_channels//self.blocks),
#                 out_channels=out_channels-3*(out_channels//self.blocks),
#                 kernel_size=kernel_size[3],
#                 stride=stride,
#                 padding=padding3,
#                 dilation=dilation,
#                 groups=out_channels-3*(out_channels//self.blocks),
#                 bias=bias,
#                 padding_mode=padding_mode,
#         )

#     def forward(self, fea):
#         fea = self.pw(fea)
#         # print('self.blocks:',self.blocks)
#         self.split_channels = [fea.shape[1] // self.blocks, fea.shape[1] // self.blocks,fea.shape[1] // self.blocks,fea.shape[1]- 3*(fea.shape[1] // self.blocks)]
#         # print('!!!!!!!!!!!!!!!!!!!!!!!!!self.split_channels:',self.split_channels)
#         fea0,fea1,fea2,fea3= torch.split(fea, self.split_channels, dim=1)

#         fea0 = self.dw0(fea0)
#         fea1 = self.dw1(fea1)
#         fea2 = self.dw2(fea2)  
#         fea3 = self.dw3(fea3)
#         fea = torch.cat((fea0,fea1,fea2,fea3),1)

#         return fea

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
        # kernel_size=[1,3,5]
        kernel_size=[3,5,7]
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
                in_channels=out_channels-2*out_channels//self.blocks,
                out_channels=out_channels-2*out_channels//self.blocks,
                kernel_size=kernel_size[2],
                stride=stride,
                padding=padding2,
                dilation=dilation,
                groups=out_channels-2*out_channels//self.blocks,
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
        self.split_channels = [fea.shape[1] // self.blocks, fea.shape[1] // self.blocks,fea.shape[1]- 2*fea.shape[1] // self.blocks]
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

class eca_layer1(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        #self.acti = nn.LeakyReLU(0.2, inplace=True)
        #self.conv2 = nn.Conv2d(channel,channel,kernel_size=1,padding=0,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        
        #y = soft_pool2d(x)
        #print(y.size())
        #y = self.max_pool(x)
        # Two different branches of ECA module
        y = self.conv1(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        return x * y.expand_as(x)

class eca_layer2(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer2, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        #self.acti = nn.LeakyReLU(0.2, inplace=True)
        #self.conv2 = nn.Conv2d(channel,channel,kernel_size=1,padding=0,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        # y = self.avg_pool(x)
        
        #y = soft_pool2d(x)
        #print(y.size())
        y = self.max_pool(x)
        # Two different branches of ECA module
        y = self.conv1(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        return x * y.expand_as(x)



# class Attention(nn.Module):

#     def __init__(self, dim):
#         super().__init__()
#         self.pointwise = nn.Conv2d(dim, dim, 1)
#         self.depthwise = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
#         self.depthwise_dilated = nn.Conv2d(dim, dim, 5, stride=1, padding=6, groups=dim, dilation=3)

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
      


class Attention(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.pointwise = nn.Conv2d(dim, dim, 1)
        
        self.depthwise_1k = nn.Conv2d(dim, dim, (1,9), padding=4, groups=dim)
        self.depthwise_k1 = nn.Conv2d(dim, dim, (9,1), padding=0, groups=dim)
        self.depthwise_dilated_1k = nn.Conv2d(dim, dim, (1,9), stride=1, padding=20, groups=dim, dilation=5)
        self.depthwise_dilated_k1 = nn.Conv2d(dim, dim, (9,1), stride=1, padding=0, groups=dim, dilation=5)

        # self.depthwise_1k = nn.Conv2d(dim, dim, (1,5), padding=2, groups=dim)
        # self.depthwise_k1 = nn.Conv2d(dim, dim, (5,1), padding=0, groups=dim)
        # self.depthwise_dilated_1k = nn.Conv2d(dim, dim, (1,5), stride=1, padding=6, groups=dim, dilation=3)
        # self.depthwise_dilated_k1 = nn.Conv2d(dim, dim, (5,1), stride=1, padding=0, groups=dim, dilation=3)
        
    def forward(self, x):
        # u = x.clone()
        # attn = self.pointwise(x)

        # attn = self.depthwise_1k(attn)
        # attn = self.depthwise_dilated_1k(attn)     

        # attn = self.depthwise_k1(attn)
        # attn = self.depthwise_dilated_k1(attn)
        
        # # attn = self.pointwise(attn)
        # return u * attn

        u = x.clone()
        attn = self.pointwise(x)

        attn = self.depthwise_dilated_1k(attn) 
        attn = self.depthwise_1k(attn)
            
        attn = self.depthwise_dilated_k1(attn)
        attn = self.depthwise_k1(attn)
        
        # attn = self.pointwise(attn)
        return u * attn

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

        # self.c4 = BSConvU(self.rc, self.dc, kernel_size=3, padding=1)
        self.c4 = nn.Conv2d(self.rc, self.dc, kernel_size=1)
        self.act = nn.GELU()

        self.c5 = nn.Conv2d(self.dc * 4, self.atten_channels, 1)
        self.atten = Attention(self.atten_channels)
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

        out_fused = self.atten(out)
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
