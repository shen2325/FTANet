import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# PnPNet: Pull-and-Push Networks for Volumetric Segmentation with Boundary Confusion
# 2D图像分割即插即用特征融合模块

class Attention(nn.Module):
    def __init__(self, dim, num_heads=None, bias=False):
        super(Attention, self).__init__()
        # 动态设置 num_heads，确保 dim 能被 num_heads 整除
        self.num_heads = num_heads if num_heads is not None else max(1, dim // 4)
        if dim % self.num_heads != 0:
            self.num_heads = 1  # 回退到单头注意力
        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(0.)
        self.attn1 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def forward(self, x):
        b, c, h, w = x.shape
        #print(f"Attention 输入形状: {x.shape}")
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v =rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, C, _ = q.shape

        mask1 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask2 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask3 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask4 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        index = torch.topk(attn, k=max(1, int(C/2)), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=max(1, int(C*2/3)), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=max(1, int(C*3/4)), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=max(1, int(C*4/5)), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))

        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)
        attn4 = attn4.softmax(dim=-1)

        out1 = (attn1 @ v)
        out2 = (attn2 @ v)
        out3 = (attn3 @ v)
        out4 = (attn4 @ v)

        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        #print(f"Attention 输出形状: {out.shape}")
        return out

class SDC(nn.Module):
    def __init__(self, in_channels, guidance_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):
        super(SDC, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.conv1 = Conv2dbn(guidance_channels, in_channels, kernel_size=3, padding=1)
        self.theta = theta
        self.guidance_channels = guidance_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        # Top-K 注意力模块，动态设置 num_heads
        self.attention = Attention(dim=in_channels, num_heads=None, bias=False)

        # initialize
        x_initial = torch.randn(in_channels, 1, kernel_size, kernel_size)
        x_initial = self.kernel_initialize(x_initial)

        self.x_kernel_diff = nn.Parameter(x_initial)
        self.x_kernel_diff[:, :, 0, 0].detach()
        self.x_kernel_diff[:, :, 0, 2].detach()
        self.x_kernel_diff[:, :, 2, 0].detach()
        self.x_kernel_diff[:, :, 2, 2].detach()

        guidance_initial = torch.randn(in_channels, 1, kernel_size, kernel_size)
        guidance_initial = self.kernel_initialize(guidance_initial)

        self.guidance_kernel_diff = nn.Parameter(guidance_initial)
        self.guidance_kernel_diff[:, :, 0, 0].detach()
        self.guidance_kernel_diff[:, :, 0, 2].detach()
        self.guidance_kernel_diff[:, :, 2, 0].detach()
        self.guidance_kernel_diff[:, :, 2, 2].detach()

    def kernel_initialize(self, kernel):
        kernel[:, :, 0, 0] = -1
        kernel[:, :, 0, 2] = 1
        kernel[:, :, 2, 0] = 1
        kernel[:, :, 2, 2] = -1
        return kernel

    def forward(self, x, guidance):
        guidance_channels = self.guidance_channels
        in_channels = self.in_channels
        kernel_size = self.kernel_size

        #print(f"SDC 输入 x 形状: {x.shape}")
        #print(f"SDC 输入 guidance 形状: {guidance.shape}")

        guidance = self.conv1(guidance)

        x_diff = F.conv2d(input=x, weight=self.x_kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=1,
                          groups=in_channels)

        guidance_diff = F.conv2d(input=guidance, weight=self.guidance_kernel_diff, bias=self.conv.bias,
                                 stride=self.conv.stride, padding=1, groups=in_channels)
        
        # 差分特征融合
        diff_product = x_diff * guidance_diff * guidance_diff
        #print(f"diff_product 形状: {diff_product.shape}")
        
        # 应用 Top-K 注意力机制
        attn_out = self.attention(diff_product)
        #print(f"attn_out 形状: {attn_out.shape}")
        
        # 融合差分特征和注意力特征
        out = self.conv(diff_product + attn_out)
        #print(f"SDC 输出形状: {out.shape}")
        return out

class TAD(nn.Module):
    def __init__(self, in_channel=3, guidance_channels=2):
        super(TAD, self).__init__()
        self.sdc1 = SDC(in_channel, guidance_channels)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(in_channel)

    def forward(self, feature, guidance):
        #print(f"SDM 输入 feature 形状: {feature.shape}")
        #print(f"SDM 输入 guidance 形状: {guidance.shape}")
        
        boundary_enhanced = self.sdc1(feature, guidance)
        
        #print(f"SDC 输出 boundary_enhanced 形状: {boundary_enhanced.shape}")
        
        boundary = self.relu(self.bn(boundary_enhanced))
        boundary_enhanced = boundary + feature
        
        #print(f"SDM 输出 boundary_enhanced 形状: {boundary_enhanced.shape}")
        return boundary_enhanced

class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels)
        super(Conv2dReLU, self).__init__(conv, bn, relu)

class Conv2dbn(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        bn = nn.BatchNorm2d(out_channels)
        super(Conv2dbn, self).__init__(conv, bn)

class Conv2dGNReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        gelu = nn.GELU()
        gn = nn.GroupNorm(4, out_channels)
        super(Conv2dGNReLU, self).__init__(conv, gn, gelu)

class Conv2dGN(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        gn = nn.GroupNorm(4, out_channels)
        super(Conv2dGN, self).__init__(conv, gn)

if __name__ == '__main__':
    import torch

    # 定义输入张量的形状
    input_shape = (1, 3, 32, 32)  # (batch_size, channels, height, width)
    input_tensor = torch.randn(input_shape)
    guidance_tensor = torch.randn((1, 3, 32, 32))  # 假设引导张量与输入张量大小相同

    # 创建模型
    model = TAD(in_channel=3, guidance_channels=3)
    model.eval()

    # 打印输入张量的形状
    #print("输入张量1的形状:", input_tensor.shape)
    #print("输入张量2的形状:", guidance_tensor.shape)

    # 执行前向传播
    output_tensor = model(input_tensor, guidance_tensor)

    # 打印输出张量的形状
    #print("输出张量的形状:", output_tensor.shape)