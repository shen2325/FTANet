import torch
import torch.nn as nn

class simam_module(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(simam_module, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)

class FourierUnit(nn.Module):
    def __init__(self, dim, groups=1, fft_norm='ortho'):
        super().__init__()
        self.groups = groups
        self.fft_norm = fft_norm
        self.conv_layer = nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=1, stride=1,
                                    padding=0, groups=self.groups, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        batch, c, h, w = x.size()
        # print("Input shape:", x.shape)
        r_size = x.size()

        # 1. FFT
        ffted = torch.fft.rfft2(x, norm=self.fft_norm)  # [B, C, H, W//2+1], complex

        # 2. 转为实部/虚部的真实张量
        ffted = torch.view_as_real(ffted)  # [B, C, H, W//2+1, 2]

        # 3. 调整维度，使通道变为 C*2
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # [B, C, 2, H, W//2+1]
        ffted = ffted.view(batch, c * 2, h, ffted.size(-1))  # [B, C*2, H, W//2+1]
        # print("FFT shape:", ffted.shape)
        # 4. 频域卷积
        ffted = self.conv_layer(ffted)  # [B, C*2, H, W//2+1]
        ffted = self.act(ffted)

        # 5. 恢复回复数形式
        ffted = ffted.view(batch, c, 2, h, ffted.size(-1))  # [B, C, 2, H, W//2+1]
        ffted = ffted.permute(0, 1, 3, 4, 2).contiguous()  # [B, C, H, W//2+1, 2]
        ffted = torch.view_as_complex(ffted)  # [B, C, H, W//2+1], complex

        # 6. IFFT
        output = torch.fft.irfft2(ffted, s=r_size[2:], norm=self.fft_norm)
        return output


class CrossModalFourierAttention(nn.Module):
    def __init__(self, dim):
        super(CrossModalFourierAttention, self).__init__()
        self.fourier_inp1 = FourierUnit(dim=dim)
        self.fourier_inp2 = FourierUnit(dim=dim)
        self.query = nn.Conv2d(dim, dim, kernel_size=1)
        self.key = nn.Conv2d(dim, dim, kernel_size=1)
        self.value = nn.Conv2d(dim, dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inp1, inp2):
        inp1_fourier = self.fourier_inp1(inp1)
        inp2_fourier = self.fourier_inp2(inp2)
        b, c, h, w = inp1_fourier.size()
        
        # Attention from inp1 to inp2
        q1 = self.query(inp1_fourier).view(b, c, -1).permute(0, 2, 1)
        k1 = self.key(inp2_fourier).view(b, c, -1)
        v1 = self.value(inp2_fourier).view(b, c, -1).permute(0, 2, 1)
        attn1 = torch.bmm(q1, k1) / (c ** 0.5)
        attn1 = self.softmax(attn1)
        out1 = torch.bmm(attn1, v1).permute(0, 2, 1).view(b, c, h, w)
        attended_inp1 = out1 + inp1_fourier  # Residual connection
        
        # Attention from inp2 to inp1
        q2 = self.query(inp2_fourier).view(b, c, -1).permute(0, 2, 1)
        k2 = self.key(inp1_fourier).view(b, c, -1)
        v2 = self.value(inp1_fourier).view(b, c, -1).permute(0, 2, 1)
        attn2 = torch.bmm(q2, k2) / (c ** 0.5)
        attn2 = self.softmax(attn2)
        out2 = torch.bmm(attn2, v2).permute(0, 2, 1).view(b, c, h, w)
        attended_inp2 = out2 + inp2_fourier  # Residual connection
        
        return attended_inp1, attended_inp2

class FCF(nn.Module):
    def __init__(self, inp, out):
        super(FCF, self).__init__()
        C = int(inp / 2)
        self.cross_fourier_attn = CrossModalFourierAttention(dim=C)
        self.pre_siam = simam_module()
        self.lat_siam = simam_module()
        out_1 = C
        self.conv_1 = nn.Conv2d(inp, out_1, padding=1, kernel_size=3, groups=out_1, dilation=1)
        self.conv_2 = nn.Conv2d(inp, out_1, padding=2, kernel_size=3, groups=out_1, dilation=2)
        self.conv_3 = nn.Conv2d(inp, out_1, padding=3, kernel_size=3, groups=out_1, dilation=3)
        self.conv_4 = nn.Conv2d(inp, out_1, padding=4, kernel_size=3, groups=out_1, dilation=4)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_1 * 4, out_1, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_1),
            nn.ReLU(inplace=True)
        )
        self.fuse_siam = simam_module()
        self.out = nn.Sequential(
            nn.Conv2d(out_1, out, kernel_size=3, padding=1),
            nn.BatchNorm2d(out),
            nn.ReLU(inplace=True)
        )

    def forward(self, inp1, inp2, last_feature=None):
        attended_inp1, attended_inp2 = self.cross_fourier_attn(inp1, inp2)
        x = torch.cat([attended_inp1, attended_inp2], dim=1)
        c1 = self.conv_1(x)
        c2 = self.conv_2(x)
        c3 = self.conv_3(x)
        c4 = self.conv_4(x)
        cat = torch.cat([c1, c2, c3, c4], dim=1)
        fuse = self.fuse(cat)
        inp1_siam = self.pre_siam(inp1)
        inp2_siam = self.lat_siam(inp2)
        inp1_mul = torch.mul(inp1_siam, fuse)
        inp2_mul = torch.mul(inp2_siam, fuse)
        fuse = self.fuse_siam(fuse)
        if last_feature is None:
            out = self.out(fuse + inp1_mul + inp2_mul)
        else:
            out = self.out(fuse + inp1_mul + inp2_mul + last_feature)
        return out

if __name__ == '__main__':
    input_channels = 128
    output_channels = 256
    batch_size = 1
    height, width = 16, 16
    fcf = FCF(inp=input_channels, out=output_channels)
    inp1 = torch.rand(batch_size, input_channels // 2, height, width)
    inp2 = torch.rand(batch_size, input_channels // 2, height, width)
    last_feature = torch.rand(batch_size, input_channels // 2, height, width)
    output = fcf(inp1, inp2, last_feature)
    print("inp1 shape:", inp1.shape)
    print("inp2 shape:", inp2.shape)
    print("Output shape:", output.shape)