import torch
import torch.nn as nn

OPS = {
    'none': lambda C, stride, affine: Zero(stride),

    'avg_pool_1x3': lambda C, stride, affine: nn.AvgPool1d(3, stride=stride, padding=1, count_include_pad=False),
    'avg_pool_1x5': lambda C, stride, affine: nn.AvgPool1d(5, stride=stride, padding=2, count_include_pad=False),
    'avg_pool_1x7': lambda C, stride, affine: nn.AvgPool1d(7, stride=stride, padding=3, count_include_pad=False),

    'max_pool_1x3': lambda C, stride, affine: nn.MaxPool1d(3, stride=stride, padding=1),
    'max_pool_1x5': lambda C, stride, affine: nn.MaxPool1d(5, stride=stride, padding=2),
    'max_pool_1x7': lambda C, stride, affine: nn.MaxPool1d(7, stride=stride, padding=3),

    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_1x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_1x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_1x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'sep_conv_1x9': lambda C, stride, affine: SepConv(C, C, 9, stride, 4, affine=affine),
    'sep_conv_1x11': lambda C, stride, affine: SepConv(C, C, 11, stride, 5, affine=affine),
    'sep_conv_1x13': lambda C, stride, affine: SepConv(C, C, 13, stride, 6, affine=affine),
    'sep_conv_1x15': lambda C, stride, affine: SepConv(C, C, 15, stride, 7, affine=affine),
    'sep_conv_1x17': lambda C, stride, affine: SepConv(C, C, 17, stride, 8, affine=affine),
    'sep_conv_1x19': lambda C, stride, affine: SepConv(C, C, 19, stride, 9, affine=affine),
    'sep_conv_1x21': lambda C, stride, affine: SepConv(C, C, 21, stride, 10, affine=affine),
    'sep_conv_1x23': lambda C, stride, affine: SepConv(C, C, 23, stride, 11, affine=affine),
    'sep_conv_1x25': lambda C, stride, affine: SepConv(C, C, 25, stride, 12, affine=affine),
    'sep_conv_1x27': lambda C, stride, affine: SepConv(C, C, 27, stride, 13, affine=affine),

}


class ReLUConvBN(nn.Module):
    # 我们使用ReLU-Conv-BN顺序进行卷积运算
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv1d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


# class DilConv(nn.Module):
#
#     def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
#         super(DilConv, self).__init__()
#         self.op = nn.Sequential(
#             nn.ReLU(inplace=False),
#             nn.Conv1d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
#                       groups=C_in, bias=False),
#             nn.Conv1d(C_in, C_out, kernel_size=1, padding=0, bias=False),
#             nn.BatchNorm1d(C_out, affine=affine),
#         )
#
#     def forward(self, x):
#         return self.op(x)


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv1d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.BatchNorm1d(C_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        # print(x.shape)
        # x = torch.zeros(x.shape[0], x.shape[1] , x.shape[2]//2).cuda()
        return x[:, :, ::self.stride].mul(0.)


class FactorizedReduce(nn.Module):
    # 因式分解
    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv1d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv1d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm1d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        # print(self.conv_1(x).shape, self.conv_2(x).shape)
        out = torch.cat([self.conv_1(x), self.conv_2(x)], dim=1)
        # print(out.shape, self.bn)
        out = self.bn(out)
        return out
