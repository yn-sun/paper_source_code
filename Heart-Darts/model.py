import numpy
import torch
import torch.nn as nn
from operations import *
from torch.autograd import Variable
from utils import drop_path

data_channel = 1


class Cell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            # stride = 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


class AuxiliaryHeadCIFAR(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),  # image size = 2 x 2
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class AuxiliaryHeadImageNet(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 14x14"""
        super(AuxiliaryHeadImageNet, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            # NOTE: This batchnorm was omitted in my earlier implementation due to a typo.
            # Commenting it out for consistency with the experiments in the paper.
            # nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class NetworkCIFAR(nn.Module):

    def __init__(self, C, num_classes, layers, auxiliary, genotype):
        super(NetworkCIFAR, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        stem_multiplier = 3
        C_curr = stem_multiplier * C
        # C_curr = 32
        print(C_curr)
        self.stem = nn.Sequential(
            nn.Conv1d(data_channel, C_curr, 5, stride=2, padding=1, bias=False),  # 前三个参数分别是输入图片的通道数，卷积核的数量，卷积核的大小
            nn.BatchNorm1d(C_curr),  # BatchNorm2d对minibatch 3d数据组成的4d输入进行batchnormalization操作，num_features为(N,C,H,W)的C
            nn.ReLU(inplace=False),
            nn.MaxPool1d(3, stride=2)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = s1 = self.stem(input)
        t0, t1 = s0, s1
        # print(t0.size(), t1.size())
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            # if i == 0 or i == 1 * self._layers // 3 or i == 2 * self._layers // 3:
            #     t1 = s1
            # elif i == (1 * self._layers // 3) - 1 or i == (2 * self._layers // 3) - 1:
            #     # print(t1.size(),s0.size())
            #     s0 = nn.ReLU(inplace=True)(s0 + t1)
            #     s1 = nn.ReLU(inplace=True)(s1 + t1)

            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)

        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux


class NetworkImageNet(nn.Module):

    def __init__(self, C, num_classes, layers, auxiliary, genotype):
        super(NetworkImageNet, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary

        self.stem0 = nn.Sequential(
            nn.Conv1d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(C // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(C // 2, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(C),
        )

        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv1d(C, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(C),
        )

        C_prev_prev, C_prev, C_curr = C, C, C

        self.cells = nn.ModuleList()
        reduction_prev = True
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AvgPool1d(7)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = self.stem0(input)
        s1 = self.stem1(s0)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux


class Network9layersNet(nn.Module):
    def __init__(self, C, num_classes):
        super(Network9layersNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(C, 5, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=False),
            nn.MaxPool1d(2, stride=2, padding=0),
            nn.Conv1d(5, 10, kernel_size=4, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=False),
            nn.MaxPool1d(2, stride=2, padding=0),
            nn.Conv1d(10, 20, kernel_size=4, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=False),
            nn.MaxPool1d(2, stride=2, padding=0)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(35 * 20, 30),
            nn.ReLU(inplace=False),
            nn.Linear(30, 20),
            nn.ReLU(inplace=False),
            nn.Linear(20, num_classes)
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        # print(conv1.size())
        x = self.layer2(conv1.view(conv1.size(0), -1))
        # print(numpy.array(x).shape)
        return x


class Network11layersNet(nn.Module):
    def __init__(self, C, num_classes):
        super(Network11layersNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(C, 3, kernel_size=27, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=False),
            nn.MaxPool1d(2, stride=2, padding=0),
            nn.Conv1d(3, 10, kernel_size=14, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=False),
            nn.MaxPool1d(2, stride=2, padding=0),
            nn.Conv1d(10, 10, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=False),
            nn.MaxPool1d(2, stride=2, padding=0),
            nn.Conv1d(10, 10, kernel_size=4, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=False),
            nn.MaxPool1d(2, stride=2, padding=0)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(13 * 10, 30),
            nn.ReLU(inplace=False),
            nn.Linear(30, 10),
            nn.ReLU(inplace=False),
            nn.Linear(10, num_classes)
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        # print(conv1.size())
        x = self.layer2(conv1.view(conv1.size(0), -1))

        return x


class ResNet_31(nn.Module):
    def __init__(self, C, num_classes):
        super(ResNet_31, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=1, stride=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=False),
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=False),
            nn.Conv1d(32, 128, kernel_size=1, stride=1),
            nn.BatchNorm1d(128)
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=1, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=False),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=False),
            nn.Conv1d(64, 256, kernel_size=1, stride=1),
            nn.BatchNorm1d(256)
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=1, stride=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=False),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=False),
            nn.Conv1d(128, 512, kernel_size=1, stride=1),
            nn.BatchNorm1d(512)
        )

        self.block4 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1, stride=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=False),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=False),
            nn.Conv1d(256, 1024, kernel_size=1, stride=1),
            nn.BatchNorm1d(1024)
        )

        self.preprocess = nn.Sequential(
            nn.Conv1d(C, 32, kernel_size=5, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=False),
            nn.MaxPool1d(3, stride=2, padding=1)
        )
        self.global_avg_pool = nn.AvgPool1d(3, stride=10)
        self.layer2 = nn.Sequential(

            nn.Linear(1024, 40),
            nn.Linear(40, num_classes)
        )
        self.skip0 = nn.Sequential(
            nn.Conv1d(32, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128)
        )
        self.skip1 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256)
        )
        self.skip2 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512)
        )
        self.skip3 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(1024)
        )
        self.skip4 = nn.Sequential(
            nn.Conv1d(1024, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(1024)
        )

    def forward(self, x):
        data = self.preprocess(x)
        t = self.skip0(data)
        data = self.block1(data)

        data = nn.ReLU(inplace=False)(data + t)
        # print(data.size())
        t = self.skip1(data)
        # print(t.size(), data.size())
        data = self.block2(data)

        data = nn.ReLU(inplace=False)(data + t)
        t = self.skip2(data)

        data = self.block3(data)
        data = nn.ReLU(inplace=False)(data + t)
        t = self.skip3(data)

        data = self.block4(data)
        data = nn.ReLU(inplace=False)(data + t)

        data = self.global_avg_pool(data)
        data = data.view(-1, 1024)
        output = self.layer2(data)
        # print(output.size())

        return output


class ResNet_25(nn.Module):
    def __init__(self, C, num_classes):
        super(ResNet_25, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=1, stride=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=False),
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=False),
            nn.Conv1d(32, 128, kernel_size=1, stride=1),
            nn.BatchNorm1d(128)
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=1, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=False),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=False),
            nn.Conv1d(64, 256, kernel_size=1, stride=1),
            nn.BatchNorm1d(256)
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=1, stride=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=False),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=False),
            nn.Conv1d(128, 512, kernel_size=1, stride=1),
            nn.BatchNorm1d(512)
        )

        self.preprocess = nn.Sequential(
            nn.Conv1d(C, 32, kernel_size=5, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=False),
            nn.MaxPool1d(3, stride=2, padding=1)
        )
        self.global_avg_pool = nn.AvgPool1d(3, stride=19)
        self.layer2 = nn.Sequential(
            nn.Linear(512, 40),
            nn.Linear(40, num_classes)
        )
        self.skip0 = nn.Sequential(
            nn.Conv1d(32, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128)
        )
        self.skip1 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256)
        )
        self.skip2 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512)
        )

    def forward(self, x):
        data = self.preprocess(x)
        t = self.skip0(data)

        data = self.block1(data)
        data = nn.ReLU(inplace=False)(data + t)
        t = self.skip1(data)

        data = self.block2(data)
        data = nn.ReLU(inplace=False)(data + t)
        t = self.skip2(data)

        data = self.block3(data)
        data = nn.ReLU(inplace=False)(data + t)

        data = self.global_avg_pool(data)
        # print(data.size())
        data = data.view(-1, 512)
        output = self.layer2(data)
        # print(output.size())

        return output
