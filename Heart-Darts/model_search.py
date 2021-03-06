import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype
import numpy as np
from operations import OPS, FactorizedReduce, ReLUConvBN

data_channel = 2

class MixedOp(nn.Module):

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm1d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction
        # input nodes的结构固定不变，不参与搜索
        # 决定第一个input nodes的结构，取决于前一个cell是否是reduction
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            # 我们使用ReLU-Conv-BN顺序进行卷积运算
            # 第一个input_nodes是cell k-2的输出，cell k-2的输出通道数为C_prev_prev，所以这里操作的输入通道数为C_prev_prev
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        # 第二个input_nodes是cell k-1的输出
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        # 每个cell中有几个nodes
        self._steps = steps  # 每个cell中有steps个节点的连接状态待确定(这里假设4)
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()

        for i in range(self._steps):
            # 遍历当前结点i的所有前驱节点
            for j in range(2 + i):  # 对第i个节点来说，他有j个前驱节点（每个节点的input都由前两个cell的输出和当前cell的前面的节点组成）
                # For the Reduction Cell, we make the initial operation applied to the cell’s inputs have a stride of two to reduce the height and width
                if reduction and j < 2:
                    stride = 2
                else:
                    stride = 1

                # stride = 1
                op = MixedOp(C, stride)
                # print(op)
                # op是构建两个节点之间的混合
                self._ops.append(op)  # 所有边的混合操作添加到ops，list的len为2+3+4+5=14[[],[],...,[]]

    def forward(self, s0, s1, weights):
        # s0,s1是cell的input nodes
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        offset = 0

        for i in range(self._steps):
            # s为当前节点i的output（x(i）)，在ops找到i对应的操作，然后对i的所有前驱节点做相应的操作（调用了MixedOp的forward），然后把结果相加
            # weights是alpha的softmax值
            s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)  # 把当前节点i的output作为下一个节点的输入
            # states中为[s0,s1,b1,b2,b3,b4] b1,b2,b3,b4分别是四个intermediate output的输出
        return torch.cat(states[-self._multiplier:], dim=1)
        # 对intermediate的output进行concat作为当前cell的输出
        # dim=1是指对通道这个维度concat，所以输出的通道数变成原来的4倍


class Network(nn.Module):
    def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
        super(Network, self).__init__()
        self._C = C  # 初始通道数
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps  # 一个cell中有step个节点
        self._multiplier = multiplier
        # stem_multiplier = 3
        C_curr = stem_multiplier * C
        # C_curr = 5
        # C_curr = stem_multiplier * C  # 当前Sequential模块的输出通道数
        print(C_curr)
        self.stem = nn.Sequential(
            nn.Conv1d(data_channel, C_curr, 5, stride=2, padding=1, bias=False),  # 前三个参数分别是输入图片的通道数，卷积核的数量，卷积核的大小
            nn.BatchNorm1d(C_curr),  # BatchNorm2d对minibatch 3d数据组成的4d输入进行batchnormalization操作，num_features为(N,C,H,W)的C
            nn.ReLU(inplace=False),
            nn.MaxPool1d(3, stride=2)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()  # 创建一个空modulelist类型数据
        reduction_prev = False  # 连接的前一个cell是否是reduction cell
        for i in range(layers):  # 网络是8层，在1/3和2/3位置是reduction cell 其他是normal cell，reduction cell的stride是2
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr  # C_prev=multi C_prev=multiplier*C_curr是因为每个cell的输出是4个中间节点concat的，这个concat是在通道这个维度，所以输出的通道数变为原来的4倍

        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()  # 架构参数α初始化

    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input):
        s0 = s1 = self.stem(input)
        # print("s0:", s0.shape, "s1:", s1.shape, "weights:", self.alphas_normal.shape)
        # s0,s1 分别对应于Cell中的input nodes
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = F.softmax(self.alphas_reduce, dim=-1)
            else:
                weights = F.softmax(self.alphas_normal, dim=-1)
            s0, s1 = s1, cell(s0, s1, weights)

        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def _loss(self, input, target):
        target = target.squeeze()
        logits = self(input)
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        self.alphas_normal = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.alphas_reduce = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):
        # 对应论文2.4 选出来权重值大的两个前驱节点，并把(操作，前驱节点)存下来
        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                # 找出来前驱节点的哪两个边的权重最大
                edges = sorted(range(i + 2),
                               key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[
                        :2]  # sorted：对可迭代对象进行排序，key是用来进行比较的元素
                # range(i + 2)表示x取0，1，到i+2 x也就是前驱节点的序号 ，所以W[x]就是这个前驱节点的所有权重[α0,α1,α2,...,α7]
                # max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')) 就是把操作不是NONE的α放到一个list里，得到最大值
                # sorted 就是把每个前驱节点对应的权重最大的值进行逆序排序，然后选出来top2

                # 把这两条边对应的最大权重的操作找到
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))  # 把(操作，前驱节点序号)放到list gene中，[('sep_conv_3x3', 1),...,]
                start = end
                n += 1
            return gene

        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())  # 得到normal cell 的最后选出来的结果
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())  # 得到reduce cell 的最后选出来的结果

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)  # [2,3,4,5] 表示对节点2，3，4，5 concat
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype
