import os
import sys
import time
import glob
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import datetime
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


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
        x = self.layer2(conv1.view(conv1.size(0), -1))
        return x


def computeConfusionMatrix(confusion_matrix, show):
    f1Sum = torch.zeros(5)
    for m in range(len(confusion_matrix)):
        TP = 0
        FN = 0
        TN = 0
        FP = 0
        for n in range(len(confusion_matrix[m])):
            if m == n:
                TP = confusion_matrix[m][n]
            else:
                FN += confusion_matrix[m][n]
        for n2 in range(len(confusion_matrix[m])):
            if n2 != m:
                FP += confusion_matrix[n2][m]
        for s in range(5):
            for t in range(5):
                if m != s and m != t:
                    TN += confusion_matrix[s][t]
        # print('label=%d TP=%d,FN=%d, TN=%d, FP=%d ' % (m, TP, FN, TN, FP))
        acc = (TP + TN) * 1.0 / (TP + TN + FP + FN)
        sen = TP * 1.0 / (TP + FN)
        spe = TN * 1.0 / (TN + FP)
        ppr = TP * 1.0 / (TP + FP)
        belta = 1.0
        if sen > 0 and ppr > 0:
            f1Sum[m] = ((1 + belta) * sen * ppr / (sen + belta * ppr))
        if show:
            print('label=%d,TP=%d, FN=%d, TN=%d, FP=%d' % (m, TP, FN, TN, FP))
            print('label=%d,acc=%.2f, ppr=%.2f, sen=%.2f, spe=%.2f ' % (m, acc * 100, ppr * 100, sen * 100, spe * 100))
    return f1Sum


def compute_ppr_sen(outputs, labels):
    confusion_matrix2 = torch.zeros(5, 5)
    correct = 0
    num = 0
    _, predict = torch.max(outputs, 1)
    correct += (predict.view(-1) == labels.view(-1)).sum()
    num += outputs.size(0)
    for k in range(labels.size(0)):
        for t in range(5):
            if labels[k].item() == t:
                for p in range(5):
                    if predict[k].item() == p:
                        confusion_matrix2[t][p] += 1
                        # if p == 4:
                        #     print("confusion_matrix2[%d][%d]=%d" % (t, p, confusion_matrix2[t][p]))
    # print(confusion_matrix2)
    f1 = computeConfusionMatrix(confusion_matrix2, False)
    return f1


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(2)
    torch.cuda.set_device(0)
    cudnn.benchmark = True
    torch.manual_seed(2)
    cudnn.enabled = True
    torch.cuda.manual_seed(2)
    model = Network9layersNet(1, 5)
    model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        0.025,
        momentum=0.9,
        weight_decay=3e-4
    )

    train_data = torch.load(os.path.join('1_lead_data', 'train_dataset.pt'))
    valid_data = torch.load(os.path.join('1_lead_data', 'valid_dataset.pt'))
    print(len(train_data), len(valid_data))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=256, shuffle=True, pin_memory=True, num_workers=0)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=256, shuffle=False, pin_memory=True, num_workers=0)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(600))
    for epoch in range(600):
        # start_time = datetime.datetime.now()
        scheduler.step()
        print('epoch %d lr %e'% (epoch, scheduler.get_lr()[0]))

        train_f1 = train(train_queue, model, criterion, optimizer)
        print('train_f1 %s', train_f1)
        infer(valid_queue, model, criterion)
        # utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, model, criterion, optimizer):
    model.train()
    correct = 0
    val_num = 0
    for step, (input, target) in enumerate(train_queue):
        input = Variable(input.view(-1, 1, 300).type(torch.FloatTensor), requires_grad=False).cuda()
        target = Variable(target.squeeze().type(torch.LongTensor), requires_grad=False).cuda()
        optimizer.zero_grad()
        logits = model(input)
        _, val_predict = torch.max(logits, 1)
        correct += (val_predict.view(-1) == target.view(-1)).sum()
        val_num += input.size(0)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        # 求对应的f1-score
        f1 = compute_ppr_sen(logits, target)
        # print(step)
        if step % 100 == 0:
            print('train_f1 %03d %f %s' % (step, loss, f1))
            # logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top2.avg)
    accuracy = float(float(correct * 100) / float(val_num))
    print('train_accuracy:%.4f [%d,%d]' % (accuracy, correct, val_num))
    return f1


def infer(valid_queue, model, criterion):
    objs = AvgrageMeter()
    # top1 = utils.AvgrageMeter()
    # top5 = utils.AvgrageMeter()
    model.eval()

    correct = 0
    val_num = 0
    confusion_matrix2 = torch.zeros(5, 5)

    for step, (input, target) in enumerate(valid_queue):
        input = Variable(input.view(-1, 1, 300).type(torch.FloatTensor), volatile=True).cuda()
        target = Variable(target.squeeze().type(torch.LongTensor), volatile=True).cuda()
        logits = model(input)
        loss = criterion(logits, target)

        n = input.size(0)
        objs.update(loss.item(), n)

        _, val_predict = torch.max(logits, 1)
        correct += (val_predict.view(-1) == target.view(-1)).sum()
        val_num += input.size(0)
        for k in range(target.size(0)):
            for t in range(5):
                if target[k].item() == t:
                    for p in range(5):
                        if val_predict[k].item() == p:
                            confusion_matrix2[t][p] += 1
    accuracy = float(float(correct * 100) / float(val_num))
    # print(accuracy)
    # objs.update(loss.data[0], n)
    computeConfusionMatrix(confusion_matrix2, True)
    print('accuracy:%.4f [%d,%d] loss:%f' % (accuracy, correct, val_num, objs.avg))
    print('confusion_matrix:%s' % (confusion_matrix2))
    return accuracy, objs.avg


if __name__ == '__main__':
    main()
