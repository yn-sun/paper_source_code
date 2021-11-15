import os
import sys
import time
import glob
import numpy as np
import torch

import loadData
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable

from utils import compute_ppr_sen, computeConfusionMatrix
from model_search import Network
from architect import Architect
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../dataset', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=100, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--data_channels', type=int, default=2, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.8, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 5


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    # 用于更新w的优化器
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    data2, label = loadData.readSignal_2()
    typeN, typeS, typeV, typeF, typeQ = loadData.splitData(data2, label)
    # train_data是一个数据集，num_train是数据集的样本数
    # train_data, num_train = loadData.constructDataLoader(typeN, typeS, typeV, typeF, typeQ)

    train_dataset = torch.load(os.path.join('2_lead_data', 'train_dataset.pt'))
    valid_dataset = torch.load(os.path.join('2_lead_data', 'valid_dataset.pt'))

    train_queue = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=0)

    valid_queue = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=0)

    # 学习率更新参数，每次迭代调整不同的学习率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(  # 使用余弦退火调度设置各组参数组的学习率
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    # 用来更新 α 结构的
    architect = Architect(model, args)

    # 在挑选normal cell
    for epoch in range(args.epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        genotype = model.genotype()  # 得到normal cell 的最后选出来的结果
        logging.info('genotype = %s', genotype)

        # training
        train_f1, train_loss = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr)
        logging.info('train_f1 %f train_loss %f', train_f1, train_loss)

        # validation
        val_acc, val_confusion_matrix, valid_loss = infer(valid_queue, model, criterion)
        logging.info('val_acc %f valid_loss %f', val_acc, valid_loss)

        computeConfusionMatrix(val_confusion_matrix, True)
        utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr):
    objs = utils.AvgrageMeter()  # 用于保存loss的值
    top1 = utils.AvgrageMeter()  # 前1预测正确的概率
    # top2 = utils.AvgrageMeter()  # 前2预测正确的概率

    for step, (inputs, target) in enumerate(train_queue):  # 每个step取出一个batch，batchsize是64（300个数据对）
        # 启用 BatchNormalization 和 Dropout
        model.train()
        n = inputs.size(0)

        input = Variable(inputs.view(-1, args.data_channels, 300).type(torch.FloatTensor), requires_grad=False,
                         volatile=True).cuda()
        target = Variable(target.squeeze().type(torch.LongTensor), requires_grad=False).cuda()

        # 更新α是用validation set进行更新的，所以我们每次都从valid_queue拿出一个batch传入architect.step()
        # input_search,target_search是对应的valid_data,valid_label
        input_search, target_search = next(
            iter(valid_queue))  # 用于架构参数更新的一个batch 。使用iter(dataloader)返回的是一个迭代器，然后可以使用next访问；
        input_search = Variable(input_search.view(-1, args.data_channels, 300).type(torch.FloatTensor),
                                requires_grad=False, volatile=True).cuda()
        target_search = Variable(target_search.type(torch.LongTensor), requires_grad=False).cuda()

        # 对α进行一步的更新
        architect.step(input, target, input_search, target_search, lr, optimizer,
                       unrolled=args.unrolled)  # unrolled是true就是用论文的公式进行α的更新

        # 在训练集上对w进行一步的更新
        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)  # 预测值logits和真实值target的loss
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()

        # 求对应的f1-score
        f1 = compute_ppr_sen(logits, target)

        objs.update(loss.item(), input.size(0))
        top1.update(f1.item(), input.size(0))

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    # top1 = utils.AvgrageMeter()
    # top2 = utils.AvgrageMeter()
    # 不启用 BatchNormalization 和 Dropout
    model.eval()

    correct = 0
    val_num = 0
    confusion_matrix2 = torch.zeros(5, 5)

    for step, (input, target) in enumerate(valid_queue):
        input = Variable(input.view(-1, args.data_channels, 300).type(torch.FloatTensor), volatile=True).cuda()
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
    return accuracy, confusion_matrix2, objs.avg


if __name__ == '__main__':
    main()
