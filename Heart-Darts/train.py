import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import datetime
from torch.autograd import Variable
import matplotlib.pyplot as plt
from utils import compute_ppr_sen, computeConfusionMatrix
from model import NetworkCIFAR as Network, Network9layersNet, Network11layersNet, ResNet_25, ResNet_31

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--data_channels', type=int, default=1, help='num of init channels')
parser.add_argument('--layers', type=int, default=15, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved _models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
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
    genotype = eval("genotypes.%s" % args.arch)
    logging.info('genotype = %s', genotype)
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
    # model = ResNet_25(args.init_channels, CIFAR_CLASSES)
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()

    criterion = criterion.cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    train_data = torch.load(os.path.join('1_lead_data', 'train_dataset.pt'))
    valid_data = torch.load(os.path.join('1_lead_data', 'valid_dataset.pt'))
    test_data = torch.load(os.path.join('1_lead_data', 'test_dataset.pt'))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=0)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=0)

    test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=0)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    best_acc = 0.0
    for epoch in range(args.epochs):
        # start_time = datetime.datetime.now()
        scheduler.step()
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        train_f1, train_loss = train(train_queue, model, criterion, optimizer)
        logging.info('train_f1 %s train_loss %s', train_f1, train_loss)
        val_acc, val_loss = infer(valid_queue, model, criterion)
        if val_acc > best_acc:
            best_acc = val_acc
            utils.save(model, os.path.join(args.save, 'weights.pt'))

    pretrained_param = torch.load(os.path.join(args.save, 'weights.pt'))
    model.load_state_dict(pretrained_param)
    test_acc, test_loss = infer(test_queue, model, criterion)

    logging.info('test_acc %s test_loss %s', test_acc, test_loss)


def train(train_queue, model, criterion, optimizer):
    model.train()
    objs = utils.AvgrageMeter()  # 用于保存loss的值
    top1 = utils.AvgrageMeter()  # 前1预测正确的概率
    correct = 0
    val_num = 0
    for step, (input, target) in enumerate(train_queue):
        input = Variable(input.view(-1, args.data_channels, 300).type(torch.FloatTensor), requires_grad=False).cuda()
        target = Variable(target.squeeze().type(torch.LongTensor), requires_grad=False).cuda()
        optimizer.zero_grad()
        logits, logits_aux = model(input)
        # logits = model(input)
        _, val_predict = torch.max(logits, 1)
        correct += (val_predict.view(-1) == target.view(-1)).sum()
        val_num += input.size(0)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight * loss_aux
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()

        # 求对应的f1-score
        f1 = compute_ppr_sen(logits, target)
        objs.update(loss.item(), input.size(0))
        top1.update(f1.item(), input.size(0))

    accuracy = float(float(correct * 100) / float(val_num))
    logging.info('train_accuracy:%.4f [%d,%d]', accuracy, correct, val_num)
    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    # top1 = utils.AvgrageMeter()
    # top5 = utils.AvgrageMeter()
    model.eval()

    correct = 0
    val_num = 0
    confusion_matrix2 = torch.zeros(5, 5)

    for step, (input, target) in enumerate(valid_queue):
        input = Variable(input.view(-1, args.data_channels, 300).type(torch.FloatTensor), volatile=True).cuda()
        target = Variable(target.squeeze().type(torch.LongTensor), volatile=True).cuda()
        logits, logits_aux = model(input)
        # logits = model(input)
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
    computeConfusionMatrix(confusion_matrix2, True)
    logging.info('accuracy:%.4f [%d,%d] loss:%f', accuracy, correct, val_num, objs.avg)
    logging.info('confusion_matrix:%s', confusion_matrix2)
    return accuracy, objs.avg


if __name__ == '__main__':
    main()
