import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


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


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def computeConfusionMatrix(confusion_matrix, show):
    f1Sum = torch.zeros(5)
    # print(confusion_matrix)
    Ppr = AvgrageMeter()
    Sen = AvgrageMeter()
    Spe = AvgrageMeter()
    Acc = AvgrageMeter()
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
        Ppr.update(ppr)
        Sen.update(sen)
        Acc.update(acc)
        Spe.update(spe)
        belta = 1.0
        if sen > 0 and ppr > 0:
            f1Sum[m] = ((1 + belta) * sen * ppr / (sen + belta * ppr))
        if show:
            print('label=%d,TP=%d, FN=%d, TN=%d, FP=%d' % (m, TP, FN, TN, FP))
            print('label=%d,acc=%.2f, ppr=%.2f, sen=%.2f, spe=%.2f ' % (m, acc * 100, ppr * 100, sen * 100, spe * 100))
    if show:
        f1 = 2 * Ppr.avg * Sen.avg / (Ppr.avg + Sen.avg)
        print('avg_acc=%.2f avg_ppr=%.2f avg_sen=%.2f avg_spe=%.2f total_f1:%.2f' % (
            Acc.avg * 100, Ppr.avg * 100, Sen.avg * 100, Spe.avg * 100, f1*100))
    return f1Sum


def compute_ppr_sen(outputs, labels):
    confusion_matrix2 = torch.zeros(5, 5)
    correct = 0
    num = 0
    _, predict = torch.max(outputs, 1)
    # print(outputs)
    # print(predict)
    correct += (predict.view(-1) == labels.view(-1)).sum()

    num += outputs.size(0)
    # print(predict, labels, correct, num)
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


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
