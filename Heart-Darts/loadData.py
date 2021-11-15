# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import signal

import wfdb as wb
import os
import torch
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
import pywt
from IPython.core.display import display


def readSignal_1():
    file_pre = './dataset/mit-bih-arrhythmia-database-1.0.0/'
    numberSet = ['101', '105', '106', '107', '108', '109', '111', '112', '113', '115', '116', '118', '119', '121',
                 '122', '200', '201', '202', '203', '205', '207', '208', '209', '210', '212', '213', '214', '215',
                 '217', '219', '220', '221', '222', '223', '228', '230', '231', '232', '233', '234']
    data = []
    label = []
    aami_N = ['N', 'L', 'R', 'e', 'j']
    aami_S = ['A', 'a', 'J', 'S']
    aami_V = ['V', 'E']
    aami_F = ['F']
    aami_Q = ['/', 'f', 'Q']
    count = 0
    person_name = []

    for name in numberSet:
        person_data = []
        person_label = []
        file_name = file_pre + name
        person_name.append(name)

        # record的类型是一个(65000,2)的二维数组
        record = wb.rdrecord(file_name, channel_names=['MLII'])

        # # 第一個channel的数据
        record1 = np.array(record.p_signal)[:, 0]
        # # 第二個channel的数据
        # record2 = np.array(record.p_signal)[:, 1]

        data1 = record1.flatten().tolist()
        # data2 = record2.flatten().tolist()

        # 小波去噪
        # 第一个通道去噪后的数据
        rdata1 = denoise(data=data1)
        # 第二个通道去噪后的数据
        # rdata2 = denoise(data=data2)

        # annotation:注解
        annotation = wb.rdann(file_name, 'atr')

        # annotation.ann_len：心拍被标注的数量
        for i in range(annotation.ann_len):
            # annotation.symbol：标注每一个心拍的类型N，L，R等等
            if annotation.symbol[i] in aami_N:
                annotation.symbol[i] = 'N'
            if annotation.symbol[i] in aami_V:
                annotation.symbol[i] = 'V'
            if annotation.symbol[i] in aami_S:
                annotation.symbol[i] = 'S'
            if annotation.symbol[i] in aami_F:
                annotation.symbol[i] = 'F'
            if annotation.symbol[i] in aami_Q:
                annotation.symbol[i] = 'Q'

        # 去掉前后的不稳定数据
        start = 3
        end = 2
        i = start
        j = len(annotation.symbol) - end

        while i < j:
            # 并在尖峰处向前取99个信号点、向后取201个信号点，构成一个完整的心拍
            y = []
            # 一个心拍对应一个y，y由两条通道组成，类型为(2,300)
            y.append(rdata1[annotation.sample[i] - 99:annotation.sample[i] + 201])
            # y.append(rdata2[annotation.sample[i] - 99:annotation.sample[i] + 201])

            # print(np.array(y).shape)
            if (annotation.symbol[i] in ['N', 'S', 'V', 'F', 'Q']):
                person_data.append(y)
                if annotation.symbol[i] == 'N':
                    person_label.append('N')
                if annotation.symbol[i] == 'V':
                    person_label.append('V')
                if annotation.symbol[i] == 'S':
                    person_label.append('S')
                if annotation.symbol[i] == 'F':
                    person_label.append('F')
                if annotation.symbol[i] == 'Q':
                    person_label.append('Q')
            i += 1

        count += 1
        # 数据是预处理后切分出的若干心拍的列表，标签是每个心拍样本对应的心电类型
        person_data = np.array(person_data)
        data.append(person_data)
        label.append(person_label)

    return data, label


def readSignal_2():
    file_pre = './dataset/mit-bih-arrhythmia-database-1.0.0/'
    numberSet = ['101', '105', '106', '107', '108', '109', '111', '112', '113', '115', '116', '118', '119', '121',
                 '122', '200', '201', '202', '203', '205', '207', '208', '209', '210', '212', '213', '214', '215',
                 '217', '219', '220', '221', '222', '223', '228', '230', '231', '232', '233', '234']
    data = []
    label = []
    aami_N = ['N', 'L', 'R', 'e', 'j']
    aami_S = ['A', 'a', 'J', 'S']
    aami_V = ['V', 'E']
    aami_F = ['F']
    aami_Q = ['/', 'f', 'Q']
    count = 0
    person_name = []

    for name in numberSet:
        person_data = []
        person_label = []
        file_name = file_pre + name
        person_name.append(name)

        # record的类型是一个(65000,2)的二维数组
        record = wb.rdrecord(file_name, channel_names=['MLII', 'V1'])

        # 第一個channel的数据
        record1 = np.array(record.p_signal)[:, 0]
        # 第二個channel的数据
        record2 = np.array(record.p_signal)[:, 1]

        data1 = record1.flatten().tolist()
        data2 = record2.flatten().tolist()

        # 小波去噪
        # 第一个通道去噪后的数据
        rdata1 = denoise(data=data1)
        # 第二个通道去噪后的数据
        rdata2 = denoise(data=data2)

        # annotation:注解
        annotation = wb.rdann(file_name, 'atr')

        # annotation.ann_len：心拍被标注的数量
        for i in range(annotation.ann_len):
            # annotation.symbol：标注每一个心拍的类型N，L，R等等
            if annotation.symbol[i] in aami_N:
                annotation.symbol[i] = 'N'
            if annotation.symbol[i] in aami_V:
                annotation.symbol[i] = 'V'
            if annotation.symbol[i] in aami_S:
                annotation.symbol[i] = 'S'
            if annotation.symbol[i] in aami_F:
                annotation.symbol[i] = 'F'
            if annotation.symbol[i] in aami_Q:
                annotation.symbol[i] = 'Q'

        # 去掉前后的不稳定数据
        start = 3
        end = 2
        i = start
        j = len(annotation.symbol) - end

        while i < j:
            # 并在尖峰处向前取99个信号点、向后取201个信号点，构成一个完整的心拍
            y = []
            # 一个心拍对应一个y，y由两条通道组成，类型为(2,300)
            y.append(rdata1[annotation.sample[i] - 99:annotation.sample[i] + 201])
            y.append(rdata2[annotation.sample[i] - 99:annotation.sample[i] + 201])

            # print(np.array(y).shape)
            if (annotation.symbol[i] in ['N', 'S', 'V', 'F', 'Q']):
                person_data.append(y)
                if annotation.symbol[i] == 'N':
                    person_label.append('N')
                if annotation.symbol[i] == 'V':
                    person_label.append('V')
                if annotation.symbol[i] == 'S':
                    person_label.append('S')
                if annotation.symbol[i] == 'F':
                    person_label.append('F')
                if annotation.symbol[i] == 'Q':
                    person_label.append('Q')
            i += 1

        count += 1
        # 数据是预处理后切分出的若干心拍的列表，标签是每个心拍样本对应的心电类型
        person_data = np.array(person_data)
        data.append(person_data)
        label.append(person_label)

    return data, label


def readSignal_qt():
    file_pre = './dataset/qt-database-1.0.0/'
    # numberSet=[]
    numberSet = ['sel302', 'sel16265', 'sele0509', 'sele0211', 'sel16273', 'sele0116', 'sele0124', 'sel873', 'sel16795',
                 'sel308', 'sel847', 'sele0126', 'sel307', 'sele0136', 'sel230', 'sel840', 'sele0607', 'sel853',
                 'sele0606', 'sel820', 'sel15814', 'sele0112', 'sele0129', 'sel306', 'sel231', 'sel891', 'sele0612',
                 'sel213', 'sele0203', 'sel301', 'sele0411', 'sel116', 'sel17453', 'sel811', 'sel16483', 'sel310',
                 'sel16539', 'sele0603', 'sele0170', 'sel14157', 'sele0166', 'sel100', 'sele0406', 'sel871', 'sele0111',
                 'sele0106', 'sel872', 'sel114', 'sele0409', 'sel117', 'sele0121', 'sele0104', 'sele0704', 'sele0114',
                 'sele0303', 'sele0110', 'sel16420', 'sel233', 'sel803', 'sel232', 'sel104', 'sele0133', 'sele0107',
                 'sel883', 'sel103', 'sele0210', 'sele0405', 'sel16786', 'sel102', 'sel821', 'sele0122', 'sel16773',
                 'sele0604', 'sel123', 'sele0609', 'sel221', 'sel17152', 'sel14046', 'sel808', 'sel14172', 'sel223',
                 'sel16272']
    # print(len(numberSet))
    data = []
    label = []
    aami_N = ['N', 'L', 'R', 'e', 'j']
    aami_S = ['A', 'a', 'J', 'S']
    aami_V = ['V', 'E']
    aami_F = ['F']
    aami_Q = ['/', 'f', 'Q']
    count = 0
    person_name = []
    for name in numberSet:
        # if 'dat' in name:
        person_data = []
        person_label = []
        # if name.strip('.dat'):
        file_name = file_pre + name
        if os.path.exists(file_name + '.atr'):
            # numberSet.append(name)
            person_name.append(name)
            # record的类型是一个(65000,2)的二维数组
            record = wb.rdrecord(file_name)

            # # 第一個channel的数据
            record1 = np.array(record.p_signal)[:, 0]

            # # 第二個channel的数据
            # record2 = np.array(record.p_signal)[:, 1]

            data1 = record1.flatten().tolist()

            # data2 = record2.flatten().tolist()

            # 小波去噪
            # 第一个通道去噪后的数据
            rdata1 = denoise(data=data1)
            # 第二个通道去噪后的数据
            # rdata2 = denoise(data=data2)
            # print(os.path.exists(file_name+'.atr'))

            # annotation:注解
            annotation = wb.rdann(file_name, 'atr')

            # annotation.ann_len：心拍被标注的数量
            for i in range(annotation.ann_len):
                # annotation.symbol：标注每一个心拍的类型N，L，R等等
                if annotation.symbol[i] in aami_N:
                    annotation.symbol[i] = 'N'
                if annotation.symbol[i] in aami_V:
                    annotation.symbol[i] = 'V'
                if annotation.symbol[i] in aami_S:
                    annotation.symbol[i] = 'S'
                if annotation.symbol[i] in aami_F:
                    annotation.symbol[i] = 'F'
                if annotation.symbol[i] in aami_Q:
                    annotation.symbol[i] = 'Q'

            # 去掉前后的不稳定数据
            start = 3
            end = 2
            i = start
            j = len(annotation.symbol) - end

            while i < j:
                # 并在尖峰处向前取99个信号点、向后取201个信号点，构成一个完整的心拍
                y = []
                # 一个心拍对应一个y，y由两条通道组成，类型为(2,300)
                y.append(rdata1[annotation.sample[i] - 100:annotation.sample[i] + 120])
                # y.append(rdata2[annotation.sample[i] - 100:annotations.sample[i] + 120])

                if (annotation.symbol[i] in ['N', 'S', 'V', 'F', 'Q']):
                    person_data.append(y)
                    if annotation.symbol[i] == 'N':
                        person_label.append('N')
                    if annotation.symbol[i] == 'V':
                        person_label.append('V')
                    if annotation.symbol[i] == 'S':
                        person_label.append('S')
                    if annotation.symbol[i] == 'F':
                        person_label.append('F')
                    if annotation.symbol[i] == 'Q':
                        person_label.append('Q')
                i += 1

            count += 1
            # 数据是预处理后切分出的若干心拍的列表，标签是每个心拍样本对应的心电类型
            person_data = np.array(person_data)
            data.append(person_data)
            # print(person_data.shape,np.array(data).shape)
            label.append(person_label)
    # print(numberSet)
    return data, label


def readSignal_incart():
    file_pre = './dataset/st-petersburg-incart-12-lead-arrhythmia-database-1.0.0/'
    data = []
    label = []
    aami_N = ['N', 'L', 'R', 'e', 'j']
    aami_S = ['A', 'a', 'J', 'S']
    aami_V = ['V', 'E']
    aami_F = ['F']
    aami_Q = ['/', 'f', 'Q']
    count = 0
    person_name = []

    for name in os.listdir(file_pre):
        if 'dat' in name:
            person_data = []
            person_label = []
            if name.strip('.dat'):
                file_name = file_pre + name.strip('.dat')
                person_name.append(name.strip('.dat'))
                # record的类型是一个(65000,2)的二维数组
                record = wb.rdrecord(file_name, channel_names=['II', 'V1'])

                # 第一個channel的数据
                record1 = np.array(record.p_signal)[:, 0]
                # 第二個channel的数据
                record2 = np.array(record.p_signal)[:, 1]

                data1 = record1.flatten().tolist()
                data2 = record2.flatten().tolist()

                # 小波去噪
                # 第一个通道去噪后的数据
                rdata1 = denoise(data=data1)
                # 第二个通道去噪后的数据
                rdata2 = denoise(data=data2)

                # annotation:注解
                annotation = wb.rdann(file_name, 'atr')
                # annotation.ann_len：心拍被标注的数量
                for i in range(annotation.ann_len):
                    # annotation.symbol：标注每一个心拍的类型N，L，R等等
                    if annotation.symbol[i] in aami_N:
                        annotation.symbol[i] = 'N'
                    if annotation.symbol[i] in aami_V:
                        annotation.symbol[i] = 'V'
                    if annotation.symbol[i] in aami_S:
                        annotation.symbol[i] = 'S'
                    if annotation.symbol[i] in aami_F:
                        annotation.symbol[i] = 'F'
                    if annotation.symbol[i] in aami_Q:
                        annotation.symbol[i] = 'Q'

                # 去掉前后的不稳定数据
                start = 3
                end = 2
                i = start
                j = len(annotation.symbol) - end

                while i < j:
                    # 并在尖峰处向前取99个信号点、向后取201个信号点，构成一个完整的心拍
                    y = []
                    # 一个心拍对应一个y，y由两条通道组成，类型为(2,300)
                    y.append(rdata1[annotation.sample[i] - 99:annotation.sample[i] + 151])
                    y.append(rdata2[annotation.sample[i] - 99:annotation.sample[i] + 151])

                    # print(np.array(y).shape)
                    if (annotation.symbol[i] in ['N', 'S', 'V', 'F']):
                        person_data.append(y)
                        if annotation.symbol[i] == 'N':
                            person_label.append('N')
                        if annotation.symbol[i] == 'V':
                            person_label.append('V')
                        if annotation.symbol[i] == 'S':
                            person_label.append('S')
                        if annotation.symbol[i] == 'F':
                            person_label.append('F')
                        if annotation.symbol[i] == 'Q':
                            person_label.append('Q')
                    i += 1

                count += 1
                # 数据是预处理后切分出的若干心拍的列表，标签是每个心拍样本对应的心电类型
                person_data = np.array(person_data)
                data.append(person_data)
                label.append(person_label)

    return data, label


def splitData(data, label):
    # 将数据分到对应的类别组中
    typeN = []
    typeS = []
    typeV = []
    typeF = []
    typeQ = []
    count = 0
    t=0
    for i in range(len(data)):
        # print(np.array(data[i]).shape)
        for j in range(len(data[i])):
            if label[i][j] == 'N':
                # plt.plot(data[i][j][0])
                # plt.plot(data[i][j][1])
                # plt.show()
                typeN.append(data[i][j])
            elif label[i][j] == 'S':

                typeS.append(data[i][j])
            elif label[i][j] == 'V':
                # print(data[i][j][0])
                # plt.plot(data[i][j][0])
                # plt.plot(data[i][j][1])

                typeV.append(data[i][j])
            elif label[i][j] == 'F':

                typeF.append(data[i][j])
            elif label[i][j] == 'Q':
                typeQ.append(data[i][j])
            count += 1

    # print(np.array(typeN).shape)
    print(' numN =', len(typeN), '\n',
          'numS =', len(typeS), '\n',
          'numV =', len(typeV), '\n',
          'numF =', len(typeF), '\n',
          'numQ =', len(typeQ), '\n',
          'numA =', count)
    return typeN, typeS, typeV, typeF, typeQ


def constructDataLoader(typeN, typeS, typeV, typeF, typeQ):
    labelN = []
    labelS = []
    labelV = []
    labelF = []
    labelQ = []
    for i in range(5):
        if i == 0:
            for j in range(len(typeN)):
                labelN.append(0)
        elif i == 1:
            for j in range(len(typeS)):
                labelS.append(1)
        elif i == 2:
            for j in range(len(typeV)):
                labelV.append(2)
        elif i == 3:
            for j in range(len(typeF)):
                labelF.append(3)
        elif i == 4:
            for j in range(len(typeQ)):
                labelQ.append(4)
    data = typeN + typeS + typeV + typeF + typeQ
    # data = typeN + typeS + typeV + typeF
    label = labelN + labelS + labelV + labelF + labelQ
    # label = labelN + labelS + labelV + labelF
    tensor_data = torch.from_numpy(np.array(data))
    tensor_label = torch.from_numpy(np.array(label))
    torch_dataset = Data.TensorDataset(tensor_data, tensor_label)

    return torch_dataset, len(data)


def tenFoldCrossDataSplit(dataloader, count):
    data10_temp = []
    label10_temp = []
    data10 = []
    label10 = []
    num = count // 10
    for i, (data, label) in enumerate(dataloader):
        if (i + 1) % num == 0:
            data10_temp.append(data[0].numpy())
            label10_temp.append(data[1].numpy())
            data10.append(data10_temp)
            label10.append(label10_temp)
            data10_temp = []
            label10_temp = []
        else:
            data10_temp.append(data[0].numpy())
            label10_temp.append(data[1].numpy())
        if i >= num * 10:
            data10[9].append(data[0].numpy())
            label10[9].append(data[1].numpy())

    return data10, label10


def cunstructValidDataset(dataloader):
    data2 = []
    label2 = []

    for i, (data, label) in enumerate(dataloader):
        data2.append(data.numpy())
        label2.append(label.numpy())

    tensor_data = torch.from_numpy(np.array(data2))
    tensor_label = torch.from_numpy(np.array(label2))
    torch_dataset = Data.TensorDataset(tensor_data, tensor_label)

    return torch_dataset


# 小波去噪预处理
def denoise(data):
    # 小波变换
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

    # 阈值去噪
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    cD1.fill(0)
    cD2.fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)

    # 小波反变换,获取去噪后的信号
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata
