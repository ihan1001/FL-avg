'''
-*- coding: utf-8 -*-
@Time    : 2023/11/23 17:35
@Author  : ihan
@File    : FederateAI
@Software: PyCharm
'''

import torch
from torchvision import datasets, transforms

#获取数据集
#调用语句 train_datasets, eval_datasets = datasets.get_dataset("./data/", conf["type"])
def get_dataset(dir, name):
    if name == 'mnist':
        train_dataset = datasets.MNIST(dir, train=True, download=True, transform=transforms.ToTensor())
        print(train_dataset)
        print("--------------------")
        '''
        dir 数据路径  
        train=True 代表训练数据集   Flase代表测试数据集
        download  是否从互联网下载数据集并存放在dir中
        transform=transforms.ToTensor() 表示将图像转换为张量形式
        '''
        eval_dataset = datasets.MNIST(dir, train=False, transform=transforms.ToTensor())
        print(eval_dataset)
        print("--------------------")

    elif name == 'cifar':
        '''
        transforms.Compose 是将多个transform组合起来使用（由transform构成的列表）
        '''
        transform_train = transforms.Compose([
            #
            transforms.RandomCrop(32, padding=4),#transforms.RandomCrop： 切割中心点的位置随机选取
            #随机裁剪原始图像，裁剪的大小为32x32像素，padding参数表示在图像的四周填充4个像素，以避免裁剪后图像边缘信息丢失。
            transforms.RandomHorizontalFlip(),#以0.5的概率对图像进行随机水平翻转，增加数据集的多样性。
            transforms.ToTensor(),#将图像转换为张量形式，便于后续处理。
            #transforms.Normalize： 给定均值：(R,G,B) 方差：（R，G，B），将会把Tensor正则化
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            #对图像进行标准化操作，减去均值并除以标准差。这里使用的均值和标准差是CIFAR-10数据集的全局均值和标准差，可以使模型更容易收敛。
        ])
        print(transform_train)
        print("--------------------")

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        print(transform_test)
        print("--------------------")

        train_dataset = datasets.CIFAR10(dir, train=True, download=True,
                                         transform=transform_train)
        eval_dataset = datasets.CIFAR10(dir, train=False, transform=transform_test)

    return train_dataset, eval_dataset
# if __name__ == '__main__':
#     train_datasets, eval_datasets = get_dataset("./data/", "cifar")
#     # print(train_datasets)
#     # print(eval_datasets)