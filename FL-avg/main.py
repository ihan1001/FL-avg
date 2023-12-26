'''
-*- coding: utf-8 -*-
@Time    : 2023/11/23 17:33
@Author  : ihan
@File    : FederateAI
@Software: PyCharm
'''
import argparse, json
import datetime
import os
import logging
import torch, random
import ihan

from server import *
from client import *
import models, datasets

if __name__ == '__main__':
    '''
    这是一个使用 Python 的 argparse 库创建命令行参数解析器的代码片段。argparse.ArgumentParser 是 argparse 模块中的一个类，用于创建命令行参数解析器对象。
    在这个例子中，创建了一个名为 parser 的参数解析器对象，并指定了一个描述字符串 'Federated Learning'。该参数解析器将用于解析命令行参数的值。
    通过 argparse 库，你可以定义和添加各种命令行参数，包括位置参数、可选参数、布尔标志等。然后，使用 parser.parse_args() 方法来解析命令行参数，并将其转换为相应的数据类型。
    '''
    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('-c', '--conf', dest='conf')
    #conf中有-c或者--conf 参数后的值
    #python main.py -c ./utils/conf.json
    args = parser.parse_args()
    #使用 parse_args() 方法来解析命令行参数，并将解析后的结果保存在 args 变量中
    #args.conf=./utils/conf.json
    #读取配置文件信息。
    with open(args.conf, 'r') as f:
        conf = json.load(f)

    ihan.saveIn(conf["model_name"],conf["no_models"],conf["type"],conf["global_epochs"],conf["local_epochs"],conf["k"])

    #获取数据集
    train_datasets, eval_datasets = datasets.get_dataset("./data/", conf["type"])
    #分别定义一个服务端对象和多个客户端对象，用来模拟横向联邦训练场景。
    server = Server(conf, eval_datasets)
    clients = []
    #根据no_models新建no_models个客户，train_datasets整个数据集的信息，在client中进行划分，其实这里应该划分
    for c in range(conf["no_models"]):
        clients.append(Client(conf, server.global_model, train_datasets, c))
    #clients列表中有no_models个客户端
    print("\n\n")

    #每一轮的迭代，服务端会从当前的客户端集合中随机挑选一部分参与本轮迭代训练，
    #被选中的客户端调用本地训练接口local_train进行本地训练，
    #最后服务端调用模型聚合函数model_aggregate来更新全局模型
    for e in range(conf["global_epochs"]):
        #使用了 Python 内置模块 random 中的 sample() 函数，从列表 clients 中随机选取 conf["k"] 个元素，并将其作为一个新列表赋值给变量 candidates
        candidates = random.sample(clients, conf["k"])

        weight_accumulator = {}
        '''
        这段代码是在遍历server.global_model的状态字典，并且为每个参数创建一个与其形状相同的零张量，并将其存储在weight_accumulator字典中。
        具体来说，对于模型中的每个参数，都会创建一个与其形状相同的全零张量，并将其存储在weight_accumulator字典中，以参数的名称作为键。
        这个操作通常用于在训练神经网络时用于累积梯度的操作，比如在一些分布式或异步更新的训练算法中，需要累积每个参数的梯度信息，以便进行后续的参数更新。
        '''
        for name, params in server.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params)
        #使用本轮次的全局模型进行训练
        for c in candidates:
            #这个循环执行结束后，所有候选人的模型差值被累加。存放在weight_accumulator里面
            #一次本地训练，返回和全局模型的差值
            diff = c.local_train(server.global_model)

            #将差异信息累积到全局模型的参数上
            #server.global_model.state_dict() 返回全局模型 server.global_model 的状态字典，包含了全局模型的所有可学习参数。
            #for name, params in server.global_model.state_dict().items(): 是一个循环语句，遍历全局模型的状态字典。在每次循环中，name 表示参数名，params 表示参数值。
            #weight_accumulator 是一个字典，用于保存累积的差异信息。
            #weight_accumulator[name].add_(diff[name])
            #这一行代码将之前计算得到的 diff 字典中对应参数名 name 的差异信息累积到全局模型的参数上。
            #使用 .add_() 方法可以实现原地累积操作，将 diff[name] 中存储的本地模型参数与全局模型参数的差异信息累积到 weight_accumulator[name] 中。
            #这个操作会直接更新 weight_accumulator[name] 的数值，使其加上 diff[name]。
            for name, params in server.global_model.state_dict().items():
                weight_accumulator[name].add_(diff[name])

        #一次全局模型结束，所有候选人的模型差值被累加。存放在weight_accumulator里面，接下来聚合全局模型，更新全局模型
        server.model_aggregate(weight_accumulator)
        #看server.py的model_aggregate(weight_accumulator)
        #应该在上一步更新的全局模型，现在的server中的全局模型时更新过的
        acc, loss = server.model_eval()

        print("Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss))
        ihan.savePara(e,acc,loss)








