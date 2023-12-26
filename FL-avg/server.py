'''
-*- coding: utf-8 -*-
@Time    : 2023/11/23 17:37
@Author  : ihan
@File    : FederateAI
@Software: PyCharm
'''

import models, torch


class Server(object):
    '''
    定义构造函数。在构造函数中，服务端的工作包括：
    第一，将配置信息拷贝到服务端中；
    第二，按照配置中的模型信息获取模型，这里我们使用torchvision的models模块内置的ResNet - 18模型。
    '''
    def __init__(self, conf, eval_dataset):
        # 导入配置文件
        self.conf = conf
        # 根据配置获取模型文件
        self.global_model = models.get_model(self.conf["model_name"])
        # 生成一个测试集合加载器                                                           # 设置单个批次大小     # 打乱数据集
        self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.conf["batch_size"], shuffle=True)

    '''
    定义模型聚合函数。前面我们提到服务端的主要功能是进行模型的聚合,
    因此定义构造函数后，我们需要在类中定义模型聚合函数，通过接收客户端上传的模型，使用聚合函数更新全局模型。
    聚合方案有很多种，本节我们采用经典的FedAvg算法。
    '''
    # 全局聚合模型
    # weight_accumulator 存储了每一个客户端的上传参数变化值/差值
    def model_aggregate(self, weight_accumulator):
        # 遍历服务器的全局模型
        #利用下面的循环获取到未更新前的全局模型，如果是第一次，拿到的就是初始化中的模型
        for name, data in self.global_model.state_dict().items():#是引用性传递，所以改变data就是改变global_model
            # print("name:%s\n"%name)
            # print("更新前全局模型的data:\n")
            # print(data)

            # 更新每一层乘上lambda
            update_per_layer = weight_accumulator[name] * self.conf["lambda"]
            # 因为update_per_layer的type是floatTensor，所以将起转换为模型的LongTensor（有一定的精度损失）
            if data.type() != update_per_layer.type():
                data.add_(update_per_layer.to(torch.int64))
                # print("name:%s\n" % name)
                # print("更新后全局模型的data:\n")
                # print(data)
            else:
                data.add_(update_per_layer)
                # print("name:%s\n" % name)
                # print("更新后全局模型的data:\n")
                # print(data)

    '''
    定义模型评估函数。对当前的全局模型，利用评估数据评估当前的全局模型性能。
    通常情况下，服务端的评估函数主要对当前聚合后的全局模型进行分析，用于判断当前的模型训练是需要进行下一轮迭代、还是提前终止，或者模型是否出现发散退化的现象。
    根据不同的结果，服务端可以采取不同的措施策略。
    '''
    def model_eval(self):
        self.global_model.eval()#将全局模型设置为评估模式

        total_loss = 0.0
        correct = 0
        dataset_size = 0
        for batch_id, batch in enumerate(self.eval_loader):
            data, target = batch
            dataset_size += data.size()[0]#dataset_size表示数据集的大小

            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            output = self.global_model(data)#前向传播
            '''
            在这行代码中，torch.nn.functional.cross_entropy 是一个用于计算交叉熵损失的函数。它接受两个参数：output 和 target。
            output 是模型的输出结果，可以是一个张量或一个批次的张量。target 是目标值，通常是一个包含正确类别标签的张量。
            cross_entropy 函数会计算模型输出 output 与目标值 target 之间的交叉熵损失。交叉熵损失是一种常用的损失函数，用于衡量模型输出与目标值之间的差异。
            在这行代码中，通过调用 cross_entropy 函数计算出的损失值被累加到 total_loss 变量中。item() 方法用于获取张量中的标量值。
            最终，total_loss 表示整个批次中的损失总和，即将每个样本的损失进行累加。这个值可以用来衡量模型在当前批次中的性能，并用于后续的优化过程。
            reduction='sum' 是 torch.nn.functional.cross_entropy 函数的一个参数选项。
            这个参数用于指定损失函数的归约方式，即如何将每个样本的损失进行汇总。
            当设置 reduction='sum' 时，cross_entropy 函数会将所有样本的损失值相加，得到一个总的损失值。这个总的损失值表示整个批次中的损失总和。
            相反，如果将 reduction 设置为其他选项，比如 reduction='mean'，则 cross_entropy 函数会计算每个样本的平均损失，即将总的损失值除以样本数量。
            选择适当的归约方式取决于具体的任务和需求。如果希望获得批次中所有样本的总损失值，则可以使用 reduction='sum'；如果希望获得平均损失值，则可以使用 reduction='mean'。

            '''
            total_loss += torch.nn.functional.cross_entropy(output, target,reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            '''
            在这行代码中，output 是模型的输出结果，可以是一个张量或一个批次的张量。output.data 将 output 转换为一个张量，然后使用 max 方法找到每个样本中概率最大的类别。
            具体来说，max(1) 表示在 output.data 的第一个维度（即类别维度）上进行取最大值操作，得到一个元组 (max_values, max_indices)。
            其中，max_values 是每个样本中最大的概率值，max_indices 是对应的类别索引。
            [1] 表示获取元组中的第二个元素，即类别索引。因此，pred 变量包含了模型预测的类别标签，它是一个张量或一个批次的张量，与输入数据的形状相同。
            总之，这行代码的作用是找到模型输出中概率最大的类别，并将其作为模型的预测结果。
            
            这行代码计算了模型对输入数据的预测结果中正确的个数。让我来解释一下：
            pred.eq(target.data.view_as(pred))：这部分代码比较了模型的预测结果 pred 与目标标签 target.data 是否相等，生成一个布尔类型的张量，其中相等的位置为 True，不相等的位置为 False。
            .cpu().sum().item()：首先，调用 .cpu() 方法将上一步得到的布尔类型张量移动到 CPU 上进行处理，然后调用 .sum() 方法计算所有 True 的总数，最后使用 .item() 方法将张量中的单个元素提取为 Python 数值，表示模型在这批数据中预测正确的数量。
            最终，这行代码将正确预测的数量累加到名为 correct 的变量中，用于后续计算模型的准确率或其他评估指标。
            '''
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size

        return acc, total_l