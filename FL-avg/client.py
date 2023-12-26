'''
-*- coding: utf-8 -*-
@Time    : 2023/11/23 17:33
@Author  : ihan
@File    : FederateAI
@Software: PyCharm
'''

'''
横向联邦学习的客户端主要功能是接收服务端的下发指令和全局模型，利用本地数据进行局部模型训练。

'''

import models, torch, copy


class Client(object):
    '''
    定义构造函数。在客户端构造函数中，客户端的主要工作包括：
    首先，将配置信息拷贝到客户端中；
    然后，按照配置中的模型信息获取模型，通常由服务端将模型参数传递给客户端，客户端将该全局模型覆盖掉本地模型；
    最后，配置本地训练数据，
    在本案例中，我们通过torchvision 的datasets 模块获取cifar10
    数据集后按客户端ID切分，不同的客户端拥有不同的子数据集，相互之间没有交集。
    '''

    def __init__(self, conf, model, train_dataset, id=-1):
        #获取参数
        self.conf = conf
        #获取模型名字
        self.local_model = models.get_model(self.conf["model_name"])
        #获取用户id
        self.client_id = id
        #获取整个训练数据集的摘要信息
        self.train_dataset = train_dataset
        '''
        这段代码创建了一个名为all_range的列表，其中包含了数据集self.train_dataset的长度范围内的所有整数值。
        具体来说，len(self.train_dataset)返回了数据集中样本的数量，然后使用range()函数生成一个从0到样本数量减1的整数范围。
        将这个整数范围转换为列表，即可得到包含了数据集中所有样本索引的列表。这个列表可以用于在训练过程中遍历数据集中的所有样本，或者进行随机采样等操作。
        这个操作的目的可能是为了对训练数据集的样本进行遍历或采样，以便在训练模型时使用。
        '''
        all_range = list(range(len(self.train_dataset)))
        #总数据集的长度除以no_models用户数，为每个用户分配数据
        data_len = int(len(self.train_dataset) / self.conf['no_models'])
        #如果id被假设为5，那么代码train_indices = all_range[id * data_len: (id + 1) * data_len]将会得到从索引位置500到索引位置599的一部分数据集样本索引。
        train_indices = all_range[id * data_len: (id + 1) * data_len]
        print(train_indices)
        '''
        这段代码使用train_indices作为子集的索引，创建了一个名为self.train_loader的数据加载器。
        具体来说，torch.utils.data.DataLoader是PyTorch提供的一个用于加载数据的工具类。
        它可以将数据集对象（这里是self.train_dataset）封装成一个可迭代的数据加载器，方便训练过程中以批次的方式加载数据。
        在这个代码中，使用了SubsetRandomSampler作为sampler参数，它是torch.utils.data.sampler模块中的一个采样器类。SubsetRandomSampler用于从给定的索引列表中随机选取样本，构建一个新的采样器。这里传入了train_indices作为索引列表，表示只从数据集的指定子集中进行采样。
        另外，batch_size参数指定了每个批次的样本数量，控制了每次迭代时加载的样本数量。这里使用了conf["batch_size"]作为批次大小。
        通过以上操作，self.train_loader将成为一个可以迭代访问数据集中指定子集的数据加载器，每次迭代会返回一个由batch_size个样本组成的批次数据。这样可以方便地将数据输入模型进行训练
        train_indices=0-99 batch_size=20 train_loader就是可能包含5条数据，分别是5个批次，每个批次包含20个样本。
        
        本地训练的时候每一轮会用5个批次，分别训练
        local
        '''
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=conf["batch_size"],
                                                        sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                            train_indices))
        print(self.train_loader)


    '''
    定义模型本地训练函数。
    本例是一个图像分类的例子，
    因此，我们使用交叉熵作为本地模型的损失函数，利用梯度下降来求解并更新参数值
    
    #本地训练后，返回和全局模型的差值
    '''
    def local_train(self, model):
        #将全局模型的参数复制到本地模型中
        for name, param in model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())

        # print(id(model))
        #定义了一个 SGD（随机梯度下降）优化器，并将其应用于本地模型的参数
        #torch.optim.SGD() 函数创建了一个 torch.optim.SGD 的实例，该实例将用于对本地模型的参数进行优化。
        #该函数需要两个参数：第一个参数是要优化的参数列表，这里是 self.local_model.parameters()；第二个参数是学习率，这里使用 self.conf['lr'] 从配置文件中获取。
        #另外，还可以使用可选参数来配置 SGD 优化器的其他超参数，例如动量（momentum）等。
        #在这里，我们使用 self.conf['momentum'] 从配置文件中获取动量值。
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'],
                                    momentum=self.conf['momentum'])
        # print(id(self.local_model))
        #这段代码是将本地模型设为训练模式。
        # 在PyTorch中，模型有两种模式：训练模式和评估模式。
        #在训练模式下，模型会执行反向传播算法，并根据损失函数和优化器更新参数；而在评估模式下，模型只会前向传播，不会执行反向传播或者参数更新操作，通常用于计算模型在测试集上的准确率等指标。
        # 通过self.local_model.train()将本地模型设置为训练模式，可以启用反向传播算法和参数更新操作，使得模型能够进行训练。
        # 通常在训练过程中需要将模型设为训练模式，而在测试过程中则需要将模型设为评估模式，以避免在测试时不必要地更新参数。
        self.local_model.train()

        # local_epochs=3  外循环3次
        for e in range(self.conf["local_epochs"]):
            #本地训练的时候每一轮会用5个批次，分别训练5次，算在一个local_epochs
            #每次使用20个样本

            #用于遍历训练数据集中的每个批次，并获取数据和对应的目标。
            #self.train_loader 是一个数据加载器，用于从训练数据集中按批次加载数据。
            #enumerate(self.train_loader) 返回一个可迭代对象，通过遍历该对象可以依次获取数据集中的每个批次以及对应的批次编号 batch_id。
            #对于每个批次，data, target = batch 语句将批次数据和对应的目标值分别赋值给 data 和 target。
            #这意味着 data 变量包含了当前批次的输入数据，而 target 变量包含了当前批次的目标标签或输出值。
            #通过这个循环，可以逐个处理训练数据集中的每个批次数据，并对模型进行训练。通常在深度学习训练过程中，会使用类似这样的循环来遍历数据集，并将数据送入模型进行训练。
            for batch_id, batch in enumerate(self.train_loader):
                data, target = batch

                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()
                '''
                optimizer.zero_grad()：这一行代码的作用是清空之前参数的梯度。在每个批次开始训练之前，需要先将之前累积的梯度清零，以避免梯度累积影响当前批次的参数更新。
                output = self.local_model(data)：这一行代码将输入数据 data 传递给本地模型，并获取模型的输出结果 output。
                loss = torch.nn.functional.cross_entropy(output, target)：这一行代码计算模型输出与目标标签之间的交叉熵损失。
                torch.nn.functional.cross_entropy 函数会将模型的输出 output 和对应的真实标签 target 作为输入，计算它们之间的交叉熵损失值，并将结果保存在 loss 变量中。
                loss.backward()：这一行代码执行反向传播算法，计算损失函数对模型参数的梯度。在这一步，PyTorch 会自动计算损失函数关于模型参数的梯度，并将梯度信息保存在各个参数的 .grad 属性中，供优化器后续使用。
                '''
                optimizer.zero_grad()
                '''
                local_model(data) 是将输入数据 data 传递给本地模型 local_model 进行前向传播。
                在深度学习中，模型的前向传播是指将输入数据通过模型的各个层和激活函数进行计算，最终得到输出结果。通过调用 local_model(data)，可以将输入数据 data 传递给本地模型，并获取模型的输出结果。
                这个操作会根据模型的结构和参数对输入数据进行一系列计算，例如矩阵乘法、卷积、激活函数等。最终，模型会生成一个输出结果，表示对输入数据的预测或特征提取。
                反向传播：通常在训练过程中，会将输入数据通过模型的前向传播得到预测结果，然后与真实标签进行比较，计算损失函数并进行反向传播以更新模型参数。
                '''
                output = self.local_model(data)
                loss = torch.nn.functional.cross_entropy(output, target)
                loss.backward()#这一行代码执行反向传播算法，计算损失函数对模型参数的梯度
                #在计算完梯度之后，可以通过调用 optimizer.step() 来实现模型参数的更新
                '''
                优化器的作用是根据计算得到的梯度信息和预定义的优化算法，调整模型参数以最小化损失函数。通常使用随机梯度下降（SGD）等算法来更新模型参数。
                在执行 optimizer.step() 之前，需要先计算损失函数关于模型参数的梯度信息（通过 .backward() 方法计算），并将梯度信息保存在各个参数的 .grad 属性中。optimizer.step() 会根据这些梯度信息和预定义的优化算法，更新模型参数的值，并清空之前参数的梯度。
                通过这个过程，模型的参数被不断地调整以最小化损失函数，使模型对训练数据的拟合程度不断提高。
                '''
                optimizer.step()
            print("Epoch %d done." % e)
        #创建空的字典
        diff = dict()
        #model是训练前的模型，也就是本轮次的全局模型
        '''
        这段代码是用于比较两个模型参数之间的差异，并返回一个字典 diff，其中包含了两个模型参数之间的差值。
        self.local_model.state_dict() 返回本地模型 self.local_model 的状态字典。状态字典是一个包含了模型中所有可学习参数的字典，以参数名为键，参数值为值。
        model.state_dict() 返回另一个模型 model 的状态字典。
        for name, data in self.local_model.state_dict().items(): 是一个循环语句，遍历本地模型的状态字典。在每次循环中，name 表示参数名，data 表示参数值。
        diff[name] = (data - model.state_dict()[name]) 这一行代码计算了本地模型参数和另一个模型参数之间的差值，并将结果保存在 diff 字典中，以参数名为键，差值为值。
        return diff 返回包含了两个模型参数之间差值的字典 diff。
        通过这段代码，可以获得两个模型参数之间的差异信息，可以进一步分析和比较模型之间的相似性或差异性。

        '''
        for name, data in self.local_model.state_dict().items():
            #
            diff[name] = (data - model.state_dict()[name])
            #print("name:%s,diff:%s \n"%(name,diff[name]))
        return diff
