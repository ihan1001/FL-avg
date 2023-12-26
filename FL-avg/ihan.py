'''
-*- coding: utf-8 -*-
@Time    : 2023/12/25 22:53
@Author  : ihan
@File    : FederateAI
@Software: PyCharm
'''
# import random
# train_loss = train_loss - random.uniform(0.01, 0.017)
# train_acc = train_acc + random.uniform(0.025, 0.035)



from datetime import datetime
import time		# python里的日期模块

def savePara(e,acc,loss):

    date = time.strftime('%Y-%m-%d %H:%M:%S').split()		# 按空格分开，这里宫格是我自己添加的
    print(date)     # ['2021-07-22', '16:11:00']
    hour_minute = date[1].split(':')
    print(hour_minute)      # 时，分，秒

    filepath = "/home/sysadmin/"+ date[0]+'.txt'
    output = "%s：Epoch %d  training accuracy :  %g, train Loss : %f " % (datetime.now(), e, acc, loss)
    with open(filepath,"a+") as f:
        f.write(output+'\n')
        f.close
def saveIn(model_name,no_models,date_type,global_epochs,local_epochs,current_clients):

    date = time.strftime('%Y-%m-%d %H:%M:%S').split()		# 按空格分开，这里宫格是我自己添加的
    print(date)     # ['2021-07-22', '16:11:00']
    hour_minute = date[1].split(':')
    print(hour_minute)      # 时，分，秒

    filepath = "/home/sysadmin/"+ date[0]+'.txt'
    output="%s：model_name %s,no_models %d,date_type %s,global_epochs %d,local_epochs %d,current_clients %d start"\
        % (datetime.now(), model_name,no_models,date_type,global_epochs,local_epochs,current_clients)
    with open(filepath, "a+") as f:
        f.write(output + '\n')
        f.close

