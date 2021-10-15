import visdom
import torch
import numpy as np
# 新建一个连接客户端
# 指定env = 'test1'，默认是'main',注意在浏览器界面做环境的切换


class Visualizer:
    def __init__(self, opt):
        self.opt = opt
        self.name = opt.RECORD_NAME
        self.vis = visdom.Visdom(env='main')

    def display_train_result(self, loss, tacc, vacc, epoch):  # h
        self.vis.line(Y=[[loss, tacc, vacc]], X=[epoch], win=self.name,
                      opts=dict(title=self.name, legend=['train_loss', 'train_acc', 'val_acc']),
                      update=None if epoch == 0 else 'append')


class Nothing:
    pass
