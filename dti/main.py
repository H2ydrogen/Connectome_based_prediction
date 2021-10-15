import logging
import time

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch
import torch.nn.functional as functional
import numpy as np
from DTI import models, dataset, cli, utils, analyse, visualizer
import operator
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main():
    utils.setup_seed(123)
    start_time = time.time()
    vis = visualizer.Visualizer(args)

    # 数据集加载
    train_set = dataset.CreateDataset(args, dataset.get_hcp_s1200(args), usage='train')
    val_set = dataset.CreateDataset(args, dataset.get_hcp_s1200(args), usage='val')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, drop_last=False, shuffle=True, pin_memory=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, drop_last=True, shuffle=True, pin_memory=True,
                            num_workers=args.num_workers)

    # 网络加载
    nets = []
    for i in range(len(args.INPUT_FEATURES)):
        if args.MODEL == '1D-CNN':
            nets.append(models.HARmodel(args).to(device))
        elif args.MODEL == '2D-CNN':
            nets.append(models.CNN_2D(args).to(device))
        elif args.MODEL == 'Lenet':
            nets.append(models.Lenet(args).to(device))
        elif args.MODEL == '1.5D':
            nets.append(models.new_CNN(args).to(device))
        else:
            raise

    # if os.path.exists(args.LOAD_PATH):
    #     model.load_state_dict(torch.load(args.LOAD_PATH))
    optimizer = []
    for model in nets:
        optimizer.append(torch.optim.SGD(model.parameters(), lr=args.LR, weight_decay=args.L2))

    # 训练
    val_loss = []
    train_loss = []
    val_metric = []
    train_metric = []

    # 打印训练信息
    LOG.info("Args:{}".format(args))

    for epoch in range(args.epochs):
        train_results = train(train_loader, nets, optimizer, epoch)
        val_results = validation(nets, val_loader)

        # 训练结果记录
        train_loss.append(train_results['loss'])
        val_loss.append(val_results['loss'])
        if args.target == 'sex':
            train_metric.append(train_results['acc'])
            val_metric.append(val_results['acc'])
        elif args.target == 'age':
            train_metric.append(train_results['MAE'])
            val_metric.append(val_results['MAE'])
        else:
            train_metric, val_metric = None, None

        # 打印最好成绩
        if args.target == 'sex':
            max_train_metric_index, max_train_metric = max(enumerate(train_metric), key=operator.itemgetter(1))
            max_val_metric_index, max_val_metric = max(enumerate(val_metric), key=operator.itemgetter(1))
            min_train_loss_index, min_train_loss = min(enumerate(train_loss), key=operator.itemgetter(1))
        elif args.target == 'age':
            max_train_metric_index, max_train_metric = min(enumerate(train_metric), key=operator.itemgetter(1))
            max_val_metric_index, max_val_metric = min(enumerate(val_metric), key=operator.itemgetter(1))
            min_train_loss_index, min_train_loss = min(enumerate(train_loss), key=operator.itemgetter(1))
        print('best train_loss；{}({}epoch)---best t_metric:{}({}epoch)---best val_metric；{}({}epoch),'
              .format(min_train_loss, min_train_loss_index+1, max_train_metric, max_train_metric_index+1, max_val_metric, max_val_metric_index+1,))

        # 训练结果可视化
        vis.display_train_result(train_loss[-1], train_metric[-1], val_metric[-1], epoch)

    # 训练结果存档
    # torch.save(model.state_dict(), '.\\LOG\\{}.pkl'.format(args.RECORD_NAME))
    f = open('.\\LOG\\{}.txt'.format(args.RECORD_NAME), 'a+')
    f.writelines('args'+str(args)+'\n')
    f.writelines('train_loss'+str(train_loss)+'\n')
    f.writelines('val_loss' + str(val_loss)+'\n')
    f.writelines('train_metric' + str(train_metric)+'\n')
    f.writelines('val_metric' + str(val_metric)+'\n')
    f.close()
    LOG.info("--- main.py finish in %s seconds ---" % (time.time() - start_time))


def train(dataloader, nets, optimizer, epoch):
    for model in nets:
        model.train()
    train_loss = 0
    pred_list = []
    target_list = []
    start_time = time.time()
    for batch_index, batch_samples in enumerate(dataloader):
        # 1.load data to CUDA
        x, y = batch_samples['x'].to(device), batch_samples['y'].to(device)  # x.size = (bs,C, len); y.size = (bs)

        # 2.forward
        total_loss = 0
        output = np.zeros(1)
        for index, model in enumerate(nets):
            out = model(x[:, index:index+1, :])  # output.size = (bs, num_classes)
            if args.LOSS == 'CE':
                criteria = nn.CrossEntropyLoss()
                loss = criteria(out, y.long())
            elif args.LOSS == 'MSE':
                criteria = nn.MSELoss()
                loss = criteria(out.reshape(-1), y)
            else:
                loss = None

            # 3.backward
            optimizer[index].zero_grad()  # 把所有Variable的grad成员数值变为0
            loss.backward()  # 反向传播grad
            optimizer[index].step()  # 每个Variable的grad都被计算出来后，更新每个Variable的数值（优化更新）
            total_loss += loss
            if output.any():
                out = out.unsqueeze(2)
                output = np.concatenate((output, out.detach().cpu().numpy()), axis=2)  # output(bs, num_models)
            else:
                out = out.unsqueeze(2)
                output = out.detach().cpu().numpy()

        # 6.result
        if args.target == 'sex':  # voting来获得最终结果
            output = np.argmax(output, axis=1)
            output = np.mean(output, axis=1)
            output = (output > 0.5).astype(np.int)
            pred = output
        elif args.target == 'age':  # average来求得最终结果
            output = np.mean(output, axis=2)
            output = np.squeeze(output, 1)
            pred = output
        loss = total_loss.detach().cpu().numpy()  # 只有没有grad的tensor可以.numpy(),用.detech()解除grad
        y = y.cpu().numpy()
        train_loss += loss
        pred_list = np.append(pred_list, pred)
        target_list = np.append(target_list, y)

        # 提示训练进度
        if batch_index % args.display_batch == 0:
            LOG.info("--- training progress rate {}/{} ---".format(batch_index, len(dataloader)))

    if args.target == 'sex':
        train_result = analyse.analyse_classification(train_loss, target_list, pred_list)
    elif args.target == 'age':
        train_result = analyse.analyse_regression(train_loss, target_list, pred_list)
    else:
        train_result = None
    LOG.info("--- training epoch {} finish in {} seconds ---".format(epoch, round(time.time() - start_time, 4)))

    return train_result


def validation(nets, val_loader):
    for model in nets:
        model.eval()
    test_loss = 0
    start_time = time.time()
    with torch.no_grad():
        pred_list = []
        target_list = []
        for batch_index, batch_samples in enumerate(val_loader):
            #  1.load data to CUDA
            x, y = batch_samples['x'].to('cuda'), batch_samples['y'].to('cuda')

            # 2.forward
            output = np.zeros(1)
            for index, model in enumerate(nets):
                out = model(x[:, index:index+1, :])
                if args.LOSS == 'CE':
                    criteria = nn.CrossEntropyLoss()
                    loss = criteria(out, y.long())
                elif args.LOSS == 'MSE':
                    criteria = nn.MSELoss()
                    loss = criteria(out.reshape(-1), y)
                else:
                    pass

                if output.any():
                    out = out.unsqueeze(2)
                    output = np.concatenate((output, out.detach().cpu().numpy()), axis=2)  # output(bs, num_models)
                else:
                    out = out.unsqueeze(2)
                    output = out.detach().cpu().numpy()

            #  3.result
            if args.target == 'sex':  # voting来获得最终结果
                output = np.argmax(output, axis=1)
                output = np.mean(output, axis=1)
                output = (output > 0.5).astype(np.int)
                pred = output
            elif args.target == 'age':  # average来求得最终结果
                output = np.mean(output, axis=2)
                output = np.squeeze(output, 1)
                pred = output
            test_loss += loss.cpu().numpy()
            y = y.cpu().numpy()
            pred_list = np.append(pred_list, pred)
            target_list = np.append(target_list, y)

    if args.target == 'sex':
        val_result = analyse.analyse_classification(test_loss, target_list, pred_list)
    elif args.target == 'age':
        val_result = analyse.analyse_regression(test_loss, target_list, pred_list)
    else:
        val_result = None
    LOG.info("--- validation epoch finish in {} seconds ---".format(round(time.time() - start_time, 4)))

    return val_result


if __name__ == '__main__':
    LOG = logging.getLogger('main')
    logging.basicConfig(level=logging.INFO)
    args = cli.create_parser().parse_args()
    main()
