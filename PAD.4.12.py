from __future__ import print_function
import argparse
import os
import pandas as pd
from matplotlib import pyplot as plt
import models
import torch
import torch.nn as nn
import torch.optim as optim
from utils import progress_bar, set_logging_defaults
from datasets import load_dataset, setup_seed
import logging


class KDLoss(nn.Module):
    def __init__(self, temp_factor):
        super(KDLoss, self).__init__()
        self.temp_factor = temp_factor
        self.kl_div = nn.KLDivLoss(reduction="sum")

    def forward(self, input, target):
        log_p = torch.log_softmax(input/self.temp_factor, dim=1)
        q = torch.softmax(target/self.temp_factor, dim=1)
        loss = self.kl_div(log_p, q)*(self.temp_factor**2)/input.size(0)
        return loss


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    train_ads_loss = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        batch_size = inputs.size(0)

        if not args.ads:
            inputs = inputs.reshape(args.batch_size, 1, 1000)
            F_odd = inputs[:, :, 1::2]  # 提取奇数位置
            F_even = inputs[:, :, ::2]  # 提取偶数位置
            min_length = min(F_odd.shape[2], F_even.shape[2])
            inputs_x = F_odd[:, :, :min_length]
            inputs_y = F_even[:, :, :min_length]

            outputs_x = net(inputs_x)
            outputs_y = net(inputs_y)
            loss_x = torch.mean(criterion(outputs_x, targets))
            loss_y = torch.mean(criterion(outputs_y, targets))
            combined_outputs = outputs_x + outputs_y
            _, predicted = torch.max(combined_outputs, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).sum().float().cpu()
            loss = (args.lamda * loss_x + args.lamda * loss_y)
            train_loss += loss.item()

            ads_loss_self = kdloss(outputs_x, outputs_y.detach())

            loss += (1 - 2 * args.lamda) * ads_loss_self

        else:
            inputs = inputs.reshape(args.batch_size * 2, 1, 1000)
            targets_ = targets[:batch_size // 2]
            inputs_half = inputs[:batch_size // 2]
            F_odd = inputs_half[:, :, 1::2]  # 提取奇数位置
            F_even = inputs_half[:, :, ::2]  # 提取偶数位置
            min_length = min(F_odd.shape[2], F_even.shape[2])
            inputs_x = F_odd[:, :, :min_length]
            inputs_y = F_even[:, :, :min_length]
            # Preservative Spectra Decoupling
            outputs_x = net(inputs_x)
            outputs_y = net(inputs_y)
            combined_outputs = outputs_x + outputs_y
            # Adaptive guide
            loss_x = torch.mean(criterion(outputs_x, targets_))
            loss_y = torch.mean(criterion(outputs_y, targets_))
            loss = (args.lamda * loss_x + args.lamda * loss_y)

            train_loss += loss.item()

            with torch.no_grad():
                inputs_half_end = inputs[batch_size // 2:]
                F_odd_end = inputs_half_end[:, :, 1::2]  # 提取奇数位置
                F_even_end = inputs_half_end[:, :, ::2]  # 提取偶数位置
                min_length = min(F_odd_end.shape[2], F_even_end.shape[2])
                inputs_x_end = F_odd_end[:, :, :min_length]
                inputs_y_end = F_even_end[:, :, :min_length]
                outputs_x_end = net(inputs_x_end)
                outputs_y_end = net(inputs_y_end)
            # Adaptive follow
            ads_loss_xy = kdloss(outputs_x, outputs_y_end.detach())
            ads_loss_yx = kdloss(outputs_y, outputs_x_end.detach())
            total_ads_loss = (ads_loss_xy + ads_loss_yx) * (1 - 2 * args.lamda)

            loss += total_ads_loss
            train_ads_loss += ((ads_loss_xy + ads_loss_yx)/2).item()

            _, predicted = torch.max(combined_outputs, 1)
            total += targets_.size(0)
            correct += predicted.eq(targets_.data).sum().float().cpu()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar(batch_idx, len(trainloader),
                     'Loss: %.3f | Acc: %.3f%% (%d/%d) | ads: %.3f '
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, train_ads_loss/(batch_idx+1)))
    train_loss_list.append(train_loss / (batch_idx + 1))
    train_acc_list.append((100. * correct / total).item())
    logger = logging.getLogger('train')
    logger.info('[Epoch {}] [Loss {:.3f}] [ads {:.3f}] [Acc {:.3f}]'.format(
        epoch,
        train_loss/(batch_idx+1),
        train_ads_loss/(batch_idx+1),
        100.*correct/total))

    return train_loss/batch_idx, 100.*correct/total, train_ads_loss/batch_idx


def val(epoch):
    global best_val
    net.eval()
    val_loss = 0.0
    correct = 0.0
    total = 0.0

    # Define a data loader for evaluating
    loader = valloader

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs = inputs.reshape(args.batch_size, 1, 1000)
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            F_odd = inputs[:, :, 1::2]  # 提取奇数位置
            F_even = inputs[:, :, ::2]  # 提取偶数位置
            min_length = min(F_odd.shape[2], F_even.shape[2])
            inputs_x = F_odd[:, :, :min_length]
            inputs_y = F_even[:, :, :min_length]

            outputs_x = net(inputs_x)
            outputs_y = net(inputs_y)
            loss_x = criterion(outputs_x, targets)
            loss_y = criterion(outputs_y, targets)
            loss = (loss_x + loss_y) / 2  # 例如取平均

            val_loss += loss.item()
            combined_outputs = outputs_x + outputs_y
            _, predicted = torch.max(combined_outputs, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().float()

            progress_bar(batch_idx, len(loader),
                         'Loss: %.3f | Acc: %.3f%% (%d/%d) '
                         % (val_loss/(batch_idx+1), 100.*correct/total, correct, total))

    val_loss_list.append(val_loss / (batch_idx + 1))
    val_acc_list.append((100. * correct / total).item())
    acc = 100.*correct/total
    logger = logging.getLogger('val')
    logger.info('[Epoch {}] [Loss {:.3f}] [Acc {:.3f}]'.format(
        epoch,
        val_loss/(batch_idx+1),
        acc))
    #
    if acc > best_val:
        best_val = acc

    return (val_loss/(batch_idx+1), acc)


def savepoint(epoch):
    # Save checkpoint.
    print("Saving epoch...: {} ".format(epoch))
    checkpoint = {
        'net': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    torch.save(checkpoint, os.path.join(logdir, f'ckpt_{epoch}.pth'))


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = args.lr
    if epoch >= 0.5 * args.epoch:
        lr /= 10
    if epoch >= 0.75 * args.epoch:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'clinical_isolates')
    parser = argparse.ArgumentParser(description='PAD Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--model', default="MobileNetV3Small", type=str, help='DRSN_ResNet_6, LSTM, ResNet_6, MobileNetV3Small')
    parser.add_argument('--batch-size', default=10, type=int, help='batch size')
    parser.add_argument('--epoch', default=200, type=int, help='total epochs to run')
    parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--ngpu', default=1, type=int, help='number of gpu')
    parser.add_argument('--sgpu', default=0, type=int, help='gpu index (start)')
    parser.add_argument('--dataset', default='clinical', type=str, help='the name for dataset')
    parser.add_argument('--dataroot', default=data_dir, type=str, help='data directory')
    parser.add_argument('--saveroot', default=current_dir, type=str, help='save directory')
    parser.add_argument('--ads', '-ads', action='store_true', help='adding ads loss')
    parser.add_argument('--temp', default=4.0, type=float, help='temperature scaling')
    parser.add_argument('--lamda', default=0.01, type=float, help='ads loss weight ratio')

    args = parser.parse_args()
    setup_seed(42)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
    use_cuda = torch.cuda.is_available()

    args.ads = True
    if not args.ads:
        ads = 'sd'
        logdir = os.path.join(args.saveroot, args.dataset, args.model, ads)
    else:
        ads = 'PAD'
        logdir = os.path.join(args.saveroot, args.dataset, args.model, ads)
    set_logging_defaults(logdir, args)
    logger = logging.getLogger('main')
    logname = os.path.join(logdir, 'log.csv')

    # args.resume = True
    checkpoint_dir = os.path.join(logdir, 'ckpt_29.pth')

    # best validation accuracy
    best_val = 0
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    # Data
    print('==> Preparing dataset: {}'.format(args.dataset))
    if not args.ads:
        trainloader, valloader = load_dataset(args.dataroot, batch_size=args.batch_size)
        print('class-wise: ', args.ads)
    else:
        trainloader, valloader = load_dataset(args.dataroot, 'pair', batch_size=args.batch_size)
        print('class-wise: ', args.ads)

    num_class = trainloader.dataset.num_classes
    print('Number of train dataset: ', len(trainloader.dataset))
    print('Number of validation dataset: ', len(valloader.dataset))
    print('Number of classes: ', num_class)

    # Model
    print('==> Building model: {}'.format(args.model))
    kdloss = KDLoss(args.temp)
    net = models.load_model(args.model, num_class)
    # print(net)

    if use_cuda:
        torch.cuda.set_device(args.sgpu)
        net.cuda()
        print("Using {} CUDA".format(torch.cuda.device_count()))
    if args.ngpu > 1:
        net = torch.nn.DataParallel(net, device_ids=list(range(args.sgpu, args.sgpu + args.ngpu)))
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)

    # Resume
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(checkpoint_dir)
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        rng_state = checkpoint['rng_state']
        torch.set_rng_state(rng_state)
        print('加载 epoch {} 成功！'.format(start_epoch - 1))
        print('-----------------------------')
    else:
        start_epoch = 0

    criterion = nn.CrossEntropyLoss()

    # Logs
    for epoch in range(start_epoch, args.epoch):
        train_loss, train_acc, train_ads_loss = train(epoch)
        val_loss, val_acc = val(epoch)
        adjust_learning_rate(optimizer, epoch)

        # if (epoch + 1) % 10 == 0:
        #     savepoint(epoch)
        #     print('Saved all parameters!\n')
        savepoint(epoch)
        print('Saved all parameters!\n')

    print("Best Accuracy : {}".format(best_val))
    logger = logging.getLogger('best')
    logger.info('[Acc {:.3f}]'.format(best_val))

    path = os.path.join(logdir, f'{args.model}_acc_loss_{ads}.xlsx')
    data = {
        'Train Loss': train_loss_list,
        'Val Loss': val_loss_list,
        'Train Acc': train_acc_list,
        'Val Acc': val_acc_list
    }
    df = pd.DataFrame(data)
    df.to_excel(path, index=False)
    print(f"数据已保存到 {path}")

    x1 = range(start_epoch, args.epoch)
    x2 = range(start_epoch, args.epoch)
    y1 = train_loss_list
    y2 = val_loss_list
    y3 = train_acc_list
    y4 = val_acc_list

    plt.figure(figsize=(5, 4))

    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'b-o', label='train_loss')
    plt.plot(x1, y2, 'g-*', label='val_loss')
    plt.xticks(range(start_epoch, args.epoch, 50))
    plt.title(' Loss vs. epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim(-0.25, 2)

    plt.subplot(2, 1, 2)
    plt.plot(x2, y3, 'b-o', label='train_acc')
    plt.plot(x2, y4, 'g-*', label='val_acc')
    plt.xticks(range(start_epoch, args.epoch, 50))
    plt.title('Acc. vs. epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim(20, 105)

    # 自动调整子图参数以给标题、标签留出空间
    plt.tight_layout()
    plt.savefig(os.path.join(logdir, f'{args.model}_ccuracy_loss_{ads}.jpg'))
    plt.show()
