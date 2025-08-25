import argparse
import logging
import os
import torch
from matplotlib import pyplot as plt
from datasets import load_test_dataset
import models
from utils import set_logging_defaults
from visualization import confusion_matrix


def test(epoch):
    global best_test
    net.eval()
    correct = 0.0
    total = 0.0
    loader = testloader
    conf_matrix = torch.zeros(4, 4)
    incorrect_indices = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs = inputs.reshape(args.batch_size, 1, 2126)
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            F_odd = inputs[:, :, 1::2] 
            F_even = inputs[:, :, ::2]  
            min_length = min(F_odd.shape[2], F_even.shape[2])
            inputs_x = F_odd[:, :, :min_length]
            inputs_y = F_even[:, :, :min_length]

            outputs_x = net(inputs_x)
            outputs_y = net(inputs_y)
            combined_outputs = outputs_x + outputs_y

            _, predicted = torch.max(combined_outputs, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().float()
            conf_matrix = confusion_matrix(combined_outputs, targets, conf_matrix)

    incorrect_batch_indices = (predicted != targets).nonzero(as_tuple=True)[0]
    incorrect_indices.extend(incorrect_batch_indices.cpu().numpy().tolist())
    print("Incorrect indices:", incorrect_indices)
    acc = 100. * correct / total
    test_acc_list.append(acc)
    if acc > best_test:
        best_test = acc
    logger = logging.getLogger('test')
    logger.info('[Epoch {}] [Acc {:.3f}] [Confu_matrix {} ] [incorrect_indices {}]'.format(
        epoch, acc, conf_matrix, incorrect_indices))


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    parser = argparse.ArgumentParser(description='xxx')
    parser.add_argument('--model', default="xxx", type=str, help='DRSN_ResNet_6, LSTM, ResNet_6, MobileNetV3Small')
    parser.add_argument('--batch-size', default=100, type=int, help='batch size')
    parser.add_argument('--dataset', default='seaweed', type=str, help='the name for dataset')
    parser.add_argument('--dataroot', default=data_dir, type=str, help='data directory')
    parser.add_argument('--saveroot', default=current_dir, type=str, help='save directory')
    parser.add_argument('--sgpu', default=0, type=int, help='gpu index (start)')
    parser.add_argument('--ads', '-ads', action='store_true', help='adding ads loss')
    parser.add_argument('--epoch', default=200, type=int, help='total epochs to run')

    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    testloader = load_test_dataset(data_dir, batch_size=args.batch_size)
    num_class = testloader.dataset.num_classes
    logdir = os.path.join(args.saveroot, args.dataset, args.model)

    net = models.load_model(args.model, num_class)

    test_acc_list = []
    best_test = 0
    ads="xx"
    logdir = os.path.join(args.saveroot, args.dataset, args.model, ads)

    logger = logging.getLogger('main')
    logname = os.path.join(logdir, 'log.csv')

    if use_cuda:
        torch.cuda.set_device(args.sgpu)
        net.cuda()
        print("Using {} CUDA".format(torch.cuda.device_count()))

    for epoch in range(0, args.epoch, 1):
        checkpoint_dir = os.path.join(logdir, f'ckpt_{epoch}.pth')
        checkpoint = torch.load(checkpoint_dir)
        net.load_state_dict(checkpoint['net'])
        test(epoch)
    print('Final Best Accuracy: {}'.format(best_test))
    logger = logging.getLogger('best')
    logger.info('[Acc {:.3f}]'.format(best_test))

    path = os.path.join(logdir, f'{args.model}_test_acc_{ads}.txt')
    with open(path, 'w') as f:
        f.write("Test Acc\n")
     
        for acc in test_acc_list:
            f.write(f"{acc}\n")
    print("data is saved at test_acc.txt")

    x1 = range(0, args.epoch)
    y1 = test_acc_list
    plt.figure(figsize=(5, 4))
    plt.plot(x1, y1, 'b-o', label='test_acc')
    plt.xticks(range(0, args.epoch+1, 50))
    plt.title(' Test vs. epochs')
    plt.ylabel('Acc')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim(20, 100)

    plt.savefig(os.path.join(logdir, f'{args.model}_test_{ads}.jpg'))
    plt.show()
