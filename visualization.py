import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def normalize(x):
    return x / (torch.sqrt(torch.mean(torch.square(x))) + 1e-5)


def interp_1d(array, shape):
    res = np.zeros(shape)
    if array.shape[0] >= shape:
        ratio = array.shape[0]/shape
        for i in range(array.shape[0]):
            res[int(i/ratio)] += array[i]*(1-(i/ratio-int(i/ratio)))
            if int(i/ratio) != shape-1:
                res[int(i/ratio)+1] += array[i]*(i/ratio-int(i/ratio))
            else:
                res[int(i/ratio)] += array[i]*(i/ratio-int(i/ratio))
        res = res[::-1]
        array = array[::-1]
        for i in range(array.shape[0]):
            res[int(i/ratio)] += array[i]*(1-(i/ratio-int(i/ratio)))
            if int(i/ratio) != shape-1:
                res[int(i/ratio)+1] += array[i]*(i/ratio-int(i/ratio))
            else:
                res[int(i/ratio)] += array[i]*(i/ratio-int(i/ratio))
        # [::-1] turn operation
        res = res[::-1]/(2*ratio)
        array = array[::-1]
    else:
        ratio = shape/array.shape[0]
        left = 0
        right = 1
        for i in range(shape):
            if left < int(i/ratio):
                left += 1
                right += 1
            if right > array.shape[0]-1:
                res[i] += array[left]
            else:
                res[i] += array[right] * \
                    (i - left * ratio)/ratio+array[left]*(right*ratio-i)/ratio
        res = res[::-1]
        array = array[::-1]
        left = 0
        right = 1
        for i in range(shape):
            if left < int(i/ratio):
                left += 1
                right += 1
            if right > array.shape[0]-1:
                res[i] += array[left]
            else:
                res[i] += array[right] * \
                    (i - left * ratio)/ratio+array[left]*(right*ratio-i)/ratio
        res = res[::-1]/2
        array = array[::-1]
    return res


class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = []
        self.activations = []

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations.append(output)

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        self.gradients = [grad_output[0]] + self.gradients

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)


class BaseCAM:
    def __init__(self, model, target_layer, use_cuda=False):
        self.model = model.eval()
        self.target_layer = target_layer
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.activations_and_grads = ActivationsAndGradients(self.model, target_layer)

    def forward(self, input_img):
        return self.model(input_img)

    def get_cam_weights(self,
                        input_tensor,
                        target_category,
                        activations,
                        grads):
        raise Exception("Not Implemented")

    def get_loss(self, output, target_category):
        return output[0, target_category]

    def __call__(self, input_tensor, target_category=None):
        if self.cuda:
            input_tensor = input_tensor.cuda()

        output = self.activations_and_grads(input_tensor)

        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy())
            print('default (pre) class', target_category)
        else:
            # 检查 target_category 是否有效
            num_classes = output.shape[1]
            if target_category < 0 or target_category >= num_classes:
                raise ValueError(
                    f"Invalid target_category {target_category}. It must be in the range [0, {num_classes - 1}]")

        self.model.zero_grad()
        loss = self.get_loss(output, target_category)

        loss.backward(retain_graph=True)
        # 取批次第一个样本 激活值和梯度列表里最后一个数组
        activations = self.activations_and_grads.activations[-1].cpu().data.numpy()[0, :]
        grads = self.activations_and_grads.gradients[-1].cpu().data.numpy()[0, :]
        weights = self.get_cam_weights(input_tensor, target_category, activations, grads)
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :]
        cam = interp_1d(cam, (input_tensor.shape[2]))
        cam = np.maximum(cam, 0)
        heatmap = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-10)
        return heatmap


class GradCAM(BaseCAM):
    def __init__(self, model, target_layer, use_cuda=False):
        super(GradCAM, self).__init__(model, target_layer, use_cuda)

    def get_cam_weights(self, input_tensor,
                              target_category,
                              activations, grads):
        return np.mean(grads, axis=1)


def visualization_save_acc_loss(logdir, model, loss, start_epoch, total_epoch, train_loss_list, val_loss_list,
                                train_acc_list, val_acc_list):
    path = os.path.join(logdir, f'{model}_acc_loss_{loss}.xlsx')
    data = {
        'Train Loss': train_loss_list,
        'Val Loss': val_loss_list,
        'Train Acc': train_acc_list,
        'Val Acc': val_acc_list
    }
    df = pd.DataFrame(data)
    df.to_excel(path, index=False)
    print(f"Data is saved to {path}")

    x1 = range(start_epoch, total_epoch)
    x2 = range(start_epoch, total_epoch)
    y1 = train_loss_list
    y2 = val_loss_list
    y3 = train_acc_list
    y4 = val_acc_list

    plt.figure(figsize=(5, 4))

    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'b-o', label='train_loss')
    plt.plot(x1, y2, 'g-*', label='val_loss')
    plt.xticks(range(start_epoch, total_epoch, 50))
    plt.title(' Loss vs. epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim(-0.25, 2)

    plt.subplot(2, 1, 2)
    plt.plot(x2, y3, 'b-o', label='train_acc')
    plt.plot(x2, y4, 'g-*', label='val_acc')
    plt.xticks(range(start_epoch, total_epoch, 50))
    plt.title('Acc. vs. epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim(20, 105)

    plt.tight_layout()
    plt.savefig(os.path.join(logdir, f'{model}_ccuracy_loss_{loss}.jpg'))
    plt.show()


def confusion_matrix(preds, labels, conf_matrix):
    preds = torch.argmax(preds, 1)


    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
        # if p != t:
        #     incorrect_indices.append(idx)


    return conf_matrix
