import torch
from torch import nn
from torchinfo import summary


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels * BasicBlock.expansion),
        )
        # shortcut
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class RSNet(nn.Module):

    def __init__(self, block, num_block, num_classes=4):
        super(RSNet, self).__init__()

        self.in_channels = 16

        self.conv1 = nn.Sequential(

            nn.Conv1d(1, 16, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
            )
        self.layer1 = self._make_layer(block, 32, num_block[0], 1)
        self.layer2 = self._make_layer(block, 64, num_block[1], 2)
        self.layer3 = self._make_layer(block, 128, num_block[2], 2)
        self.layer4 = self._make_layer(block, 256, num_block[3], 2)
        # 额外添加
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def ResNet_10(**kwargs):

    return RSNet(BasicBlock, [1, 1, 1, 1])

#
if __name__ == '__main__':
    model = ResNet_10()
    # print(model)

    input = torch.ones((64, 1, 2126))
    output = model(input)
    print(output.shape)
    summary(model, (1, 1, 2126))
