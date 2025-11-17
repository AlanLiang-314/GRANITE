import torch
import torch.nn as nn
import torch.nn.functional as F

choices = ['LeakyReLU', 'ReLU', 'Softplus']
init_choices = ['default', 'xavier', 'kaiming', 'uniform', 'normal']

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, act, stride=1, init_type='default', seed=None):
        super(BasicBlock, self).__init__()
        assert act in choices
        assert init_type in init_choices

        self.arg_act = act
        self.act = {'LeakyReLU': F.leaky_relu, 'ReLU': F.relu, 'Softplus': F.softplus}[act]

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

        self._initialize_weights(init_type, seed)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.act(out)
        return out

    def _initialize_weights(self, init_type, seed):
        if seed is not None:
            torch.manual_seed(seed)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if init_type == 'xavier':
                    nn.init.xavier_uniform_(m.weight)
                elif init_type == 'kaiming':
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                elif init_type == 'uniform':
                    nn.init.uniform_(m.weight)
                elif init_type == 'normal':
                    nn.init.normal_(m.weight)
            
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_channels, act, init_type='default', seed=None, num_classes=10):
        super(ResNet, self).__init__()
        assert act in choices
        assert init_type in init_choices

        self.arg_act = act
        self.act = {'LeakyReLU': F.leaky_relu, 'ReLU': F.relu, 'Softplus': F.softplus}[act]
        self.in_planes = num_channels[0]

        self.conv1 = nn.Conv2d(3, num_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels[0])
        self.layer1 = self._make_layer(block, num_channels[0], num_blocks[0], act, init_type, seed, stride=1)
        self.layer2 = self._make_layer(block, num_channels[1], num_blocks[1], act, init_type, seed, stride=2)
        self.layer3 = self._make_layer(block, num_channels[2], num_blocks[2], act, init_type, seed, stride=2)
        self.layer4 = self._make_layer(block, num_channels[3], num_blocks[3], act, init_type, seed, stride=2)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(num_channels[3] * block.expansion, num_classes)

        self._initialize_weights(init_type, seed)

    def _make_layer(self, block, planes, num_blocks, act, init_type, seed, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, act, stride, init_type, seed))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def _initialize_weights(self, init_type, seed):
        if seed is not None:
            torch.manual_seed(seed)  # 固定初始化随机种子

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if init_type == 'xavier':
                    nn.init.xavier_uniform_(m.weight)
                elif init_type == 'kaiming':
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                elif init_type == 'uniform':
                    nn.init.uniform_(m.weight)
                elif init_type == 'normal':
                    nn.init.normal_(m.weight)
                
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def ResNet18(act='ReLU', init_type='default', seed=None, num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], [64, 128, 256, 512], act, init_type, seed, num_classes=num_classes)


def test():
    seed = 42
    net1 = ResNet18(act='ReLU', init_type='xavier', seed=seed)
    net2 = ResNet18(act='ReLU', init_type='xavier', seed=seed)

    for p1, p2 in zip(net1.parameters(), net2.parameters()):
        assert torch.equal(p1, p2), "Initialization mismatch!"

    print("Initialization is deterministic!")


if __name__ == '__main__':
    test()
