import torch
import torch.nn.functional as F
import torch.nn as nn

class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        # 첫번째층
        # ImgIn shape=(?, 28, 28, 1)
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 두번째층
        # ImgIn shape=(?, 14, 14, 32)
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 전결합층 7x7x64 inputs -> 10 outputs
        self.fc = torch.nn.Linear(7 * 7 * 64, 10, bias=True)

        # 전결합층 한정으로 가중치 초기화
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)  # 전결합층을 위해서 Flatten
        out = self.fc(out)
        return out




class BasicBlock(nn.Module):
    # mul은 추후 ResNet18, 34, 50, 101, 152등 구조 생성에 사용됨
    mul = 1

    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicBlock, self).__init__()


        # stride를 통해 너비와 높이 조정
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)

        # stride = 1, padding = 1이므로, 너비와 높이는 항시 유지됨
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

        # x를 그대로 더해주기 위함
        self.shortcut = nn.Sequential()

        # 만약 size가 안맞아 합연산이 불가하다면, 연산 가능하도록 모양을 맞춰줌
        if stride != 1:  # x와
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)  # 필요에 따라 layer를 Skip
        out = F.relu(out)


        return out


class BottleNeck(nn.Module):
    # 논문의 구조를 참고하여 mul 값은 4로 지정, 즉, 64 -> 256
    mul = 4

    def __init__(self, in_planes, out_planes, stride=1):
        super(BottleNeck, self).__init__()


        # 첫 Convolution은 너비와 높이 downsampling
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)

        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.conv3 = nn.Conv2d(out_planes, out_planes * self.mul, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes * self.mul)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != out_planes * self.mul:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes * self.mul, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes * self.mul)
            )

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks,num_classes=10,mnst_ver=True):
        super(ResNet, self).__init__()
        # RGB 3개채널에서 64개의 Kernel 사용 (논문 참고)
        self.in_planes = 64

        # Resnet 논문 구조의 conv1 파트 그대로 구현
        if mnst_ver ==True:
            self.conv1 = nn.Conv2d(1, self.in_planes, kernel_size=7, stride=2, padding=3)
        else:
            self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Basic Resiudal Block일 경우 그대로, BottleNeck일 경우 4를 곱한다.
        self.linear = nn.Linear(512 * block.mul, num_classes)

    # 다양한 Architecture 생성을 위해 make_layer로 Sequential 생성
    def make_layer(self, block, out_planes, num_blocks, stride):
        # layer 앞부분에서만 크기를 절반으로 줄이므로, 아래와 같은 구조
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            layers.append(block(self.in_planes, out_planes, strides[i]))
            self.in_planes = block.mul * out_planes
        return nn.Sequential(*layers)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.maxpool1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #print(out.size())
        out = self.avgpool(out)
        #print(out.size())
        out = torch.flatten(out, 1)
        out = self.linear(out)
        #print(out.size())


        return out


class ResNetBackbone(nn.Module):
    def __init__(self, block, num_blocks,last_out_feature=1000):
        super(ResNetBackbone, self).__init__()
        # RGB 3개채널에서 64개의 Kernel 사용 (논문 참고)
        self.in_planes = 64

        # Resnet 논문 구조의 conv1 파트 그대로 구현

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(512 * block.mul, last_out_feature)


    # 다양한 Architecture 생성을 위해 make_layer로 Sequential 생성
    def make_layer(self, block, out_planes, num_blocks, stride):
        # layer 앞부분에서만 크기를 절반으로 줄이므로, 아래와 같은 구조
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            layers.append(block(self.in_planes, out_planes, strides[i]))
            self.in_planes = block.mul * out_planes
        return nn.Sequential(*layers)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.maxpool1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = F.relu(self.linear(out))

        return out


class Datacon1model(nn.Module):
    def __init__(self,modelKind,backboneOutFeature, LinNum,totalCropNum):
        super(Datacon1model,self).__init__()

        if modelKind == 'resnet18':
            self.backbone = ResNetBackbone(block=BasicBlock,num_blocks=[2,2,2,2],last_out_feature=backboneOutFeature)
        elif modelKind == 'resnet34':
            self.backbone = ResNetBackbone(block=BasicBlock,num_blocks=[3,4,6,3],last_out_feature=backboneOutFeature)
        elif modelKind == 'resnet50':
            self.backbone = ResNetBackbone(block=BottleNeck,num_blocks=[3,4,6,3],last_out_feature=backboneOutFeature)
        elif modelKind == 'resnet101':
            self.backbone = ResNetBackbone(block=BottleNeck,num_blocks=[3,4,23,3],last_out_feature=backboneOutFeature)
        else:
            self.backbone = ResNetBackbone(block=BottleNeck,num_blocks=[3,8,36,3],last_out_feature=backboneOutFeature)

        self.backboneOutFeature = backboneOutFeature

        self.lin1 = nn.Linear(in_features=backboneOutFeature,out_features=2*LinNum)
        self.lin2 = nn.Linear(in_features=2*LinNum, out_features= LinNum)
        self.lin3 = nn.Linear(in_features= LinNum, out_features= totalCropNum)


    def forward(self,x):

        out = self.backbone(x)
        out = F.relu(self.lin1(out))
        out = F.relu(self.lin2(out))
        out = self.lin3(out)

        return out
























class BasicBlock4one(nn.Module):
    # mul은 추후 ResNet18, 34, 50, 101, 152등 구조 생성에 사용됨
    mul = 1

    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicBlock4one, self).__init__()


        # stride를 통해 너비와 높이 조정
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


        # stride = 1, padding = 1이므로, 너비와 높이는 항시 유지됨
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

        # x를 그대로 더해주기 위함
        self.shortcut = nn.Sequential()

        # 만약 size가 안맞아 합연산이 불가하다면, 연산 가능하도록 모양을 맞춰줌
        if stride != 1:  # x와

            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):

        out = self.conv1(x)


        out = F.relu(out)

        out = self.conv2(out)


        out += self.shortcut(x)  # 필요에 따라 layer를 Skip

        out = F.relu(out)


        return out


class BottleNeck4one(nn.Module):
    # 논문의 구조를 참고하여 mul 값은 4로 지정, 즉, 64 -> 256
    mul = 4

    def __init__(self, in_planes, out_planes, stride=1):
        super(BottleNeck4one, self).__init__()

        # 첫 Convolution은 너비와 높이 downsampling
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)


        self.conv3 = nn.Conv2d(out_planes, out_planes * self.mul, kernel_size=1, stride=1, bias=False)


        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != out_planes * self.mul:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes * self.mul, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes * self.mul)
            )

    def forward(self, x):

        out = self.conv1(x)

        out = F.relu(out)
        out = self.conv2(out)

        out = F.relu(out)
        out = self.conv3(out)

        out += self.shortcut(x)
        out = F.relu(out)

        return out


class ResNet4one(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, mnst_ver=True):
        super(ResNet4one, self).__init__()
        # RGB 3개채널에서 64개의 Kernel 사용 (논문 참고)
        self.in_planes = 64

        # Resnet 논문 구조의 conv1 파트 그대로 구현
        if mnst_ver == True:
            self.conv1 = nn.Conv2d(1, self.in_planes, kernel_size=7, stride=2, padding=3)
        else:
            self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3)

        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Basic Resiudal Block일 경우 그대로, BottleNeck일 경우 4를 곱한다.
        self.linear = nn.Linear(512 * block.mul, num_classes)

    # 다양한 Architecture 생성을 위해 make_layer로 Sequential 생성
    def make_layer(self, block, out_planes, num_blocks, stride):
        # layer 앞부분에서만 크기를 절반으로 줄이므로, 아래와 같은 구조
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            layers.append(block(self.in_planes, out_planes, strides[i]))
            self.in_planes = block.mul * out_planes
        return nn.Sequential(*layers)

    def forward(self, x):

        out = self.conv1(x)

        out = F.relu(out)

        out = self.maxpool1(out)

        out = self.layer1(out)

        out = self.layer2(out)

        out = self.layer3(out)

        out = self.layer4(out)

        out = self.avgpool(out)

        out = torch.flatten(out, 1)
        out = self.linear(out)

        return out
