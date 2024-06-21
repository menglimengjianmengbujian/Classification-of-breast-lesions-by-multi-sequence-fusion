import torch
import torch.nn as nn
import torchvision.models as models


# 定义自定义的ResNet-18网络
class CustomResNet18(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CustomResNet18, self).__init__()

        # 加载预训练的ResNet-18模型
        resnet = models.resnet18(pretrained=True)

        # 修改第一个卷积层的输入通道数
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 修改全连接层的输出特征数
        resnet.fc = nn.Linear(512, num_classes)

        # 将修改后的模型赋值给自定义的ResNet-18网络
        self.model = resnet

    def forward(self, x):
        features = self.model.conv1(x)
        features = self.model.bn1(features)
        features = self.model.relu(features)
        features = self.model.maxpool(features)

        features = self.model.layer1(features)
        features = self.model.layer2(features)
        features = self.model.layer3(features)
        features = self.model.layer4(features)

        output = self.model.avgpool(features)
        output1 = output.view(output.size(0), -1)
        output = self.model.fc(output1)

        return output1, output


# # 创建一个ResNet-18网络实例
# resnet18 = CustomResNet18(in_channels=1, num_classes=2)
#
# # 打印网络结构
# print(resnet18)
#
# # 在输入数据上进行前向传播
# input_data = torch.randn(1, 1, 224, 224)  # 假设输入数据尺寸为1x224x224
# features, output = resnet18(input_data)
# print("前连接层特征尺寸:", features.size())
# print("最后输出尺寸:", output.size())