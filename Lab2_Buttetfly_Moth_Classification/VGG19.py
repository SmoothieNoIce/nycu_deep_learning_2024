print("Please define your VGG19 in this file.")

import torch
import torch.nn as nn

class Conv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        groups=1,
        activation=True,
    ):
        super(Conv, self).__init__()
        padding = kernel_size // 2 if padding is None else padding
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=True,
        )
        self.act = nn.ReLU(inplace=True) if activation else nn.Identity()

    def forward(self, x):
        return self.act(self.conv(x))


class VGG19_alt(nn.Module):
    def __init__(self, num_classes):
        super(VGG19_alt, self).__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64), 
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block_4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block_5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(), # 在訓練過程的前向傳播中，讓每個神經元以一定機率p處於不活化的狀態。以達到減少過擬合的效果，eval 會關掉
                nn.Linear(4096, num_classes),
        )
        # weight 使用高斯分佈初始化，mean = 0, variance = 10^-2
        """ for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # VGG采用了Kaiming initialization
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)  
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)  
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)    """  

        # 初始化 weight 超級重要，害我卡好幾天= =
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0)
    
        
    def forward(self, x):

        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x) 
        x = self.avgpool(x)
        # 拉平     
        x = torch.flatten(x, 1)      
        x = self.classifier(x)
        # 一般不用softmax
        # x = F.softmax(x, dim=1)
        return x
    
if __name__ == "__main__":
    a=torch.randn(3,224,224)
    a=a.unsqueeze(0)
    print(a.size())
    net = VGG19_alt(100)
    x=net(a)
    print(x.size())

class VGG19(nn.Module):
    def __init__(self, num_classes):
        super(VGG19, self).__init__()
        self.stages = nn.Sequential(
            *[
                self._make_stage(3, 64, num_blocks=2, max_pooling=True),
                self._make_stage(64, 128, num_blocks=2, max_pooling=True),
                self._make_stage(128, 256, num_blocks=4, max_pooling=True),
                self._make_stage(256, 512, num_blocks=4, max_pooling=True),
                self._make_stage(512, 512, num_blocks=4, max_pooling=True),
            ]
        )
        self.head = nn.Sequential(
            *[
                nn.Flatten(start_dim=1, end_dim=-1),
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            ]
        )   

    @staticmethod
    #
    def _make_stage(in_channels, out_channels, num_blocks, max_pooling):
        layers = [Conv(in_channels, out_channels, kernel_size=3, stride=1)]
        for _ in range(1, num_blocks):
            layers.append(Conv(out_channels, out_channels, kernel_size=3, stride=1))
        if max_pooling:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.head(self.stages(x))


if __name__ == "__main__":
    inputs = torch.rand((8, 3, 224, 224)).cuda()
    model = VGG19(num_classes=100).cuda().train()
    outputs = model(inputs)
    print(outputs.shape)
