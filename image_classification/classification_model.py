import torch
from torch import nn


# 自定义神经网络类：基于CNN的去噪器
class Classifier(nn.Module):
    def __init__(self,n_classes=5):
        super(Classifier,self).__init__()
        self.model=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=8,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Flatten(),
            nn.Linear(in_features=4096,out_features=n_classes),
        )
    def forward(self,x):
        return self.model(x)
if __name__ == '__main__':
    input = torch.randn(32, 3, 64, 64)
    model = Classifier()
    output = model(input)
    print(output.shape)