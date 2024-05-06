import torch.nn as nn


class DTMFNet(nn.Module):
    def __init__(self):
        super(DTMFNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1) #16
        self.batch1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) #32
        self.batch2 = nn.BatchNorm2d(64)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        self.fc = nn.Linear(64 * 128, 13)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.relu2(x)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(x.size(0), x.size(1), -1)

        x = self.fc(x)
        return x