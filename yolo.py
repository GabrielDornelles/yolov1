import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, pad) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                kernel_size=kernel_size, stride=stride, padding=pad, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class LargeConvBlock(nn.Module):

    def __init__(self, in_channels=256, out_channels=1024) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=256, 
            kernel_size=1, stride=1, pad=0), # 1x1x256
            ConvBlock(in_channels=256, out_channels=512, 
            kernel_size=3, stride=1, pad=1), # 3x3x512

            ConvBlock(in_channels=512, out_channels=256, 
            kernel_size=1, stride=1, pad=0), # 1x1x256
            ConvBlock(in_channels=256, out_channels=512, 
            kernel_size=3, stride=1, pad=1), # 3x3x512

            ConvBlock(in_channels=512, out_channels=256, 
            kernel_size=1, stride=1, pad=0), # 1x1x256
            ConvBlock(in_channels=256, out_channels=512, 
            kernel_size=3, stride=1, pad=1), # 3x3x512

            ConvBlock(in_channels=512, out_channels=256, 
            kernel_size=1, stride=1, pad=0), # 1x1x256
            ConvBlock(in_channels=256, out_channels=512, 
            kernel_size=3, stride=1, pad=1), # 3x3x512

            ConvBlock(in_channels=512, out_channels=512, 
            kernel_size=1, stride=1, pad=0), # 1x1x512

            ConvBlock(in_channels=512, out_channels=out_channels, 
            kernel_size=3, stride=1, pad=1), # 3x3x1024
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Yolo(nn.Module):

    def __init__(self, num_boxes, num_classes) -> None:
        super().__init__()
        self.B = num_boxes # num of boxes per grid I think
        self.C = num_classes
        self.conv1 = ConvBlock(in_channels=3, out_channels=64, kernel_size=7, stride=2, pad=3)
        self.conv2 = ConvBlock(in_channels=64, out_channels=192, kernel_size=3, stride=1, pad=1)
        self.conv3 =  nn.Sequential(
            ConvBlock(in_channels=192, out_channels=128, 
            kernel_size=1, stride=1, pad=0), # 1x1x128

            ConvBlock(in_channels=128, out_channels=256, 
            kernel_size=3, stride=1, pad=1), # 3x3x256

            ConvBlock(in_channels=256, out_channels=256, 
            kernel_size=1, stride=1, pad=0), # 1x1x256

            ConvBlock(in_channels=256, out_channels=512, 
            kernel_size=3, stride=1, pad=1) # 3x3x512
        )
        self.conv4 = LargeConvBlock(in_channels=512, out_channels=1024)
        self.conv5 = nn.Sequential(
            ConvBlock(in_channels=1024, out_channels=512, 
            kernel_size=1, stride=1, pad=0), # 1x1x512
            ConvBlock(in_channels=512, out_channels=1024, 
            kernel_size=3, stride=1, pad=1), # 3x3x1024

            ConvBlock(in_channels=1024, out_channels=512, 
            kernel_size=1, stride=1, pad=0), # 1x1x512
            ConvBlock(in_channels=512, out_channels=1024, 
            kernel_size=3, stride=1, pad=1), # 3x3x1024

            ConvBlock(in_channels=1024, out_channels=1024, 
            kernel_size=3, stride=1, pad=1), # 3x3x1024

            ConvBlock(in_channels=1024, out_channels=1024, 
            kernel_size=3, stride=2, pad=1), # 3x3x1024s-2
        )

        self.conv6 = nn.Sequential(
            ConvBlock(in_channels=1024, out_channels=1024, 
            kernel_size=3, stride=1, pad=1), # 3x3x1024

            ConvBlock(in_channels=1024, out_channels=1024, 
            kernel_size=3, stride=1, pad=1), # 3x3x1024
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(50176, 4096), # 50176 for 448x448 images, 16384 for 224x224
            nn.LeakyReLU(0.1), # skipped dropout, but it has 0.5 in the paper
            nn.Linear(4096, 7 * 7 * (self.C + self.B * 5)), 
        )

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        x = self.maxpool(self.conv2(x))
        x = self.maxpool(self.conv3(x))
        x = self.maxpool(self.conv4(x))
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.fc(x)
        # x = x.view((7,7,30)) # dont do that here, problems with batch_size > 1
        return x

if __name__ == "__main__":
    model  = Yolo(num_boxes=2, num_classes=20) # 2 boxes per grid, 20 classes (Pascal VOC)
    sample = torch.randn((1,3,448,448))
    output = model(sample)
    print(output.shape)