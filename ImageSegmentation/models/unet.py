import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, image_size):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Encoder
        self.enc1  = DoubleConv(n_channels, 64) #256 -> 256
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) #256 -> 128
        self.enc2  = DoubleConv(64, 128) #128 -> 128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) #128 -> 64
        self.enc3  = DoubleConv(128, 256) #64 -> 64
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) #64 -> 32
        self.enc4  = DoubleConv(256, 512) #32 -> 32
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) #32 -> 16
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024) #16 -> 16
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2) #16 -> 32
        self.dec4    = DoubleConv(1024, 512) #32 -> 32
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2) #32 -> 64
        self.dec3    = DoubleConv(512, 256) #64 -> 64
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) #64 -> 128
        self.dec2    = DoubleConv(256, 128) #128 -> 128
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) #128 -> 256
        self.dec1    = DoubleConv(128, 64) #256 -> 256
        
        self.outConv = nn.Conv2d(64, n_classes, kernel_size=1)
    
    def forward(self, x):
        x1 = self.enc1(x)
        
        x2 = self.pool1(x1)
        x2 = self.enc2(x2)
        
        x3 = self.pool2(x2)
        x3 = self.enc3(x3)
        
        x4 = self.pool3(x3)
        x4 = self.enc4(x4)
        
        x = self.pool4(x4)   
        x = self.bottleneck(x)

        x = self.upconv4(x)
        x = torch.cat((x4, x), dim=1)
        x = self.dec4(x)
        
        x = self.upconv3(x)
        x = torch.cat((x3, x), dim=1)
        x = self.dec3(x)
        
        x = self.upconv2(x)
        x = torch.cat((x2, x), dim=1)
        x = self.dec2(x)
        
        x = self.upconv1(x)
        x = torch.cat((x1, x), dim=1)
        x = self.dec1(x)
        
        x = self.outConv(x)
   
        return x
        
        
        
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                                         nn.BatchNorm2d(out_channels),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                                         nn.BatchNorm2d(out_channels),
                                         nn.ReLU(inplace=True)
                                         )
    def forward(self, x):
        return self.double_conv(x)
    
