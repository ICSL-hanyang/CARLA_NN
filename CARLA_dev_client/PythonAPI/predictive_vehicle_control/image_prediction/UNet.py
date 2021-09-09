# Unet Pytorch Impelementation Reference : https://www.youtube.com/watch?v=sSxdQq9CCx0

import torch
import torch.nn as nn
from torchvision.transforms.functional import pad

global_print_flag = False

def global_print(print_str):

    global global_print_flag

    if global_print_flag == True:
        print(print_str)

class UNet(nn.Module):

    def __init__(self, in_channels=3):
        super(UNet, self).__init__()

        def Conv_BatchNorm_ActiveF_2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            layer_module = nn.Sequential(*layers)

            return layer_module

        # Encoding Path
        self.enc1_1 = Conv_BatchNorm_ActiveF_2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.enc1_2 = Conv_BatchNorm_ActiveF_2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = Conv_BatchNorm_ActiveF_2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.enc2_2 = Conv_BatchNorm_ActiveF_2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = Conv_BatchNorm_ActiveF_2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)
        self.enc3_2 = Conv_BatchNorm_ActiveF_2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = Conv_BatchNorm_ActiveF_2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True)
        self.enc4_2 = Conv_BatchNorm_ActiveF_2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True)

        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = Conv_BatchNorm_ActiveF_2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=True)

        # Decoding Path
        self.dec5_1 = Conv_BatchNorm_ActiveF_2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True)

        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2, stride=2, padding=0, output_padding=1, bias=True)

        self.dec4_2 = Conv_BatchNorm_ActiveF_2d(in_channels=(512 + 512), out_channels=512, kernel_size=3, stride=1, padding=1, bias=True)
        self.dec4_1 = Conv_BatchNorm_ActiveF_2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)

        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0, output_padding=1, bias=True)

        self.dec3_2 = Conv_BatchNorm_ActiveF_2d(in_channels=(256 + 256), out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)
        self.dec3_1 = Conv_BatchNorm_ActiveF_2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)

        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = Conv_BatchNorm_ActiveF_2d(in_channels=(128 + 128), out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.dec2_1 = Conv_BatchNorm_ActiveF_2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)

        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_2 = Conv_BatchNorm_ActiveF_2d(in_channels=(64 + 64), out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.dec1_1 = Conv_BatchNorm_ActiveF_2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)

        # 1x1 Conv-based 3 Channel Image Reconstruction
        self.conv_fc = nn.Conv2d(in_channels=64, out_channels=in_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, img):

        global_print('img : {}'.format(img.size()))

        # Encoding Path
        enc1_1 = self.enc1_1(img)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)
        global_print('pool1 : {}'.format(pool1.size()))

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)
        global_print('pool2 : {}'.format(pool2.size()))

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)
        global_print('pool3 : {}'.format(pool3.size()))

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)
        global_print('pool4 : {}'.format(pool4.size()))

        enc5_1 = self.enc5_1(pool4)
        global_print('enc5_1 : {}'.format(enc5_1.size()))

        # Decoding Path
        dec5_1 = self.dec5_1(enc5_1)
        global_print('dec5_1 : {}'.format(dec5_1.size()))

        unpool4 = self.unpool4(dec5_1)
        global_print('unpool4 : {}'.format(unpool4.size()))

        cat4 = torch.cat((unpool4, enc4_2), dim=1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)
        global_print('dec4_1 : {}'.format(dec4_1.size()))

        unpool3 = self.unpool3(dec4_1)
        global_print('unpool3 : {}'.format(unpool3.size()))

        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)
        global_print('dec3_1 : {}'.format(dec3_1.size()))

        unpool2 = self.unpool2(dec3_1)
        global_print('unpool2 : {}'.format(unpool2.size()))

        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)
        global_print('dec2_1 : {}'.format(dec2_1.size()))

        unpool1 = self.unpool1(dec2_1)
        global_print('unpool1 : {}'.format(unpool1.size()))

        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)
        global_print('dec1_1 : {}'.format(dec1_1.size()))

        # 1x1 Conv-based 3 Channel Image Reconstruction
        reconstructed_img = self.conv_fc(dec1_1)
        global_print('reconstructed_img : {}'.format(reconstructed_img.size()))

        return reconstructed_img



