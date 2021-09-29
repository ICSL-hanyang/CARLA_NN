# Unet Pytorch Impelementation Reference : https://www.youtube.com/watch?v=sSxdQq9CCx0

import torch
import torch.nn as nn
from torchvision.transforms.functional import pad

import timm

global_print_flag = False

def global_print(print_str):

    global global_print_flag

    if global_print_flag == True:
        print(print_str)

class predictive_ControlNet(nn.Module):

    def __init__(self, in_channels=3, bias=True, verbose='low'):

        super(predictive_ControlNet, self).__init__()

        self.verbose = verbose

        self.adaptive_pool = nn.AdaptiveAvgPool2d((224, 224))

        def Conv_BatchNorm_ActiveF_2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            layer_module = nn.Sequential(*layers)

            return layer_module

        def FC_BatchNorm_ActiveF(in_features, out_features, batchnorm_features, bias):
            layers = []
            layers += [nn.Linear(in_features=in_features, out_features=out_features, bias=bias)]
            layers += [nn.BatchNorm1d(num_features=batchnorm_features)]
            layers += [nn.ReLU()]

            layer_module = nn.Sequential(*layers)

            return layer_module




        cnn_layers = []

        feature_extraction_backbone = timm.create_model(model_name='resnet50', pretrained=True, 
                                                        num_classes=1000, in_chans=in_channels)
        feature_extraction_backbone.reset_classifier(0, '')
        cnn_layers += [feature_extraction_backbone]

        conv1 = Conv_BatchNorm_ActiveF_2d(in_channels=2048, out_channels=2048, kernel_size=3, stride=1, padding=0, bias=bias)
        cnn_layers += [conv1]

        conv2 = Conv_BatchNorm_ActiveF_2d(in_channels=2048, out_channels=2048, kernel_size=3, stride=1, padding=0, bias=bias)
        cnn_layers += [conv2]

        conv3 = Conv_BatchNorm_ActiveF_2d(in_channels=2048, out_channels=2048, kernel_size=3, stride=1, padding=0, bias=bias)
        cnn_layers += [conv3]

        flatten3 = nn.Flatten(start_dim=1)
        cnn_layers += [flatten3]

        self.cnn = nn.Sequential(*cnn_layers)




        fc_layers = []

        fc_1 = FC_BatchNorm_ActiveF(in_features=2048, out_features=1024, batchnorm_features=1024, bias=bias)
        fc_layers += [fc_1]

        fc_2 = FC_BatchNorm_ActiveF(in_features=1024, out_features=512, batchnorm_features=512, bias=bias)
        fc_layers += [fc_2]

        fc_3 = FC_BatchNorm_ActiveF(in_features=512, out_features=256, batchnorm_features=256, bias=bias)
        fc_layers += [fc_3]

        fc_final = nn.Linear(in_features=256, out_features=2, bias=bias)
        fc_layers += [fc_final]

        self.fc = nn.Sequential(*fc_layers)

    def local_print(self, sen, verbose):

        if verbose == 'high':
            if self.verbose == 'high':
                print(sen)
        else:
            print(sen)

    def forward(self, segmented_img):

        global_print('img : {}'.format(segmented_img.size()))

        reshaped_img = self.adaptive_pool(segmented_img)
        global_print('reshaped_img : {}'.format(reshaped_img.size()))

        cnn_out = self.cnn(reshaped_img)
        self.local_print('cnn_out : {}'.format(cnn_out.size()), verbose='high')

        fc_out = self.fc(cnn_out)
        self.local_print('fc_out : {}'.format(cnn_out.size()), verbose='high')

        return fc_out



