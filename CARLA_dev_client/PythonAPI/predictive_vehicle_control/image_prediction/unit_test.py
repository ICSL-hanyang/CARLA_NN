import torch

import torch.nn as nn
import torch.optim as optim
from torch import device
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF

import cv2 as cv

import numpy as np

from tqdm import tqdm

from dataset_generator import carla_dataset_generator
from dataloader import carla_dataset

from UNet import UNet

preprocess = transforms.Compose([
    transforms.Resize((640, 1280)),
    transforms.ToTensor(),
])
postprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

carla_dataset_generator = carla_dataset_generator(dataset_save_path='./carla_dataset.hdf5',
                                                  dataset_path='/home/luwis/ICSL_Project/CARLA_NN/CARLA_dev_client/PythonAPI/automatic_dataset_collection/Recorded_Image/2021-09-08 23:50:26.834826',
                                                  verbose='low')

predictive_vehicle_control_dataset = carla_dataset(dataset_path='./carla_dataset.hdf5', mode='training',
                                          train_pretransform=preprocess, train_posttransform=postprocess,
                                          valid_pretransform=preprocess, valid_posttransform=postprocess,
                                          test_pretransform=preprocess, test_posttransform=postprocess)

dataloader = DataLoader(dataset=predictive_vehicle_control_dataset, batch_size=1, shuffle=True, drop_last=True)

processor = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(processor)

predictive_img_model = UNet(in_channels=3)
predictive_img_model.to(processor)

learning_rate = 1e-4

optimizer = optim.Adam(predictive_img_model.parameters(), lr=learning_rate)

criterion = nn.MSELoss()

for epoch in range(10):

    total_train_loss_list = []

    predictive_img_model.eval()

    for batch_idx, (prev_img, current_img, vehicle_control_feature_vector) in enumerate(tqdm(dataloader)):

        # prev_img = prev_img.to(processor).float()
        # current_img = current_img.to(processor).float()

        print('prev_img : {}'.format(prev_img.size()))
        print('current_img : {}'.format(current_img.size()))
        print('vehicle_control_feature_vector : {}'.format(vehicle_control_feature_vector.size()))

        # optimizer.zero_grad()
        # reconstructed_img = predictive_img_model(prev_img)
        # reconstruction_loss = criterion(reconstructed_img, current_img)
        # reconstruction_loss.backward()
        # optimizer.step()

        # total_train_loss_list.append(reconstruction_loss.item())

    # print('reconstruction_loss : {}'.format(np.mean(total_train_loss_list)))