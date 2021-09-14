import torch

import torch.nn as nn
import torch.optim as optim
from torch import device
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF

import cv2 as cv

import numpy as np
import datetime

from tqdm import tqdm

from dataset_generator import carla_dataset_generator
from dataloader import carla_dataset

from UNet import UNet

start_time = str(datetime.datetime.now())
training_writer = SummaryWriter(log_dir='./runs/' + start_time + '/predictive_vehicle_control_training')
validation_writer = SummaryWriter(log_dir='./runs/' + start_time + '/predictive_vehicle_control_validation')

preprocess = transforms.Compose([
    transforms.Resize((640, 1280)),
    transforms.ToTensor(),
])
postprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

carla_dataset_generator = carla_dataset_generator(dataset_save_path='./carla_dataset.hdf5',
                                                  dataset_path='/home/luwis/ICSL_Project/CARLA_NN/CARLA_dev_client/PythonAPI/automatic_dataset_collection/Recorded_Image/2021-09-14 16:20:03.124257',
                                                  verbose='low')

predictive_vehicle_control_dataset_training = carla_dataset(dataset_path='./carla_dataset.hdf5', mode='training',
                                                            train_pretransform=preprocess, train_posttransform=postprocess,
                                                            valid_pretransform=preprocess, valid_posttransform=postprocess,
                                                            test_pretransform=preprocess, test_posttransform=postprocess)

predictive_vehicle_control_dataset_validation = carla_dataset(dataset_path='./carla_dataset.hdf5', mode='validation',
                                                            train_pretransform=preprocess, train_posttransform=postprocess,
                                                            valid_pretransform=preprocess, valid_posttransform=postprocess,
                                                            test_pretransform=preprocess, test_posttransform=postprocess)

training_dataloader = DataLoader(dataset=predictive_vehicle_control_dataset_training, batch_size=16, shuffle=True, drop_last=True, num_workers=5, prefetch_factor=5, persistent_workers=False, pin_memory=False)

validation_dataloader = DataLoader(dataset=predictive_vehicle_control_dataset_validation, batch_size=16, shuffle=True, drop_last=True, num_workers=5, prefetch_factor=5, persistent_workers=False, pin_memory=False)

processor = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(processor)

predictive_img_model = UNet(in_channels=3)
predictive_img_model.to(processor)

learning_rate = 1e-4

optimizer = optim.SGD(predictive_img_model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)

criterion = nn.MSELoss()

for epoch in range(400):

    total_train_loss_list = []
    total_valid_loss_list = []

    print('Current State [Training] - [EPOCH : {}]'.format(str(epoch)))
    predictive_img_model.train()
    for batch_idx, (prev_img, current_img, vehicle_control_feature_vector) in enumerate(tqdm(training_dataloader)):

        prev_img = prev_img.to(processor).float()
        current_img = current_img.to(processor).float()
        vehicle_control_feature_vector = vehicle_control_feature_vector.to(processor).float()

        # print('prev_img : {}'.format(prev_img.size()))
        # print('current_img : {}'.format(current_img.size()))
        # print('vehicle_control_feature_vector : {}'.format(vehicle_control_feature_vector.size()))

        optimizer.zero_grad()
        reconstructed_img = predictive_img_model(prev_img, vehicle_control_feature_vector)
        reconstruction_loss = criterion(reconstructed_img, current_img)
        reconstruction_loss.backward()
        optimizer.step()

        total_train_loss_list.append(reconstruction_loss.item())

    training_writer.add_scalar('Next Image Reconstruction Loss [{}]'.format(start_time), np.mean(total_train_loss_list), global_step=epoch)

    print('Current State [Validation] - [EPOCH : {}]'.format(str(epoch)))
    predictive_img_model.eval()
    with torch.no_grad():

        for batch_idx, (prev_img, current_img, vehicle_control_feature_vector) in enumerate(tqdm(validation_dataloader)):

            prev_img = prev_img.to(processor).float()
            current_img = current_img.to(processor).float()
            vehicle_control_feature_vector = vehicle_control_feature_vector.to(processor).float()

            # print('prev_img : {}'.format(prev_img.size()))
            # print('current_img : {}'.format(current_img.size()))
            # print('vehicle_control_feature_vector : {}'.format(vehicle_control_feature_vector.size()))

            reconstructed_img = predictive_img_model(prev_img, vehicle_control_feature_vector)
            reconstruction_loss = criterion(reconstructed_img, current_img)

            total_valid_loss_list.append(reconstruction_loss.item())

        validation_writer.add_scalar('Next Image Reconstruction Loss [{}]'.format(start_time), np.mean(total_valid_loss_list), global_step=epoch)
