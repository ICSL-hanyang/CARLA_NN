import torch

import torch.nn as nn
import torch.optim as optim
from torch import device
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF

import PIL

import cv2 as cv

import numpy as np
import datetime

from tqdm import tqdm

from dataset_generator import carla_dataset_generator, N_frame_delay_carla_dataset_generator
from dataloader import carla_dataset

from UNet import UNet

from predictive_ControlNet import predictive_ControlNet

start_time = str(datetime.datetime.now())

preprocess = transforms.Compose([
    transforms.Resize((640, 1280)),
    transforms.ToTensor(),
])
postprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

n_frame_delay = 1

if n_frame_delay > 1:

    carla_dataset_generator = N_frame_delay_carla_dataset_generator(dataset_save_path='./carla_dataset.hdf5',
                                                                    dataset_path='/home/luwis/ICSL_Project/CARLA_NN/CARLA_dev_client/PythonAPI/automatic_dataset_collection/Recorded_Image/2021-09-19 19:13:13.616261_with_segmentation',
                                                                    n_frame_delay=n_frame_delay,
                                                                    verbose='low')

    predictive_vehicle_control_dataset_training = carla_dataset(dataset_path='./carla_dataset[frame_delay_' + str(n_frame_delay) + '].hdf5', mode='training',
                                                                train_pretransform=preprocess, train_posttransform=postprocess,
                                                                valid_pretransform=preprocess, valid_posttransform=postprocess,
                                                                test_pretransform=preprocess, test_posttransform=postprocess)

    predictive_vehicle_control_dataset_validation = carla_dataset(dataset_path='./carla_dataset[frame_delay_' + str(n_frame_delay) + '].hdf5', mode='validation',
                                                                train_pretransform=preprocess, train_posttransform=postprocess,
                                                                valid_pretransform=preprocess, valid_posttransform=postprocess,
                                                                test_pretransform=preprocess, test_posttransform=postprocess)
else:
    carla_dataset_generator = carla_dataset_generator(dataset_save_path='./carla_dataset.hdf5',
                                                      dataset_path='/home/luwis/ICSL_Project/CARLA_NN/CARLA_dev_client/PythonAPI/automatic_dataset_collection/Recorded_Image/2021-09-19 19:13:13.616261_with_segmentation',
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

predictive_ControlNet_model = predictive_ControlNet(in_channels=3, bias=True, verbose='low')
predictive_ControlNet_model.to(processor)

learning_rate = 1e-7

# optimizer = optim.SGD(predictive_img_model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
# optimizer = optim.RMSprop(predictive_img_model.parameters(), lr=learning_rate, momentum=0.9)
optimizer = optim.Adam(predictive_img_model.parameters(), lr=learning_rate)
optimizer_control = optim.Adam(predictive_ControlNet_model.parameters(), lr=learning_rate)

criterion = nn.MSELoss()
criterion_control = nn.MSELoss()

training_writer = SummaryWriter(log_dir='./runs/' + start_time + '/predictive_vehicle_control_training')
validation_writer = SummaryWriter(log_dir='./runs/' + start_time + '/predictive_vehicle_control_validation')

for epoch in range(400):

    total_train_loss_list = []
    total_valid_loss_list = []

    total_train_control_loss_list = []
    total_valid_control_loss_list = []

    print('Current State [Training] - [EPOCH : {}]'.format(str(epoch)))
    predictive_img_model.train()
    predictive_ControlNet_model.train()
    for batch_idx, (prev_img, prev_segmented_img, current_img, current_segmented_img, vehicle_control_feature_vector, predictive_vehicle_control_feature_vector) in enumerate(tqdm(training_dataloader)):

        prev_img = prev_img.to(processor).float()
        current_segmented_img = current_segmented_img.to(processor).float()
        vehicle_control_feature_vector = vehicle_control_feature_vector.to(processor).float()
        predictive_vehicle_control_feature_vector = predictive_vehicle_control_feature_vector.to(processor).float()

        # print('prev_img : {}'.format(prev_img.size()))
        # print('current_segmented_img : {}'.format(current_segmented_img.size()))
        # print('vehicle_control_feature_vector : {}'.format(vehicle_control_feature_vector.size()))
        # print('predictive_vehicle_control_feature_vector : {}'.format(predictive_vehicle_control_feature_vector.size()))

        optimizer.zero_grad()
        reconstructed_img = predictive_img_model(prev_img, vehicle_control_feature_vector)
        reconstruction_loss = criterion(reconstructed_img, current_segmented_img)
        reconstruction_loss.backward()
        optimizer.step()

        optimizer_control.zero_grad()
        control_est = predictive_ControlNet_model(current_segmented_img)
        control_est_loss = criterion_control(control_est, predictive_vehicle_control_feature_vector)
        control_est_loss.backward()
        optimizer_control.step()

        total_train_loss_list.append(reconstruction_loss.item())
        total_train_control_loss_list.append(control_est_loss.item())

    training_writer.add_scalar('Next Image Reconstruction Loss [{}]'.format(start_time), np.mean(total_train_loss_list), global_step=epoch)
    training_writer.add_scalar('Predictive Control Loss [{}]'.format(start_time), np.mean(total_train_control_loss_list), global_step=epoch)
    training_writer.add_image('Original Next Image [{}]'.format(start_time), current_segmented_img, global_step=epoch, dataformats='NCHW')
    training_writer.add_image('Next Image Reconstruction Result [{}]'.format(start_time), reconstructed_img, global_step=epoch, dataformats='NCHW')

    print('Current State [Validation] - [EPOCH : {}]'.format(str(epoch)))
    predictive_img_model.eval()
    predictive_ControlNet_model.eval()
    with torch.no_grad():

        for batch_idx, (prev_img, prev_segmented_img, current_img, current_segmented_img, vehicle_control_feature_vector, predictive_vehicle_control_feature_vector) in enumerate(tqdm(validation_dataloader)):

            prev_img = prev_img.to(processor).float()
            current_segmented_img = current_segmented_img.to(processor).float()
            vehicle_control_feature_vector = vehicle_control_feature_vector.to(processor).float()
            predictive_vehicle_control_feature_vector = predictive_vehicle_control_feature_vector.to(processor).float()

            # print('prev_img : {}'.format(prev_img.size()))
            # print('current_segmented_img : {}'.format(current_segmented_img.size()))
            # print('vehicle_control_feature_vector : {}'.format(vehicle_control_feature_vector.size()))

            reconstructed_img = predictive_img_model(prev_img, vehicle_control_feature_vector)
            reconstruction_loss = criterion(reconstructed_img, current_segmented_img)

            control_est = predictive_ControlNet_model(current_segmented_img)
            control_est_loss = criterion_control(control_est, predictive_vehicle_control_feature_vector)
            
            total_valid_loss_list.append(reconstruction_loss.item())
            total_valid_control_loss_list.append(control_est_loss.item())

        validation_writer.add_scalar('Next Image Reconstruction Loss [{}]'.format(start_time), np.mean(total_valid_loss_list), global_step=epoch)
        validation_writer.add_scalar('Predictive Control Loss [{}]'.format(start_time), np.mean(total_valid_control_loss_list), global_step=epoch)
        validation_writer.add_image('Original Next Image [{}]'.format(start_time), current_segmented_img, global_step=epoch, dataformats='NCHW')
        validation_writer.add_image('Next Image Reconstruction Result [{}]'.format(start_time), reconstructed_img, global_step=epoch, dataformats='NCHW')
