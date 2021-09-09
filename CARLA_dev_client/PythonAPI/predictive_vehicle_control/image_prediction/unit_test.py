import torch

import torch.nn as nn
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

dataloader = DataLoader(dataset=predictive_vehicle_control_dataset, batch_size=1, shuffle=False)

for batch_idx, (_) in enumerate(tqdm(dataloader)):

    pass