import os

import numpy as np
from numpy.core.shape_base import stack

import cv2 as cv

import h5py

import torch
import torch.utils.data
import torchvision.transforms.functional as TF
import torchvision
from torchvision import transforms

import json
import ast

import PIL
from PIL import Image

global_dataloder_print_flag = False
global_input_img_disp_flag = False

USE_GRAY = False

def global_print(print_str):

    global global_dataloder_print_flag

    if global_dataloder_print_flag == True:
        print(print_str)

class carla_dataset(torch.utils.data.Dataset):

    def __init__(self, dataset_path='', mode='training',
                train_pretransform=transforms.Compose([]), train_posttransform=transforms.Compose([]),
                valid_pretransform=transforms.Compose([]), valid_posttransform=transforms.Compose([]),
                test_pretransform=transforms.Compose([]), test_posttransform=transforms.Compose([]),
                scaler=None):

        self.dataset_path = dataset_path

        self.dataset_file = h5py.File(dataset_path, 'r')

        self.mode = mode

        if mode == 'training':
            self.prev_img_dataset_path_group = self.dataset_file['/training_group/prev_img_path']
            self.prev_hud_data_dataset_path_group = self.dataset_file['/training_group/prev_hud_data_path']

            self.current_img_dataset_path_group = self.dataset_file['/training_group/current_img_path']
            self.current_hud_data_dataset_path_group = self.dataset_file['/training_group/current_hud_data_path']

        elif mode == 'validation':
            self.prev_img_dataset_path_group = self.dataset_file['/validation_group/prev_img_path']
            self.prev_hud_data_dataset_path_group = self.dataset_file['/validation_group/prev_hud_data_path']

            self.current_img_dataset_path_group = self.dataset_file['/validation_group/current_img_path']
            self.current_hud_data_dataset_path_group = self.dataset_file['/validation_group/current_hud_data_path']

        elif mode == 'test':
            self.prev_img_dataset_path_group = self.dataset_file['/test_group/prev_img_path']
            self.prev_hud_data_dataset_path_group = self.dataset_file['/test_group/prev_hud_data_path']

            self.current_img_dataset_path_group = self.dataset_file['/test_group/current_img_path']
            self.current_hud_data_dataset_path_group = self.dataset_file['/test_group/current_hud_data_path']

        self.len = self.prev_img_dataset_path_group.__len__()

        self.scaler = scaler

        print('[Dataset Status]')
        print('dataset_path : {}'.format(self.dataset_path))
        print('dataset_file : {}'.format(self.dataset_file))
        print('dataset_mode : {}'.format(self.mode))
        print('dataset_length : {}'.format(self.len), end='\n\n')

    def __getitem__(self, index):

        ### Data Path Preparation ############################################################################
        idx = str(index).zfill(10)

        prev_img_path = str((self.prev_img_dataset_path_group[idx][()])[0], 'utf-8')
        prev_hud_data_path = str((self.prev_hud_data_dataset_path_group[idx][()])[0], 'utf-8')

        current_img_path = str((self.current_img_dataset_path_group[idx][()])[0], 'utf-8')
        current_hud_data_path = str((self.current_hud_data_dataset_path_group[idx][()])[0], 'utf-8')

        global_print('prev_img_path : {}'.format(prev_img_path))
        global_print('prev_hud_data_path : {}'.format(prev_hud_data_path))
        global_print('current_img_path : {}'.format(current_img_path))
        global_print('current_hud_data_path : {}'.format(current_hud_data_path))
        ######################################################################################################

        ### Image Data Loading ###############################################################################
        height = 720
        width = 1280

        prev_img = cv.imread(prev_img_path)
        prev_img = prev_img[(height * 3)//4 : height, width//10 : (width * 9)//10, ...]
        prev_img = cv.resize(prev_img, (120, 120))
        prev_img = np.transpose(prev_img, (2, 0, 1))
        prev_img = prev_img.astype(np.float)
        prev_img /= 255.0

        current_img = cv.imread(current_img_path)
        current_img = current_img[(height * 3)//4 : height, width//10 : (width * 9)//10, ...]
        current_img = cv.resize(current_img, (120, 120))
        current_img = np.transpose(current_img, (2, 0, 1))
        current_img = current_img.astype(np.float)
        current_img /= 255.0

        if global_input_img_disp_flag == True:
            disp_img = np.concatenate((current_img, prev_img), axis=1)
            disp_img = np.transpose(disp_img, (1, 2, 0))
            disp_img = 255.0 * disp_img
            disp_img = disp_img.astype(np.uint8)
            disp_img = cv.resize(disp_img, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_LINEAR)
            cv.imshow('[Up : current_img][Down : prev_img]', disp_img)
            cv.waitKey(30)
        ######################################################################################################

        ### HUD Data Loading #################################################################################
        with open(prev_hud_data_path) as f:
            prev_hud_data = f.read()
        prev_hud_data_dict = ast.literal_eval(prev_hud_data)

        with open(current_hud_data_path) as f:
            current_hud_data = f.read()
        current_hud_data_dict = ast.literal_eval(current_hud_data)

        ### Prev Vehicle Controls ###
        prev_Throttle = prev_hud_data_dict['Throttle']
        prev_Steer = prev_hud_data_dict['Steer']
        prev_Brake = prev_hud_data_dict['Brake']
        prev_Gear = prev_hud_data_dict['Gear']

        ### Prev Vehicle Driving Status ###
        prev_Speed = prev_hud_data_dict['Speed']                # Assume max vehicle speed as 100km/h
        prev_Location = prev_hud_data_dict['Location']
        prev_Heading = prev_hud_data_dict['Heading']
        prev_GNSS = prev_hud_data_dict['GNSS']
        prev_Height = prev_hud_data_dict['Height']

        ### Current Vehicle Controls ###
        current_Throttle = current_hud_data_dict['Throttle']
        current_Steer = current_hud_data_dict['Steer']
        current_Brake = current_hud_data_dict['Brake']
        current_Gear = current_hud_data_dict['Gear']

        ### Current Vehicle Driving Status ###
        current_Speed = current_hud_data_dict['Speed']          # Assume max vehicle speed as 100km/h
        current_Location = current_hud_data_dict['Location']
        current_Heading = current_hud_data_dict['Heading']
        current_GNSS = current_hud_data_dict['GNSS']
        current_Height = current_hud_data_dict['Height']

        ### Vehicle Control Feature Vector Assembly ###
        normalized_prev_Speed = prev_Speed / 100.0     # Normalize vehicle speed assuming that max vehicle speed as 100km/h
        vehicle_control_feature_vector = np.array([prev_Throttle, prev_Steer, normalized_prev_Speed, prev_Brake])
        vehicle_control_feature_vector.astype(np.float)

        global_print('prev_hud_data_dict keys : {}'.format(list(prev_hud_data_dict.keys())))
        global_print('prev_hud_data_dict : {}'.format(prev_hud_data_dict))

        global_print('current_hud_data_dict keys : {}'.format(list(current_hud_data_dict.keys())))
        global_print('current_hud_data_dict : {}'.format(current_hud_data_dict))
        ######################################################################################################

        global_print('------------------------------------------------------------')

        return prev_img, current_img, vehicle_control_feature_vector

    def __len__(self):

        return self.len

