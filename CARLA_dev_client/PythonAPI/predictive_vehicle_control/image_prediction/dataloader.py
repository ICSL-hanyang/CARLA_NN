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

global_dataloder_print_flag = True
global_tensor_disp_dataloder_print_flag = False

USE_GRAY = False

def global_print(print_str):

    global global_dataloder_print_flag

    if global_dataloder_print_flag == True:
        print(print_str)

def global_tensor_disp(img_tensor):

    global global_tensor_disp_dataloder_print_flag

    if global_tensor_disp_dataloder_print_flag == True:

        if len(img_tensor.size()) == 4:

            seq, channel, height, width = img_tensor.size()
            for disp_idx in range(seq):
                
                if USE_GRAY == True:
                    horizontal_stack = img_tensor[disp_idx, :3, :, :]
                else:                
                    horizontal_stack = torch.cat((img_tensor[disp_idx, :3, :, :], img_tensor[disp_idx, 3:, :, :]), dim=2)
        
                if disp_idx == 0:
                    disp_output = horizontal_stack
                else:
                    disp_output = torch.cat((horizontal_stack, disp_output), dim=1)

            print(disp_output.size())

            cv.imshow('img_tensor', np.array(torchvision.transforms.ToPILImage()(disp_output)))
            cv.waitKey(30)

        elif len(img_tensor.size()) == 3:

            channel, height, width = img_tensor.size()

            cv.imshow('img_tensor', np.array(torchvision.transforms.ToPILImage()(img_tensor)))
            cv.waitKey(30)

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
        prev_img = cv.imread(prev_img_path)

        current_img = cv.imread(current_img_path)

        disp_img = np.concatenate((current_img, prev_img), axis=0)
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
        prev_Speed = prev_hud_data_dict['Speed']
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
        current_Speed = current_hud_data_dict['Speed']
        current_Location = current_hud_data_dict['Location']
        current_Heading = current_hud_data_dict['Heading']
        current_GNSS = current_hud_data_dict['GNSS']
        current_Height = current_hud_data_dict['Height']

        global_print('prev_hud_data_dict keys : {}'.format(list(prev_hud_data_dict.keys())))
        global_print('prev_hud_data_dict : {}'.format(prev_hud_data_dict))

        global_print('current_hud_data_dict keys : {}'.format(list(current_hud_data_dict.keys())))
        global_print('current_hud_data_dict : {}'.format(current_hud_data_dict))
        ######################################################################################################

        global_print('------------------------------------------------------------')

        return 0

    def __len__(self):

        return self.len

