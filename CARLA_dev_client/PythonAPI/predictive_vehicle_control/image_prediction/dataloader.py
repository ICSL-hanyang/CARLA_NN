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

        ### Data Path Preparation ###
        idx = str(index).zfill(10)

        prev_img_path = str((self.prev_img_dataset_path_group[idx][()])[0], 'utf-8')
        prev_hud_data_path = str((self.prev_hud_data_dataset_path_group[idx][()])[0], 'utf-8')

        current_img_path = str((self.current_img_dataset_path_group[idx][()])[0], 'utf-8')
        current_hud_data_path = str((self.current_hud_data_dataset_path_group[idx][()])[0], 'utf-8')


        global_print('prev_img_path : {}'.format(prev_img_path))
        global_print('prev_hud_data_path : {}'.format(prev_hud_data_path))
        global_print('current_img_path : {}'.format(current_img_path))
        global_print('current_hud_data_path : {}'.format(current_hud_data_path))
        global_print('------------------------------------------------------------')

        return 0

    def __len__(self):

        return self.len

