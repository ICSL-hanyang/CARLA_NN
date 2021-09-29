import time

import numpy as np
import cv2 as cv
import os

import h5py

import math

import matplotlib.pyplot as plt

import copy
from scipy.spatial.transform import Rotation as R

from tqdm import tqdm

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler

class carla_dataset_generator():

    def __init__(self, dataset_save_path='',
                       dataset_path='', 
                       verbose='low'):

        self.dataset_save_path = dataset_save_path

        self.training_dataset_path = dataset_path + '/training'
        self.validation_dataset_path = dataset_path + '/validation'
        self.test_dataset_path = dataset_path + '/test'

        self.verbosity = verbose    # low, high

        ### Dataset HDF Preparation by Dataset Group Type ##################################################################
        main_file = h5py.File(self.dataset_save_path, 'w')

        train_group = main_file.create_group('training_group')
        train_group.attrs['type'] = 'training'
        train_group.attrs['path'] = self.training_dataset_path

        valid_group = main_file.create_group('validation_group')
        valid_group.attrs['type'] = 'validation'
        valid_group.attrs['path'] = self.validation_dataset_path

        test_group = main_file.create_group('test_group')
        test_group.attrs['type'] = 'test'
        test_group.attrs['path'] = self.test_dataset_path
        ####################################################################################################################

        self.pose_standardization_list = []

        for group in [train_group, valid_group, test_group]:

            prev_img_path_group = main_file.create_group(group.name + '/prev_img_path')
            prev_segmented_img_path_group = main_file.create_group(group.name + '/prev_segmented_img_path')
            prev_hud_data_path_group = main_file.create_group(group.name + '/prev_hud_data_path')

            current_img_path_group = main_file.create_group(group.name + '/current_img_path')
            current_segmented_img_path_group = main_file.create_group(group.name + '/current_segmented_img_path')
            current_hud_data_path_group = main_file.create_group(group.name + '/current_hud_data_path')

            data_idx = 0

            self.local_print(group.attrs['type'])
            self.local_print(group.attrs['path'])

            prev_img_dataset_path = group.attrs['path'] + '/prev_img'
            prev_segmented_img_dataset_path = group.attrs['path'] + '/prev_segmented_img'
            prev_hud_data_dataset_path = group.attrs['path'] + '/prev_hud_data'

            prev_img_name = sorted(os.listdir(prev_img_dataset_path))
            prev_segmented_img_name = sorted(os.listdir(prev_segmented_img_dataset_path))
            prev_hud_data_name = sorted(os.listdir(prev_hud_data_dataset_path))

            self.local_print(prev_img_dataset_path)
            self.local_print(prev_segmented_img_dataset_path)
            self.local_print(prev_hud_data_dataset_path)
            self.local_print(len(prev_img_name))
            self.local_print(len(prev_hud_data_name))
            self.local_print(len(prev_segmented_img_name))

            current_img_dataset_path = group.attrs['path'] + '/current_img'
            current_segmented_img_dataset_path = group.attrs['path'] + '/current_segmented_img'
            current_hud_data_dataset_path = group.attrs['path'] + '/current_hud_data'

            current_img_name = sorted(os.listdir(current_img_dataset_path))
            current_segmented_img_name = sorted(os.listdir(current_segmented_img_dataset_path))
            current_hud_data_name = sorted(os.listdir(current_hud_data_dataset_path))

            self.local_print(current_img_dataset_path)
            self.local_print(current_segmented_img_dataset_path)
            self.local_print(current_hud_data_dataset_path)
            self.local_print(len(current_img_name))
            self.local_print(len(current_hud_data_name))
            self.local_print(len(current_segmented_img_name))
            
            dataset_length = len(prev_img_name)

            self.local_print('[{} dataset length : {}]'.format(group.attrs['type'], dataset_length), level='low')

            for idx in tqdm(range(dataset_length)):
                self.local_print(prev_img_dataset_path + '/' + prev_img_name[idx])
                self.local_print(prev_segmented_img_dataset_path + '/' + prev_segmented_img_name[idx])
                self.local_print(prev_hud_data_dataset_path + '/' + prev_hud_data_name[idx])

                self.local_print(current_img_dataset_path + '/' + current_img_name[idx])
                self.local_print(current_segmented_img_dataset_path + '/' + current_segmented_img_name[idx])
                self.local_print(current_hud_data_dataset_path + '/' + current_hud_data_name[idx])

                prev_img_path_group.create_dataset(name=str(idx).zfill(10), 
                                                   data=[prev_img_dataset_path + '/' + prev_img_name[idx]],
                                                   compression='gzip', compression_opts=9)

                prev_segmented_img_path_group.create_dataset(name=str(idx).zfill(10), 
                                                            data=[prev_segmented_img_dataset_path + '/' + prev_segmented_img_name[idx]],
                                                            compression='gzip', compression_opts=9)

                prev_hud_data_path_group.create_dataset(name=str(idx).zfill(10), 
                                                        data=[prev_hud_data_dataset_path + '/' + prev_hud_data_name[idx]],
                                                        compression='gzip', compression_opts=9)

                current_img_path_group.create_dataset(name=str(idx).zfill(10), 
                                                      data=[current_img_dataset_path + '/' + current_img_name[idx]],
                                                      compression='gzip', compression_opts=9)

                current_segmented_img_path_group.create_dataset(name=str(idx).zfill(10), 
                                                                data=[current_segmented_img_dataset_path + '/' + current_segmented_img_name[idx]],
                                                                compression='gzip', compression_opts=9)
                                                      
                current_hud_data_path_group.create_dataset(name=str(idx).zfill(10), 
                                                           data=[current_hud_data_dataset_path + '/' + current_hud_data_name[idx]],
                                                           compression='gzip', compression_opts=9)

            self.local_print('----------------------------------------------------')

    def local_print(self, sen, level='high'):

        if self.verbosity == 'high':
            print(sen)

        elif self.verbosity == 'low':
            if level == 'low':
                print(sen)
