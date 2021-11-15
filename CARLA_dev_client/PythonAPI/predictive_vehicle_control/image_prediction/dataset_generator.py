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

        self.mean_timestamp_difference_length = 0   # Timestamp difference between current and next lane image from Training Dataset

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

            timestamp_difference_length_list = []

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

                if group.attrs['type'] == 'training':
                    prev_time_val_idx0 = prev_img_name[idx].find('t0_') + 3
                    prev_time_val_idx1 = prev_img_name[idx].find('.jpeg')
                    prev_time_val = float(prev_img_name[idx][prev_time_val_idx0 : prev_time_val_idx1])
                    self.local_print('Prev Timestamp : {}'.format(prev_time_val))

                    current_time_val_idx0 = current_img_name[idx].find('t1_') + 3
                    current_time_val_idx1 = current_img_name[idx].find('.jpeg')
                    current_time_val = float(current_img_name[idx][current_time_val_idx0 : current_time_val_idx1])
                    self.local_print('Current Timestamp : {}'.format(current_time_val))

                    timestamp_difference_length = current_time_val - prev_time_val
                    self.local_print('Timestamp Difference Length : {} sec'.format(timestamp_difference_length))

                    timestamp_difference_length_list.append(timestamp_difference_length)

                self.local_print('----------------------------------------------------')

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

            if group.attrs['type'] == 'training':
                self.mean_timestamp_difference_length = np.mean(timestamp_difference_length_list)
                print('mean_timestamp_difference_length : {} sec'.format(self.mean_timestamp_difference_length, level='low'))

            self.local_print('----------------------------------------------------')

    def local_print(self, sen, level='high'):

        if self.verbosity == 'high':
            print(sen)

        elif self.verbosity == 'low':
            if level == 'low':
                print(sen)

class N_frame_delay_carla_dataset_generator():

    def __init__(self, dataset_save_path='',
                       dataset_path='', 
                       n_frame_delay=1,
                       verbose='low'):

        self.dataset_save_path = dataset_save_path

        self.training_dataset_path = dataset_path + '/training'
        self.validation_dataset_path = dataset_path + '/validation'
        self.test_dataset_path = dataset_path + '/test'

        self.n_frame_delay = n_frame_delay

        self.verbosity = verbose    # low, high

        ### Exception Handling for Driving Scenario Jump ###################################################################
        ### Data Timestamps for identifying Driving Scenario Jumps #########################################################
        self.seq_jump_idx = ['19:13:13.616261_' + str(654).zfill(10), '19:13:13.616261_' + str(959).zfill(10), '19:13:13.616261_' + str(654).zfill(1721),
                             '19:13:13.616261_' + str(2037).zfill(10), '19:13:13.616261_' + str(2103).zfill(10),

                             '19:17:22.433945_' + str(1).zfill(10),

                             '19:18:32.960168_' + str(1).zfill(10), '19:18:32.960168_' + str(369).zfill(10), '19:18:32.960168_' + str(1223).zfill(10),
                             '19:18:32.960168_' + str(1670).zfill(10),
                             
                             '19:29:29.398738_' + str(1).zfill(10), '19:29:29.398738_' + str(852).zfill(10), '19:29:29.398738_' + str(999).zfill(10),
                             '19:29:29.398738_' + str(1572).zfill(10), '19:29:29.398738_' + str(2158).zfill(10), '19:29:29.398738_' + str(2224).zfill(10),
                             '19:29:29.398738_' + str(2550).zfill(10), '19:29:29.398738_' + str(2688).zfill(10), '19:29:29.398738_' + str(2795).zfill(10),
                             '19:29:29.398738_' + str(3487).zfill(10), '19:29:29.398738_' + str(3947).zfill(10), '19:29:29.398738_' + str(4118).zfill(10),
                             
                             '19:38:15.339723_' + str(1).zfill(10), '19:38:15.339723_' + str(391).zfill(10), '19:38:15.339723_' + str(639).zfill(10),
                             '19:38:15.339723_' + str(811).zfill(10), '19:38:15.339723_' + str(889).zfill(10), '19:38:15.339723_' + str(991).zfill(10),
                             '19:38:15.339723_' + str(1084).zfill(10), '19:38:15.339723_' + str(1274).zfill(10),
                             
                             '19:45:42.153549_' + str(300).zfill(10), '19:45:42.153549_' + str(597).zfill(10),
                             
                             '19:54:34.722789_' + str(1059).zfill(10), '19:54:34.722789_' + str(1712).zfill(10), '19:54:34.722789_' + str(1784).zfill(10),
                             '19:54:34.722789_' + str(3762).zfill(10), '19:54:34.722789_' + str(4442).zfill(10), '19:54:34.722789_' + str(5040).zfill(10),]
        
        exception_jump_skip_flag = False

        skip_counter = 0
        ####################################################################################################################

        ### Dataset HDF Preparation by Dataset Group Type ##################################################################
        main_file = h5py.File(self.dataset_save_path[:-5] + '[frame_delay_' + str(n_frame_delay) + ']' + '.hdf5', 'w')

        train_group = main_file.create_group('training_group')
        train_group.attrs['type'] = 'training'
        train_group.attrs['n_frame_delay'] = 'n_frame_delay'
        train_group.attrs['path'] = self.training_dataset_path

        valid_group = main_file.create_group('validation_group')
        valid_group.attrs['type'] = 'validation'
        valid_group.attrs['n_frame_delay'] = 'n_frame_delay'
        valid_group.attrs['path'] = self.validation_dataset_path

        test_group = main_file.create_group('test_group')
        test_group.attrs['type'] = 'test'
        test_group.attrs['n_frame_delay'] = 'n_frame_delay'
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

            timestamp_difference_length_list = []

            self.local_print(group.attrs['type'])
            self.local_print(group.attrs['path'])

            img_dataset_path = group.attrs['path'] + '/current_img'
            segmented_img_dataset_path = group.attrs['path'] + '/current_segmented_img'
            hud_data_dataset_path = group.attrs['path'] + '/current_hud_data'

            img_name = sorted(os.listdir(img_dataset_path))
            segmented_img_name = sorted(os.listdir(segmented_img_dataset_path))
            hud_data_name = sorted(os.listdir(hud_data_dataset_path))

            self.local_print(img_dataset_path)
            self.local_print(segmented_img_dataset_path)
            self.local_print(hud_data_dataset_path)
            self.local_print(len(img_name))
            self.local_print(len(segmented_img_name))
            self.local_print(len(hud_data_name))
            
            dataset_length = len(img_name)

            self.local_print('[{} dataset length : {}]'.format(group.attrs['type'], dataset_length), level='low')

            ### Exception Handling for Driving Scenario Jump ###################################################################
            # Reset flag variables for exception handling
            exception_jump_skip_flag = False
            skip_counter = 0
            ####################################################################################################################

            for idx in tqdm(range(dataset_length)):

                if idx < self.n_frame_delay:
                    self.local_print('\n[Skip][Seq {} - idx {}] Current idx lower than n_frame_delay'.format(group.attrs['n_frame_delay'], idx), level='low')
                    pass

                else:

                    self.local_print('prev_img_dataset_path : {}'.format(img_dataset_path + '/' + img_name[idx - self.n_frame_delay]))
                    self.local_print('prev_segmented_img_dataset_path : {}'.format(segmented_img_dataset_path + '/' + segmented_img_name[idx - self.n_frame_delay]))
                    self.local_print('prev_hud_data_dataset_path : {}'.format(hud_data_dataset_path + '/' + hud_data_name[idx - self.n_frame_delay]))

                    self.local_print('current_img_dataset_path : {}'.format(img_dataset_path + '/' + img_name[idx]))
                    self.local_print('current_segmented_img_dataset_path : {}'.format(segmented_img_dataset_path + '/' + segmented_img_name[idx]))
                    self.local_print('current_hud_data_dataset_path : {}'.format(hud_data_dataset_path + '/' + hud_data_name[idx]))


                    ### Exception Handling for Driving Scenario Jump ###################################################################
                    # Skip for N frame delay steps when encountered with Driving Scenario Jump timestamps ##############################
                    for jump_skip_val in self.seq_jump_idx:
                        if jump_skip_val in (img_dataset_path + '/' + img_name[idx]):
                            self.local_print('[Skip] Scenario Jump Detected : {}'.format(jump_skip_val), level='low')
                            exception_jump_skip_flag = True

                    if skip_counter == (self.n_frame_delay + 1):
                        self.local_print('[Skip_counter Reset]', level='low')
                        exception_jump_skip_flag = False
                        skip_counter = 0

                    if exception_jump_skip_flag == True:

                            skip_counter += 1
                    ####################################################################################################################

                    elif exception_jump_skip_flag == False:

                        if group.attrs['type'] == 'training':

                            prev_time_val_idx0 = img_name[idx - self.n_frame_delay].find('t1_') + 3
                            prev_time_val_idx1 = img_name[idx - self.n_frame_delay].find('.jpeg')
                            prev_time_val = float(img_name[idx - self.n_frame_delay][prev_time_val_idx0 : prev_time_val_idx1])
                            self.local_print('Prev Timestamp : {}'.format(prev_time_val))

                            current_time_val_idx0 = img_name[idx].find('t1_') + 3
                            current_time_val_idx1 = img_name[idx].find('.jpeg')
                            current_time_val = float(img_name[idx][current_time_val_idx0 : current_time_val_idx1])
                            self.local_print('Current Timestamp : {}'.format(current_time_val))

                            timestamp_difference_length = current_time_val - prev_time_val
                            self.local_print('Timestamp Difference Length : {} sec'.format(timestamp_difference_length))

                            timestamp_difference_length_list.append(timestamp_difference_length)
                            
                        prev_img_path_group.create_dataset(name=str(data_idx).zfill(10), 
                                                        data=[img_dataset_path + '/' + img_name[idx - self.n_frame_delay]],
                                                        compression='gzip', compression_opts=9)

                        prev_segmented_img_path_group.create_dataset(name=str(data_idx).zfill(10), 
                                                                    data=[segmented_img_dataset_path + '/' + segmented_img_name[idx - self.n_frame_delay]],
                                                                    compression='gzip', compression_opts=9)

                        prev_hud_data_path_group.create_dataset(name=str(data_idx).zfill(10), 
                                                                data=[hud_data_dataset_path + '/' + hud_data_name[idx - self.n_frame_delay]],
                                                                compression='gzip', compression_opts=9)

                        current_img_path_group.create_dataset(name=str(data_idx).zfill(10), 
                                                            data=[img_dataset_path + '/' + img_name[idx]],
                                                            compression='gzip', compression_opts=9)

                        current_segmented_img_path_group.create_dataset(name=str(data_idx).zfill(10), 
                                                                        data=[segmented_img_dataset_path + '/' + segmented_img_name[idx]],
                                                                        compression='gzip', compression_opts=9)
                                                            
                        current_hud_data_path_group.create_dataset(name=str(data_idx).zfill(10), 
                                                                data=[hud_data_dataset_path + '/' + hud_data_name[idx]],
                                                                compression='gzip', compression_opts=9)

                        data_idx += 1

                        self.local_print('----------------------------------------------------')

            if group.attrs['type'] == 'training':
                self.mean_timestamp_difference_length = np.mean(timestamp_difference_length_list)
                print('mean_timestamp_difference_length : {} sec'.format(self.mean_timestamp_difference_length, level='low'))


    def local_print(self, sen, level='high'):

        if self.verbosity == 'high':
            print(sen)

        elif self.verbosity == 'low':
            if level == 'low':
                print(sen)
