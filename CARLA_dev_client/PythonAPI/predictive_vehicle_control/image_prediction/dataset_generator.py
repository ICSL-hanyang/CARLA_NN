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
            prev_hud_data_path_group = main_file.create_group(group.name + '/prev_hud_data_path')

            current_img_path_group = main_file.create_group(group.name + '/current_img_path')
            current_hud_data_path_group = main_file.create_group(group.name + '/current_hud_data_path')

            data_idx = 0

            self.local_print(group.attrs['type'])
            self.local_print(group.attrs['path'])

            prev_img_dataset_path = group.attrs['path'] + '/prev_img'
            prev_hud_data_dataset_path = group.attrs['path'] + '/prev_hud_data'

            prev_img_name = sorted(os.listdir(prev_img_dataset_path))
            prev_hud_data_name = sorted(os.listdir(prev_hud_data_dataset_path))

            self.local_print(prev_img_dataset_path)
            self.local_print(prev_hud_data_dataset_path)
            self.local_print(len(prev_img_name))
            self.local_print(len(prev_hud_data_name))

            current_img_dataset_path = group.attrs['path'] + '/current_img'
            current_hud_data_dataset_path = group.attrs['path'] + '/current_hud_data'

            current_img_name = sorted(os.listdir(current_img_dataset_path))
            current_hud_data_name = sorted(os.listdir(current_hud_data_dataset_path))

            self.local_print(current_img_dataset_path)
            self.local_print(current_hud_data_dataset_path)
            self.local_print(len(current_img_name))
            self.local_print(len(current_hud_data_name))
            
            dataset_length = len(prev_img_name)

            self.local_print('[{} dataset length : {}]'.format(group.attrs['type'], dataset_length), level='low')

            for idx in tqdm(range(dataset_length)):
                self.local_print(prev_img_dataset_path + '/' + prev_img_name[idx])
                self.local_print(prev_hud_data_dataset_path + '/' + prev_hud_data_name[idx])
                self.local_print(current_img_dataset_path + '/' + current_img_name[idx])
                self.local_print(current_hud_data_dataset_path + '/' + current_hud_data_name[idx])

                prev_img_path_group.create_dataset(name=str(idx).zfill(10), 
                                                   data=[prev_img_dataset_path + '/' + prev_img_name[idx]],
                                                   compression='gzip', compression_opts=9)

                prev_hud_data_path_group.create_dataset(name=str(idx).zfill(10), 
                                                        data=[prev_hud_data_dataset_path + '/' + prev_hud_data_name[idx]],
                                                        compression='gzip', compression_opts=9)

                current_img_path_group.create_dataset(name=str(idx).zfill(10), 
                                                      data=[current_img_dataset_path + '/' + current_img_name[idx]],
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

    def calculate_max_min(self, mode='min_max'):

        pose_standardization_list = np.array(self.pose_standardization_list)

        ### 6 DOF distribution plotting ###
        main_fig = plt.figure(figsize=[30, 15])

        fig_original = main_fig.add_subplot(241)
        fig_original.boxplot(pose_standardization_list)
        plt.xticks(np.arange(7) + 1, ['dx', 'dy', 'dz', 'droll', 'dpitch', 'dyaw', 'ddist'])
        fig_original.set_title('Training 6DOF distribution (Original)')

        minmaxscaler = MinMaxScaler()
        pose_minmax_list_sklearn = minmaxscaler.fit_transform(pose_standardization_list)
        fig_normalized_sklearn = main_fig.add_subplot(242)
        fig_normalized_sklearn.boxplot(pose_minmax_list_sklearn)
        plt.xticks(np.arange(7) + 1, ['dx', 'dy', 'dz', 'droll', 'dpitch', 'dyaw', 'ddist'])
        fig_normalized_sklearn.set_title('Training 6DOF distribution (Min-Max Normalized / Sklearn)')

        robust_scaler = RobustScaler(unit_variance=True)
        pose_robust_list_sklearn = robust_scaler.fit_transform(pose_standardization_list)
        fig_robust_sklearn = main_fig.add_subplot(243)
        fig_robust_sklearn.boxplot(pose_robust_list_sklearn)
        plt.xticks(np.arange(7) + 1, ['dx', 'dy', 'dz', 'droll', 'dpitch', 'dyaw', 'ddist'])
        fig_robust_sklearn.set_title('Training 6DOF distribution (Robust Scaler / Sklearn)')

        minmaxscaler_neg = MinMaxScaler(feature_range=(-1, 1))
        pose_minmax_neg_list_sklearn = minmaxscaler_neg.fit_transform(pose_standardization_list)
        fig_normalized_neg_sklearn = main_fig.add_subplot(244)
        fig_normalized_neg_sklearn.boxplot(pose_minmax_neg_list_sklearn)
        plt.xticks(np.arange(7) + 1, ['dx', 'dy', 'dz', 'droll', 'dpitch', 'dyaw', 'ddist'])
        fig_normalized_neg_sklearn.set_title('Training 6DOF distribution (Min-Max Normalized (-1 ~ +1) / Sklearn)')

        std_scaler = StandardScaler()
        pose_std_list_sklearn = std_scaler.fit_transform(pose_standardization_list)
        fig_std_sklearn = main_fig.add_subplot(245)
        fig_std_sklearn.boxplot(pose_std_list_sklearn)
        plt.xticks(np.arange(7) + 1, ['dx', 'dy', 'dz', 'droll', 'dpitch', 'dyaw', 'ddist'])
        fig_std_sklearn.set_title('Training 6DOF distribution (Standardization / Sklearn)')

        fig_logit_sklearn = main_fig.add_subplot(246)
        fig_logit_sklearn.boxplot(1 / (1 + np.exp(-1 * pose_standardization_list)))
        plt.xticks(np.arange(7) + 1, ['dx', 'dy', 'dz', 'droll', 'dpitch', 'dyaw', 'ddist'])
        fig_logit_sklearn.set_title('Training 6DOF distribution (Logistic Transformation / Sklearn)')

        gt_correaltion_matrix = np.corrcoef(pose_standardization_list.T)
        fig_correlation_matrix = main_fig.add_subplot(247)
        plt.imshow(gt_correaltion_matrix)
        plt.colorbar(shrink=0.7)
        plt.xticks(np.arange(7), ['dx', 'dy', 'dz', 'droll', 'dpitch', 'dyaw', 'ddist'])
        plt.yticks(np.arange(7), ['dx', 'dy', 'dz', 'droll', 'dpitch', 'dyaw', 'ddist'])
        fig_correlation_matrix.set_title('Training 6DOF Correlation Matrix (Original)')

        gt_correaltion_matrix_minmax = np.corrcoef(pose_minmax_list_sklearn.T)
        fig_correlation_matrix_min_max = main_fig.add_subplot(248)
        plt.imshow(gt_correaltion_matrix_minmax)
        plt.colorbar(shrink=0.7)
        plt.xticks(np.arange(7), ['dx', 'dy', 'dz', 'droll', 'dpitch', 'dyaw', 'ddist'])
        plt.yticks(np.arange(7), ['dx', 'dy', 'dz', 'droll', 'dpitch', 'dyaw', 'ddist'])
        fig_correlation_matrix_min_max.set_title('Training 6DOF Correlation Matrix (Min Max Normalized)')

        plt.savefig('./6DOF Scaling Result')

        if mode == 'min_max': return minmaxscaler
        elif mode == 'robust': return robust_scaler
        elif mode == 'min_max_neg': return minmaxscaler_neg
        elif mode == 'std': return std_scaler
        elif mode == 'logit': return 'logit'