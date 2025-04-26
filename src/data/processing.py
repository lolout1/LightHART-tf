
from scipy.io import loadmat
import glob
import numpy as np
import pandas as pd
import re
import os
from einops import rearrange

def sliding_window(data, clearing_time_index, max_time, sub_window_size, stride_size):
    '''
    Sliding Window
    '''

    assert clearing_time_index >= sub_window_size - \
        1, "Clearing value needs to be greater or equal to (window size - 1)"
    start = clearing_time_index - sub_window_size + 1

    if max_time >= data.shape[0]-sub_window_size:
        max_time = max_time - sub_window_size + 1
        # 2510 // 100 - 1 25 #25999 1000 24000 = 24900

    sub_windows = (
        start +
        np.expand_dims(np.arange(sub_window_size), 0) +
        np.expand_dims(np.arange(max_time, 10), 0).T
    )

    # labels = np.round(np.mean(labels[sub_windows], axis=1))
    return data[sub_windows]


def process_data(raw_data, window_size, stride):
    '''
    Processing Sliding window

    '''

    data = sliding_window(raw_data, window_size - 1,
                          raw_data.shape[0], window_size, stride)
    return data


def sf_processing(data_dir = 'data/smartfallmm', subjects = None,
                    skl_window_size = 32, 
                    num_windows = 10,
                    acc_window_size = 32,
                    num_joints = 32, num_channels = 3):
    '''
    SmartFallMM processing
    '''
    skl_set = []
    acc_set = []
    label_set = []

    file_paths = glob.glob(f'{data_dir}/skeleton/*.csv')
    print("file paths {}".format(len(file_paths)))
    #skl_path = f"{data_dir}/{mode}_skeleton_op/"
    #skl_path = f"{data_dir}/{mode}/skeleton/"
    acc_dir = f"{data_dir}/accelerometer/watch_accelerometer"
    phone_dir = f"{data_dir}/accelerometer/phone_accelerometer"
    pattern = r'S\d+A\d+T\d+'
    act_pattern = r'(A\d+)'
    label_pattern = r'(\d+)'

    for idx,path in enumerate(file_paths):
        desp = re.findall(pattern, file_paths[idx])[0]
        if not int(desp[1:3]) in subjects:
            continue
        act_label = re.findall(act_pattern, path)[0]
        label = int(int(re.findall(label_pattern, act_label)[0])>9)
        acc_path = f'{acc_dir}/{desp}.csv'

        phone_path = f'{phone_dir}/{desp}.csv'

        if os.path.exists(acc_path):
             acc_df = pd.read_csv(acc_path, header = 0).dropna()
        else: 
             continue
        
        # if os.path.exists(phone_path):
        #      phone_df = pd.read_csv(phone_path, header = 0).dropna()
        # else: 
        #      continue

        acc_data = acc_df.bfill().iloc[2:, -3:].to_numpy(dtype=np.float32)
        # phone_data = phone_df.bfill().iloc[2:, -3:].to_numpy(dtype=np.float32)
        
        acc_stride = (acc_data.shape[0] - acc_window_size) // num_windows

       
    
        skl_df  = pd.read_csv(path, index_col =False).dropna()
        skl_data = skl_df.bfill().iloc[:, -96:].to_numpy(dtype=np.float32)
        ######## avg poolin #########
        if  acc_data.shape[0] == 0:   
            os.remove(acc_path)
            continue
        # if phone_data.shape[0] < 10 : 
        #     os.remove(phone_path)
        #     continue

        skl_stride =(skl_data.shape[0] - skl_window_size) // num_windows
        if acc_stride <= 0 or skl_stride <= 0:
            continue

        # processed_acc = process_data(acc_data, acc_window_size, acc_stride)
        # padded_acc = pad_sequence_numpy(sequence=acc_data, input_shape= acc_data.shape, max_sequence_length=acc_window_size)
        # padded_phone = pad_sequence_numpy(sequence=phone_data, input_shape=phone_data.shape, max_sequence_length=acc_window_size)
        # av = calculate_angle(padded_acc, padded_phone)
        # padded_skl = pad_sequence_numpy(sequence=skl_data, input_shape=skl_data.shape, max_sequence_length=skl_window_size)

        # combined_acc = np.concatenate((padded_acc,padded_phone), axis=1)
        
        # skl_data = rearrange(padded_skl, 't (j c) -> t j c' , j = 32, c = 3)
        # acc_set.append(combined_acc)
        # skl_set.append(skl_data)
        # label_set.append(label)
        #skl_data = rearrange(skl_df.values[:, -96:], 't (j c) -> t j c' , j = 32, c = 3)

    #     acc_stride = 10
        processed_acc = process_data(acc_data, acc_window_size, acc_stride)
        processed_skl = process_data(skl_data, skl_window_size, acc_stride)
        processed_skl = np.reshape(processed_skl, (-1, skl_window_size, num_joints, num_channels))
        sync_size = min(processed_acc.shape[0],processed_skl.shape[0])
        skl_set.append(processed_skl[:sync_size, :, : , :])
        acc_set.append(processed_acc[:sync_size, : , :])
        label_set.append(np.repeat(label, sync_size))

    concat_acc = np.concatenate(acc_set, axis = 0)
    concat_skl = np.concatenate(skl_set, axis = 0)
    # #s,w,j,c = concat_skl.shape
    concat_label = np.concatenate(label_set, axis = 0)
     #print(concat_acc.shape[0], concat_label.shape[0])
    # _,count  = np.unique(concat_label, return_counts = True)
    # print(concat_acc.shape)
    # print(concat_skl.shape)
    # #np.savez('/home/bgu9/KD_Multimodal/train.npz' , data = concat_acc, labels = concat_label)
    dataset = { 'acc_data' : concat_acc,
                 'skl_data' : concat_skl, 
                 'labels': concat_label}
    return dataset

def utd_processing(data_dir = 'data/UTD_MAAD', mode = 'val', acc_window_size = 32, skl_window_size = 32, num_windows = 10):
    skl_set = []
    acc_set = []
    label_set = []

    file_paths = glob.glob(f'{data_dir}/{mode}_inertial/*.mat')
    print("file paths {}".format(len(file_paths)))
    #skl_path = f"{data_dir}/{mode}_skeleton_op/"
    skl_path = f"{data_dir}/{mode}_skeleton/"
    pattern = r'a\d+_s\d+_t\d+'
    act_pattern = r'(a\d+)'
    label_pattern = r'(\d+)'
    for idx,path in enumerate(file_paths):
        desp = re.findall(pattern, file_paths[idx])[0]
        act_label = re.findall(act_pattern, path)[0]
        label = int(re.findall(label_pattern, act_label)[0])-1
        # if label in [0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17,18,19,23,24]:
        #     acc_stride = 4
        #     skl_stride = 1
        # else: 
        #     acc_stride = 10
        #     skl_stride = 3
        acc_data = loadmat(path)['d_iner']
        acc_stride = (acc_data.shape[0] - acc_window_size) // num_windows
        acc_data = acc_data[::2, :]
        # print(acc_stride)
        processed_acc = process_data(acc_data, acc_window_size, acc_stride)
        # print(label)
        # print(processed_acc.shape)
        # skl_file = skl_path+desp+'_color_skeleton.npy'
        skl_data = loadmat( skl_path+desp+'_skeleton.mat')['d_skel']
        skl_data = rearrange(skl_data, 'j c t -> t j c')
        
        skl_stride =(skl_data.shape[0] - skl_window_size) // num_windows

        if acc_stride == 0 or skl_stride == 0:
            # print(path)
            continue
        #skl_data = np.squeeze(np.load(skl_file))
        t, j , c = skl_data.shape
        skl_data = rearrange(skl_data, 't j c -> t (j c)')
        processed_skl = process_data(skl_data, skl_window_size, skl_stride)
        skl_data = rearrange(processed_skl, 'n t (j c) -> n t j c', j =j, c =c)
        sync_size = min(skl_data.shape[0],processed_acc.shape[0])
        skl_set.append(skl_data[:sync_size, :, : , :])
        acc_set.append(processed_acc[:sync_size, : , :])
        label_set.append(np.repeat(label, sync_size))

    concat_acc = np.concatenate(acc_set, axis = 0)
    concat_skl = np.concatenate(skl_set, axis = 0)
    concat_label = np.concatenate(label_set, axis = 0)
    _,count  = np.unique(concat_label, return_counts = True)
    dataset = { 'acc_data' : concat_acc,
                'skl_data' : concat_skl, 
                'labels': concat_label}
    
    return dataset
