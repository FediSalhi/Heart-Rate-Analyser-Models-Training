import pandas as pd
import matplotlib.pyplot as plt
import wfdb
import  numpy as np
import random
from scipy.signal import resample
from parameters import *

#################################################### data files ########################################################

#--------------------------- Arrhythmia fetus ---------------------------
path_ARR_1   = "C:\\Users\\Designer\\Desktop\BLM5135_Final_Project\\non-invasive-fetal-ecg-arrhythmia-database-1.0.0\\ARR_01"
path_ARR_2   = "C:\\Users\\Designer\\Desktop\BLM5135_Final_Project\\non-invasive-fetal-ecg-arrhythmia-database-1.0.0\\ARR_02"
path_ARR_3   = "C:\\Users\\Designer\\Desktop\BLM5135_Final_Project\\non-invasive-fetal-ecg-arrhythmia-database-1.0.0\\ARR_03"
path_ARR_4   = "C:\\Users\\Designer\\Desktop\BLM5135_Final_Project\\non-invasive-fetal-ecg-arrhythmia-database-1.0.0\\ARR_04"
path_ARR_5   = "C:\\Users\\Designer\\Desktop\BLM5135_Final_Project\\non-invasive-fetal-ecg-arrhythmia-database-1.0.0\\ARR_05"
path_ARR_6   = "C:\\Users\\Designer\\Desktop\BLM5135_Final_Project\\non-invasive-fetal-ecg-arrhythmia-database-1.0.0\\ARR_06"
path_ARR_7   = "C:\\Users\\Designer\\Desktop\BLM5135_Final_Project\\non-invasive-fetal-ecg-arrhythmia-database-1.0.0\\ARR_07"
path_ARR_8   = "C:\\Users\\Designer\\Desktop\BLM5135_Final_Project\\non-invasive-fetal-ecg-arrhythmia-database-1.0.0\\ARR_08"
path_ARR_9   = "C:\\Users\\Designer\\Desktop\BLM5135_Final_Project\\non-invasive-fetal-ecg-arrhythmia-database-1.0.0\\ARR_09"
path_ARR_10  = "C:\\Users\\Designer\\Desktop\BLM5135_Final_Project\\non-invasive-fetal-ecg-arrhythmia-database-1.0.0\\ARR_10"
path_ARR_11  = "C:\\Users\\Designer\\Desktop\BLM5135_Final_Project\\non-invasive-fetal-ecg-arrhythmia-database-1.0.0\\ARR_11"
path_ARR_12 = "C:\\Users\\Designer\\Desktop\BLM5135_Final_Project\\non-invasive-fetal-ecg-arrhythmia-database-1.0.0\\ARR_12"

#------------------------- Normal rhythm fetus --------------------------

path_NR_1   = "C:\\Users\\Designer\\Desktop\BLM5135_Final_Project\\non-invasive-fetal-ecg-arrhythmia-database-1.0.0\\NR_01"
path_NR_2   = "C:\\Users\\Designer\\Desktop\BLM5135_Final_Project\\non-invasive-fetal-ecg-arrhythmia-database-1.0.0\\NR_02"
path_NR_3   = "C:\\Users\\Designer\\Desktop\BLM5135_Final_Project\\non-invasive-fetal-ecg-arrhythmia-database-1.0.0\\NR_03"
path_NR_4   = "C:\\Users\\Designer\\Desktop\BLM5135_Final_Project\\non-invasive-fetal-ecg-arrhythmia-database-1.0.0\\NR_04"
path_NR_5   = "C:\\Users\\Designer\\Desktop\BLM5135_Final_Project\\non-invasive-fetal-ecg-arrhythmia-database-1.0.0\\NR_05"
path_NR_6   = "C:\\Users\\Designer\\Desktop\BLM5135_Final_Project\\non-invasive-fetal-ecg-arrhythmia-database-1.0.0\\NR_06"
path_NR_7   = "C:\\Users\\Designer\\Desktop\BLM5135_Final_Project\\non-invasive-fetal-ecg-arrhythmia-database-1.0.0\\NR_07"
path_NR_8   = "C:\\Users\\Designer\\Desktop\BLM5135_Final_Project\\non-invasive-fetal-ecg-arrhythmia-database-1.0.0\\NR_08"
path_NR_9   = "C:\\Users\\Designer\\Desktop\BLM5135_Final_Project\\non-invasive-fetal-ecg-arrhythmia-database-1.0.0\\NR_09"
path_NR_10  = "C:\\Users\\Designer\\Desktop\BLM5135_Final_Project\\non-invasive-fetal-ecg-arrhythmia-database-1.0.0\\NR_10"
path_NR_11  = "C:\\Users\\Designer\\Desktop\BLM5135_Final_Project\\non-invasive-fetal-ecg-arrhythmia-database-1.0.0\\NR_11"
path_NR_12 = "C:\\Users\\Designer\\Desktop\BLM5135_Final_Project\\non-invasive-fetal-ecg-arrhythmia-database-1.0.0\\NR_12"

#################################################### reading records ###################################################

#--------------------------- Arrhythmia fetus ---------------------------

record_ARR_1  = np.array(wfdb.rdrecord(path_ARR_1).p_signal[:,0]).reshape(-1,1)
record_ARR_2  = np.array(wfdb.rdrecord(path_ARR_1).p_signal[:,0]).reshape(-1,1)
record_ARR_3  = np.array(wfdb.rdrecord(path_ARR_1).p_signal[:,0]).reshape(-1,1)
record_ARR_4  = np.array(wfdb.rdrecord(path_ARR_1).p_signal[:,0]).reshape(-1,1)
record_ARR_5  = np.array(wfdb.rdrecord(path_ARR_1).p_signal[:,0]).reshape(-1,1)
record_ARR_6  = np.array(wfdb.rdrecord(path_ARR_1).p_signal[:,0]).reshape(-1,1)
record_ARR_7  = np.array(wfdb.rdrecord(path_ARR_1).p_signal[:,0]).reshape(-1,1)
record_ARR_8  = np.array(wfdb.rdrecord(path_ARR_1).p_signal[:,0]).reshape(-1,1)
record_ARR_9  = np.array(wfdb.rdrecord(path_ARR_1).p_signal[:,0]).reshape(-1,1)
record_ARR_10 = np.array(wfdb.rdrecord(path_ARR_1).p_signal[:,0]).reshape(-1,1)
record_ARR_11 = np.array(wfdb.rdrecord(path_ARR_1).p_signal[:,0]).reshape(-1,1)
record_ARR_12 = np.array(wfdb.rdrecord(path_ARR_1).p_signal[:,0]).reshape(-1,1)

#------------------------- Normal rhythm fetus --------------------------

record_NR_1  = np.array(wfdb.rdrecord(path_NR_1).p_signal[:,0]).reshape(-1,1)
record_NR_2  = np.array(wfdb.rdrecord(path_NR_2).p_signal[:,0]).reshape(-1,1)
record_NR_3  = np.array(wfdb.rdrecord(path_NR_3).p_signal[:,0]).reshape(-1,1)
record_NR_4  = np.array(wfdb.rdrecord(path_NR_4).p_signal[:,0]).reshape(-1,1)
record_NR_5  = np.array(wfdb.rdrecord(path_NR_5).p_signal[:,0]).reshape(-1,1)
record_NR_6  = np.array(wfdb.rdrecord(path_NR_6).p_signal[:,0]).reshape(-1,1)
record_NR_7  = np.array(wfdb.rdrecord(path_NR_7).p_signal[:,0]).reshape(-1,1)
record_NR_8  = np.array(wfdb.rdrecord(path_NR_8).p_signal[:,0]).reshape(-1,1)
record_NR_9  = np.array(wfdb.rdrecord(path_NR_9).p_signal[:,0]).reshape(-1,1)
record_NR_10 = np.array(wfdb.rdrecord(path_NR_10).p_signal[:,0]).reshape(-1,1)
record_NR_11 = np.array(wfdb.rdrecord(path_NR_11).p_signal[:,0]).reshape(-1,1)
record_NR_12 = np.array(wfdb.rdrecord(path_NR_12).p_signal[:,0]).reshape(-1,1)

################################################# Adding target column #################################################

#--------------------------- Arrhythmia fetus ---------------------------

record_ARR_1_labeled  = np.ones((record_ARR_1.shape[0], 2))
record_ARR_2_labeled  = np.ones((record_ARR_2.shape[0], 2))
record_ARR_3_labeled  = np.ones((record_ARR_3.shape[0], 2))
record_ARR_4_labeled  = np.ones((record_ARR_4.shape[0], 2))
record_ARR_5_labeled  = np.ones((record_ARR_5.shape[0], 2))
record_ARR_6_labeled  = np.ones((record_ARR_6.shape[0], 2))
record_ARR_7_labeled  = np.ones((record_ARR_7.shape[0], 2))
record_ARR_8_labeled  = np.ones((record_ARR_8.shape[0], 2))
record_ARR_9_labeled  = np.ones((record_ARR_9.shape[0], 2))
record_ARR_10_labeled = np.ones((record_ARR_10.shape[0], 2))
record_ARR_11_labeled = np.ones((record_ARR_11.shape[0], 2))
record_ARR_12_labeled = np.ones((record_ARR_12.shape[0], 2))

record_ARR_1_labeled[:,0]  = record_ARR_1[:,0]
record_ARR_2_labeled[:,0]  = record_ARR_2[:,0]
record_ARR_3_labeled[:,0]  = record_ARR_3[:,0]
record_ARR_4_labeled[:,0]  = record_ARR_4[:,0]
record_ARR_5_labeled[:,0]  = record_ARR_5[:,0]
record_ARR_6_labeled[:,0]  = record_ARR_6[:,0]
record_ARR_7_labeled[:,0]  = record_ARR_7[:,0]
record_ARR_8_labeled[:,0]  = record_ARR_8[:,0]
record_ARR_9_labeled[:,0]  = record_ARR_9[:,0]
record_ARR_10_labeled[:,0] = record_ARR_10[:,0]
record_ARR_11_labeled[:,0] = record_ARR_11[:,0]
record_ARR_12_labeled[:,0] = record_ARR_12[:,0]

#------------------------- Normal rhythm fetus --------------------------

record_NR_1_labeled  = np.zeros((record_NR_1.shape[0], 2))
record_NR_2_labeled  = np.zeros((record_NR_2.shape[0], 2))
record_NR_3_labeled  = np.zeros((record_NR_3.shape[0], 2))
record_NR_4_labeled  = np.zeros((record_NR_4.shape[0], 2))
record_NR_5_labeled  = np.zeros((record_NR_5.shape[0], 2))
record_NR_6_labeled  = np.zeros((record_NR_6.shape[0], 2))
record_NR_7_labeled  = np.zeros((record_NR_7.shape[0], 2))
record_NR_8_labeled  = np.zeros((record_NR_8.shape[0], 2))
record_NR_9_labeled  = np.zeros((record_NR_9.shape[0], 2))
record_NR_10_labeled = np.zeros((record_NR_10.shape[0], 2))
record_NR_11_labeled = np.zeros((record_NR_11.shape[0], 2))
record_NR_12_labeled = np.zeros((record_NR_12.shape[0], 2))

record_NR_1_labeled[:,0]  = record_NR_1[:,0]
record_NR_2_labeled[:,0]  = record_NR_2[:,0]
record_NR_3_labeled[:,0]  = record_NR_3[:,0]
record_NR_4_labeled[:,0]  = record_NR_4[:,0]
record_NR_5_labeled[:,0]  = record_NR_5[:,0]
record_NR_6_labeled[:,0]  = record_NR_6[:,0]
record_NR_7_labeled[:,0]  = record_NR_7[:,0]
record_NR_8_labeled[:,0]  = record_NR_8[:,0]
record_NR_9_labeled[:,0]  = record_NR_9[:,0]
record_NR_10_labeled[:,0] = record_NR_10[:,0]
record_NR_11_labeled[:,0] = record_NR_11[:,0]
record_NR_12_labeled[:,0] = record_NR_12[:,0]

##################################################### Combine data #####################################################

#--------------------------- Arrhythmia fetus ---------------------------

all_records_ARR_labeled = np.vstack((record_ARR_1_labeled,record_ARR_2_labeled))
all_records_ARR_labeled = np.vstack((all_records_ARR_labeled,record_ARR_3_labeled))
all_records_ARR_labeled = np.vstack((all_records_ARR_labeled,record_ARR_4_labeled))
all_records_ARR_labeled = np.vstack((all_records_ARR_labeled,record_ARR_5_labeled))
all_records_ARR_labeled = np.vstack((all_records_ARR_labeled,record_ARR_6_labeled))
all_records_ARR_labeled = np.vstack((all_records_ARR_labeled,record_ARR_7_labeled))
all_records_ARR_labeled = np.vstack((all_records_ARR_labeled,record_ARR_8_labeled))
all_records_ARR_labeled = np.vstack((all_records_ARR_labeled,record_ARR_9_labeled))
all_records_ARR_labeled = np.vstack((all_records_ARR_labeled,record_ARR_10_labeled))
all_records_ARR_labeled = np.vstack((all_records_ARR_labeled,record_ARR_11_labeled))
all_records_ARR_labeled = np.vstack((all_records_ARR_labeled,record_ARR_12_labeled))

all_records_ARR_labeled_reduced = all_records_ARR_labeled[:7200000,:]

#------------------------- Normal rhythm fetus --------------------------

all_records_NR_labeled = np.vstack((record_NR_1_labeled,record_NR_2_labeled))
all_records_NR_labeled = np.vstack((all_records_NR_labeled,record_NR_3_labeled))
all_records_NR_labeled = np.vstack((all_records_NR_labeled,record_NR_4_labeled))
all_records_NR_labeled = np.vstack((all_records_NR_labeled,record_NR_5_labeled))
all_records_NR_labeled = np.vstack((all_records_NR_labeled,record_NR_6_labeled))
all_records_NR_labeled = np.vstack((all_records_NR_labeled,record_NR_7_labeled))
all_records_NR_labeled = np.vstack((all_records_NR_labeled,record_NR_8_labeled))
all_records_NR_labeled = np.vstack((all_records_NR_labeled,record_NR_9_labeled))
all_records_NR_labeled = np.vstack((all_records_NR_labeled,record_NR_10_labeled))
all_records_NR_labeled = np.vstack((all_records_NR_labeled,record_NR_11_labeled))
all_records_NR_labeled = np.vstack((all_records_NR_labeled,record_NR_12_labeled))

all_records_NR_labeled_reduced = all_records_NR_labeled[:7200000,:]


################################################# shuffl #################################################

#TODO: resample data

# np.savetxt("foo.csv", record_NR_1, delimiter=",")


def batch_generator(time_window_size, labelled_data_x, labelled_data_y, batch_size):
    #TODO: to be implemented
    x_batch = np.zeros((batch_size, time_window_size,1))
    y_batch = np.zeros((batch_size,1))

    signal_index = 0
    start_idx = signal_index

    while (True):

        for batch_index in range(batch_size):

            x_batch[batch_index] = labelled_data_x[start_idx:start_idx + time_window_size].reshape(-1,1)
            y_batch[batch_index] = labelled_data_y[start_idx + time_window_size]
            start_idx += time_window_size
            if (start_idx >=  labelled_data_x.shape[0] - time_window_size):
                start_idx = 0

        yield x_batch, y_batch

def split_data(labelled_data_x, labelled_data_y, val_rate, test_rate):

    number_of_data_point = labelled_data_y.shape[0]
    train_rate = 1 - (val_rate + test_rate)

    y_train = labelled_data_y[:int(number_of_data_point * train_rate)]
    x_train = labelled_data_x[:int(number_of_data_point * train_rate)]

    y_val = labelled_data_y[int(number_of_data_point * train_rate) : int(number_of_data_point * (train_rate + val_rate))]
    x_val = labelled_data_x[int(number_of_data_point * train_rate) : int(number_of_data_point * (train_rate + val_rate))]

    y_test = labelled_data_y[int(number_of_data_point * (train_rate + val_rate)) : number_of_data_point ]
    x_test = labelled_data_x[int(number_of_data_point * (train_rate + val_rate)) : number_of_data_point ]

    return x_train.reshape(-1,1), y_train.reshape(-1,1), x_val.reshape(-1,1), y_val.reshape(-1,1), x_test.reshape(-1,1), y_test.reshape(-1,1)

def combine_shufle_signal(all_records_NR_labeled_reduced, all_records_ARR_labeled_reduced):
    selector = 1
    all_records_NR_ARR_labeled_shuffled = np.zeros((7200000,2))
    for i in range(0,7200000-1000,1000):
        if (selector == 1):
            all_records_NR_ARR_labeled_shuffled[i:i+1000,:] = all_records_NR_labeled_reduced[i:i+1000,:]
            selector = 0
        else:
            all_records_NR_ARR_labeled_shuffled[i:i+1000,:] = all_records_ARR_labeled_reduced[i:i+1000,:]
            selector = 1
    return all_records_NR_ARR_labeled_shuffled