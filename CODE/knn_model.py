from scipy import signal
import numpy as np
import csv
import matplotlib.pyplot as plt
import json
import re
import tensorflow as tf
import keras
from numpy import newaxis
import scipy as sp
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dropout, Dense, Conv2D, MaxPooling2D, Flatten
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy as accuracy
import keras.utils


# DIR
DIR = '/Users/zhengli/Desktop/Projects/MS-thesis/Lab_Patient_Data/'
PATIENT = 'PED2V1_2017_03_13'
RIGHT_FOOT = '/splitData/right/'
LEFT_FOOT = '/splitData/left/'

SAVED_TAG_JSON = 'saves/saved-1510847482.json'
CAP = 'leg_Capacitor_'
ACC = 'leg_Acc_'

TAG_PATH = DIR + PATIENT + RIGHT_FOOT + SAVED_TAG_JSON

# DATA_PATH = [DIR + PATIENT + RIGHT_FOOT + CAP + '1' + '.csv']

# DATA_PATH = [DIR + PATIENT + RIGHT_FOOT + CAP + '1' + '.csv',
#              DIR + PATIENT + RIGHT_FOOT + CAP + '2' + '.csv',
#              DIR + PATIENT + RIGHT_FOOT + CAP + '3' + '.csv']

#
DATA_PATH = [DIR + PATIENT + RIGHT_FOOT + CAP + '1' + '.csv',
             DIR + PATIENT + RIGHT_FOOT + CAP + '2' + '.csv',
             DIR + PATIENT + RIGHT_FOOT + CAP + '3' + '.csv',
             DIR + PATIENT + LEFT_FOOT + CAP + '1' + '.csv',
             DIR + PATIENT + LEFT_FOOT + CAP + '2' + '.csv',
             DIR + PATIENT + LEFT_FOOT + CAP + '3' + '.csv']


# DATA_PATH = [DIR + PATIENT + RIGHT_FOOT + CAP + '1' + '.csv',
#              DIR + PATIENT + RIGHT_FOOT + CAP + '2' + '.csv',
#              DIR + PATIENT + RIGHT_FOOT + CAP + '3' + '.csv',
#              DIR + PATIENT + LEFT_FOOT + CAP + '1' + '.csv',
#              DIR + PATIENT + LEFT_FOOT + CAP + '2' + '.csv',
#              DIR + PATIENT + LEFT_FOOT + CAP + '3' + '.csv',
#              DIR + PATIENT + RIGHT_FOOT + ACC + 'X' + '.csv',
#              DIR + PATIENT + RIGHT_FOOT + ACC + 'Y' + '.csv',
#              DIR + PATIENT + RIGHT_FOOT + ACC + 'Z' + '.csv',
#              DIR + PATIENT + LEFT_FOOT + ACC + 'X' + '.csv',
#              DIR + PATIENT + LEFT_FOOT + ACC + 'Y' + '.csv',
#              DIR + PATIENT + LEFT_FOOT + ACC + 'Z' + '.csv']


# read tags' json file
with open(TAG_PATH) as tag_data:
    tag = json.load(tag_data)
tag_list_with_movement = [d for d in tag if d['tagName'] == 'yes']
tag_times = [f['times'] for f in tag_list_with_movement]
for i in tag_times:
    for j in i:
        j['comment'] = re.sub(r'\d+(.)\s', '', j['comment'])


win_size = 100  # window size
# temp dataset to store the band's csv data
# spectrogram size = 11 * 9
data_from_csv = np.zeros(shape=(8000, win_size), dtype=np.float64)
# ACC_data = np.zeros(shape=(6000, 11, 9))
# Gyro_data = np.zeros(shape=(6000, 11, 9))
label_from_comment = np.zeros(shape=(8000, 1), dtype=object)
counter = 0 #counter for counting how many spectrograms have been generate

for file_num in range(len(DATA_PATH)):
    # print(DATA_PATH[file_num])
    # read split raw data from the band
    print("file num:", file_num)
    with open(DATA_PATH[file_num], 'r', newline='') as raw_data:
        csv_data = csv.reader(raw_data)
        sensor_data_list = list(csv_data)
        colunm1 = [item[1] for item in sensor_data_list]  #rextract data from column 1
        sensor_data_array = np.asarray(sensor_data_list, dtype=np.float64)
        sensor_data_array_column1 = np.asarray(colunm1, dtype=np.float64)
        # print(sensor_data_array_row1)
    START = 0
    STOP = 0
    spec_num = 0
    for i in tag_times:
        for j in i:
            start = int(j['start'])
            stop = int(j['stop'])
            comment = str(j['comment'])
            if comment == ''.join(['n', 'a', 'n']):
                comment = 'complex'
            isFirst = True
            # counter += 1
            # print(counter)
            # find out the start and stop index from the split raw data
            for idx, row in enumerate(sensor_data_array):
                if start < float(row[0]) and isFirst:
                    # print('start:' + str(idx))
                    START = idx
                    isFirst = False
                elif stop + 1 < float(row[0]):
                    # print('stop:' + str(idx))
                    STOP = idx
                    spec_num = int((STOP - START) / win_size)
                    # draw spectrogram
                    if spec_num < 1 :
                        column_array = sensor_data_array_column1[START: (START + win_size)]
                        data_from_csv[counter] = column_array
                        label_from_comment[counter] = comment
                        counter += 1
                    else:
                        column_array = sensor_data_array_column1[START: STOP]
                        num = 0
                        while num < spec_num:
                            column_sub_array = column_array[(num * win_size): (num * win_size + win_size)]
                            # print(column_sub_array.type)
                            data_from_csv[counter] = column_sub_array
                            label_from_comment[counter] = comment
                            counter += 1
                            print(counter)
                            num += 1
                    break


# remove all 0 rows of the data set
mask = ~(data_from_csv == 0).all(axis=(1))
data_from_csv = data_from_csv[mask]

#remove all 0 rows of the label set
mask = ~(label_from_comment == 0).all(axis = (1))
label_from_comment = label_from_comment[mask]

len_of_data = len(data_from_csv)

#
input_data_without_cut = np.zeros(shape=(len_of_data, win_size))
input_label_without_cut = np.zeros(shape=(len_of_data, 1),dtype=object)
data_counter = 0

#find out the 3 type of data that we need: complex, foot jolt and toes wiggling
for idx in range(len(label_from_comment)):
    if np.array_equal(label_from_comment[idx], ['complex']) or np.array_equal(label_from_comment[idx], ['foot jolt']) or np.array_equal(label_from_comment[idx], ['toes wiggling']):
        temp_label = label_from_comment[idx]
        temp_data = data_from_csv[idx]
        input_data_without_cut[data_counter] = temp_data
        input_label_without_cut[data_counter] = temp_label
        data_counter += 1

mask = ~(input_data_without_cut == 0).all(axis=(1))
final_data = input_data_without_cut[mask]

mask = ~(input_label_without_cut == 0).all(axis = (1))
final_label = input_label_without_cut[mask]

len_of_cut_data = len(final_data)

convertedLabel = np.zeros(shape=(len_of_cut_data, 1))
for idx in range(len(final_label)):
    if np.array_equal(final_label[idx], ['complex']):
        convertedLabel[idx] = 0
    elif np.array_equal(final_label[idx], ['foot jolt']):
        convertedLabel[idx] = 1
    elif np.array_equal(final_label[idx], ['toes wiggling']):
        convertedLabel[idx] = 2

# convertedLabel = np.repeat(convertedLabel, 10000, axis=0)
# final_label = np.repeat(final_data, 10000, axis=0)


idx = np.arange(len(convertedLabel))
np.random.shuffle(idx) # in-placeo operation
train_idx = idx[:int(len(idx)*2/3)]
test_idx = idx[int(len(idx)*2/3):]
train_data = final_data[train_idx,:]
train_label = convertedLabel[train_idx]
test_data = final_data[test_idx,:]
test_label = convertedLabel[test_idx]


# train knn model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_data, train_label.ravel())
pred = knn.predict(test_data)
accuracy_knn = accuracy_score(test_label, pred)
print('knn accuracy:', accuracy_knn)


#train LR model
from sklearn.linear_model import LogisticRegression

logClassifier = LogisticRegression()
logClassifier.fit(train_data, train_label.ravel())
pred = logClassifier.predict(test_data)
accuracy_lr = accuracy_score(test_label, pred)
print('lr accuracy:', accuracy_lr)

#train svm
from sklearn import svm
svmClassifier = svm.SVC()
svmClassifier.fit(train_data, train_label.ravel())
pred = svmClassifier.predict(test_data)
accuracy_svm = accuracy_score(test_label, pred)
print('svm accuracy:', accuracy_svm)