from scipy import signal
import numpy as np
import csv

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
# DATA_PATH = [DIR + PATIENT + RIGHT_FOOT + CAP + '1' + '.csv',
#              DIR + PATIENT + RIGHT_FOOT + CAP + '2' + '.csv',
#              DIR + PATIENT + RIGHT_FOOT + CAP + '3' + '.csv']
DATA_PATH = [DIR + PATIENT + RIGHT_FOOT + CAP + '1' + '.csv',
             DIR + PATIENT + RIGHT_FOOT + CAP + '2' + '.csv',
             DIR + PATIENT + RIGHT_FOOT + CAP + '3' + '.csv',
             DIR + PATIENT + LEFT_FOOT + CAP + '1' + '.csv',
             DIR + PATIENT + LEFT_FOOT + CAP + '2' + '.csv',
             DIR + PATIENT + LEFT_FOOT + CAP + '3' + '.csv',
             DIR + PATIENT + RIGHT_FOOT + ACC + 'X' + '.csv',
             DIR + PATIENT + RIGHT_FOOT + ACC + 'Y' + '.csv',
             DIR + PATIENT + RIGHT_FOOT + ACC + 'Z' + '.csv',
             DIR + PATIENT + LEFT_FOOT + ACC + 'X' + '.csv',
             DIR + PATIENT + LEFT_FOOT + ACC + 'Y' + '.csv',
             DIR + PATIENT + LEFT_FOOT + ACC + 'Z' + '.csv']


# read tags' json file
with open(TAG_PATH) as tag_data:
    tag = json.load(tag_data)
tag_list_with_movement = [d for d in tag if d['tagName'] == 'yes']
tag_times = [f['times'] for f in tag_list_with_movement]
for i in tag_times:
    for j in i:
        j['comment'] = re.sub(r'\d+(.)\s', '', j['comment'])

# temp dataset to store the band's csv data
# spectrogram size = 11 * 9
CAP_data = np.zeros(shape=(12000, 11, 9))
# ACC_data = np.zeros(shape=(6000, 11, 9))
# Gyro_data = np.zeros(shape=(6000, 11, 9))
CNN_label = np.zeros(shape=(12000, 1), dtype=object)
cap_counter = 0 #counter for counting how many spectrograms have been generate
acc_counter = 0
gyro_counter = 0

for file_num in range(len(DATA_PATH)):
    # print(DATA_PATH[file_num])
    # read split raw data from the band
    print("file num:", file_num)
    with open(DATA_PATH[file_num], 'r', newline='') as raw_data:
        csv_data = csv.reader(raw_data)
        sensor_data_list = list(csv_data)
        temp = [item[1] for item in sensor_data_list]
        sensor_data_array = np.asarray(sensor_data_list, dtype=np.float64)
        sensor_data_array_row1 = np.asarray(temp, dtype=np.float64)
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
                    spec_num = int((STOP - START) / 100)
                    # draw spectrogram
                    if spec_num < 1 :
                        graphic_array = sensor_data_array_row1[START: STOP]
                        f, t, Sxx = signal.spectrogram(graphic_array, fs=50, nperseg=20)
                        if file_num < 6:
                            CAP_data[cap_counter] = Sxx
                            CNN_label[cap_counter] = comment
                            cap_counter += 1
                        elif file_num >= 6:
                            ACC_data[cap_counter] = Sxx
                            acc_counter += 1
                        # fig = plt.pcolormesh(t, f, Sxx)
                        # plt.colorbar(fig)
                        # plt.ylabel('Frequency [Hz]')
                        # plt.xlabel('Time [sec]')
                        # save_path = '/Users/zhengli/Desktop/Projects/MS-thesis/Lab_Patient_Data/PED2V1_2017_03_13/CNN_spectrogram/'+ comment + str(counter) + '.png'
                        # plt.savefig(save_path)
                        # plt.clf()
                    else:
                        graphic_array = sensor_data_array_row1[START: STOP]
                        num = 0
                        while num < spec_num:
                            graphic_sub_array = graphic_array[(num*100): (num*100+100)]
                            f, t, Sxx = signal.spectrogram(graphic_sub_array, fs=50, nperseg=20, noverlap=10)
                            # if file_num < 6:
                            CAP_data[cap_counter] = Sxx
                            CNN_label[cap_counter] = comment
                            cap_counter += 1
                            print(cap_counter)
                            # elif file_num >= 6:
                            #     ACC_data[cap_counter] = Sxx
                            #     acc_counter += 1
                            # print(Sxx)
                            # fig = plt.pcolormesh(t, f, Sxx)
                            # plt.colorbar(fig)
                            # plt.ylabel('Frequency [Hz]')
                            # plt.xlabel('Time [sec]')
                            # plt.title(comment)
                            # save_path = '/Users/zhengli/Desktop/Projects/MS-thesis/Lab_Patient_Data/PED2V1_2017_03_13/CNN_spectrogram/Right/Capacitor_3/' + comment + str(counter) + '_' + str(num) + '.png'
                            # print(save_path)
                            # plt.savefig(save_path)
                            # plt.clf()
                            num += 1
                    break


# remove all 0 rows of the data set
mask = ~(CAP_data == 0).all(axis=(1,2))
CAP_data = CAP_data[mask]
CNN_data = np.reshape(CAP_data, (995, 12, 11, 9), order='F')

#remove all 0 rows of the label set
mask = ~(CNN_label == 0).all(axis = (1))
CNN_label = CNN_label[mask]
CNN_label = np.delete(CNN_label, np.arange(995, 12000), axis=0)

#
input_data = np.zeros(shape=(995,12,11,9))
input_label = np.zeros(shape=(995,1),dtype=object)
data_counter = 0

#find out the 3 type of data that we need: complex, foot jolt and toes wiggling
for idx in range(len(CNN_label)):
    if np.array_equal(CNN_label[idx], ['complex']) or np.array_equal(CNN_label[idx], ['foot jolt']) or np.array_equal(CNN_label[idx], ['toes wiggling']):
        temp_label = CNN_label[idx]
        temp_data = CNN_data[idx]
        input_data[data_counter] = temp_data
        input_label[data_counter] = temp_label
        data_counter += 1

mask = ~(input_data == 0).all(axis=(1,2,3))
final_data = input_data[mask]

mask = ~(input_label == 0).all(axis = (1))
final_label = input_label[mask]

convertedLabel = np.zeros(shape=(903, 1))
for idx in range(len(final_label)):
    if np.array_equal(final_label[idx], ['complex']):
        convertedLabel[idx] = 0
    elif np.array_equal(final_label[idx], ['foot jolt']):
        convertedLabel[idx] = 1
    elif np.array_equal(final_label[idx], ['toes wiggling']):
        convertedLabel[idx] = 2


idx = np.arange(len(convertedLabel))
np.random.shuffle(idx) # in-placeo operation
train_idx = idx[:int(len(idx)*2/3)]
test_idx = idx[int(len(idx)*2/3):]
train_data = final_data[train_idx,:,:]
train_label = convertedLabel[train_idx]
test_data = final_data[test_idx,:,:]
test_label = convertedLabel[test_idx]



## input format
# train_data.shape: (numSamples, numFeats, numChans), (324,256,64)
# train_label.shape: (numSamples, 1), (324,1)


# %% purely keras syntac, CNN
batch_size = 30
num_classes = 3
epochs = 100

# input dimensions
num_chans = 12
num_frequence = 11
num_timeSmps = 9

# convert class vectors to binary class matrices
train_label = keras.utils.to_categorical(train_label, num_classes)
test_label = keras.utils.to_categorical(test_label, num_classes)

# clear existing model
# del model
# K.clear_session()

model = Sequential()
model.add(Conv2D(64, kernel_size=(2),
                 activation='relu', padding='same',
                 input_shape=(num_chans, num_frequence ,num_timeSmps)))  # tensorflow format, "channels_last"
model.add(MaxPooling2D(pool_size=(1, 2)))
model.add(Conv2D(128, (2), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(1, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(momentum=0.9, decay=0.9),
              metrics=['accuracy'])

model.fit(train_data, train_label,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(test_data, test_label))
score = model.evaluate(test_data, test_label, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])