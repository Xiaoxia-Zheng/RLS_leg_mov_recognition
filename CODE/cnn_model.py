#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 10:41:05 2018

@author: xiaoxia1
"""


from scipy import signal
import numpy as np
import csv
import matplotlib.pyplot as plt
import json
import re, os
import glob
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


#Default DIR
DIR = '/home/xiaoxia1/Desktop/MS-thesis/leg_Gesture_Recognition/analysis/'

#patient name:
PATIENT =['PED1V1_2017_01_27',
          'PED2V1_2017_03_13', 
          'RZ0015_V1N1 - 2017_06_08', 
          'RZ0016_V1N1 -2017_06_29', 
          'RZ0017_V1N1-2017_07_10', 
          'RZ0018_V1N1-2017_11_07']

#Sub-folder of the split data
RIGHT_FOOT = '/split/right/'
LEFT_FOOT = '/split/left/'

#Dir for the tag json file
SAVED_TAG_JSON = '/saves/tags/'


# temp dataset to store the band's csv data
dataRow = 11
dataColumn = 11
data_from_csv = np.zeros(shape=(1500000, dataRow, dataColumn), dtype=np.float64)
label_from_comment = np.zeros(shape=(1500000, 1), dtype=object)
win_size = 100 #window size
counter = 0 #counter for counting how many spectrograms have been generate

for patient in PATIENT:
    DATA_PATH = []
    patient_dir = DIR + patient
    json_file_dir = patient_dir + SAVED_TAG_JSON
    
    
    #   Get left and right raw data's path        
    for file in glob.glob(patient_dir + RIGHT_FOOT + '*[0-9].*'):
        if file == patient_dir + RIGHT_FOOT + 'leg_Relative_Time_sec.csv' or file == patient_dir + RIGHT_FOOT + 'leg_Time.csv':
            continue
        DATA_PATH.append(file)
    
    for file in glob.glob(patient_dir + LEFT_FOOT + '*[0-9].*'):
        if file == patient_dir + LEFT_FOOT + 'leg_Relative_Time_sec.csv' or file == patient_dir + LEFT_FOOT + 'leg_Time.csv':
            continue
        DATA_PATH.append(file)    
    
    print(DATA_PATH)


# Get all tag json files' path
# read tags' json file
    for file in glob.glob(os.path.join(json_file_dir, '*.json')):        
        with open(file) as tag_data:
            tag = json.load(tag_data)
        tag_list_with_movement = [d for d in tag if d['tagName'] == 'yes']
        tag_times = [f['times'] for f in tag_list_with_movement]
        for i in tag_times:
            for j in i:
                j['comment'] = re.sub(r'\d+(.)\s', '', j['comment'])        
        
        
        for file_num in range(len(DATA_PATH)):
            # print(DATA_PATH[file_num])
            # read split raw data from the band
    #        print("file num:", file_num)
            with open(DATA_PATH[file_num], 'r', newline='') as raw_data:
                csv_data = csv.reader(raw_data)
                sensor_data_list = list(csv_data)
                temp = [item[1] for item in sensor_data_list]
                sensor_data_array = np.asarray(sensor_data_list, dtype=np.float64)
                sensor_data_array_row1 = np.asarray(temp, dtype=np.float64)
                print(len(sensor_data_array_row1))
            
            START = 0
            STOP = 0
            spec_num = 0
            
            for i in tag_times:
                for j in i:
                    start = int(j['start'])
                    stop = int(j['stop'])
                    
                    comment = str(j['comment'])
                    
                    if comment == 'toes wiggling' or comment =='toe movement' or comment =='points right toes down':
                        comment = 'toes movement'
                    elif comment == 'foot jolt'or \
                            comment == 'extends foot' or \
                            comment == 'pulled legs in and then back out' or \
                            comment == 'brings legs in and then extends' or \
                            comment == 'leg shakes side to side' or \
                            comment == 'flexed right foot multiple times and shifted right leg' or \
                            comment == 'flexed right foot then rotated inwards' or \
                            comment == 'foot jerk' or \
                            comment == 'foot movement' or \
                            comment == 'unpoints foot':
                        comment = 'foot movement'
                    else:
                        comment = 'complex leg mov'
    #                print(comments)
                    
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
                                graphic_array = sensor_data_array_row1[START: (START + win_size)]
                                f, t, Sxx = signal.spectrogram(graphic_array, fs=50, nperseg=20, noverlap=12)
                                data_from_csv[counter] = Sxx
                                label_from_comment[counter] = comment
                                counter += 1
                                # fig = plt.pcolormesh(t, f, Sxx)
                                # plt.colorbar(fig)
                                # plt.ylabel('Frequency [Hz]')
                                # plt.xlabel('Time [sec]')
                                # save_path = '/Users/zhengli/Desktop/Projects/MS-thesis/Lab_Patient_Data/PED2V1_2017_03_13/CNN_spectrogram/Right/Capacitor_1/'+ comment + str(cap_counter) + '.png'
                                # plt.savefig(save_path)
                                # plt.clf()
                            else:
                                graphic_array = sensor_data_array_row1[START: STOP]
                                num = 0
                                while num < spec_num:
                                    graphic_sub_array = graphic_array[(num * win_size): (num * win_size + win_size)]
                                    f, t, Sxx = signal.spectrogram(graphic_sub_array, fs=50, nperseg=20, noverlap=12)
                                    # if file_num < 6:
                                    data_from_csv[counter] = Sxx
                                    label_from_comment[counter] = comment
                                    counter += 1
    #                                print(counter)
#                                    print(Sxx.shape)
                                    # fig = plt.pcolormesh(t, f, Sxx)
                                    # plt.colorbar(fig)
                                    # plt.ylabel('Frequency [Hz]')
                                    # plt.xlabel('Time [sec]')
                                    # plt.title(comment)
                                    # save_path = '/Users/zhengli/Desktop/Projects/MS-thesis/Lab_Patient_Data/PED2V1_2017_03_13/CNN_spectrogram/Right/Capacitor_1/' + comment + str(cap_counter) + '_' + str(num) + '.png'
                                    # # print(save_path)
                                    # plt.savefig(save_path)
                                    # plt.clf()
                                    num += 1
                            break



# remove all 0 rows of the data set
mask = ~(data_from_csv == 0).all(axis=(1, 2))
data_from_csv = data_from_csv[mask]

#remove all 0 rows of the label set
mask = ~(label_from_comment == 0).all(axis = (1))
label_from_comment = label_from_comment[mask]

#reshape data to multiple layers
layers = 1
len_of_data = int(len(data_from_csv) / layers)

# reshape data
CNN_data = np.reshape(data_from_csv, (len_of_data, layers, dataRow, dataColumn), order='F')
CNN_label = np.delete(label_from_comment, np.arange(len_of_data, 150000), axis=0)

#
input_data_no_cut = np.zeros(shape=(len_of_data, layers, dataRow, dataColumn))
input_label_no_cut = np.zeros(shape=(len_of_data, 1),dtype=object)
data_counter = 0

#find out the 3 type of data that we need: complex, foot jolt and toes wiggling
for idx in range(len(CNN_label)):
    if np.array_equal(CNN_label[idx], ['complex leg mov']) or \
                     np.array_equal(CNN_label[idx], ['foot movement']) or \
                     np.array_equal(CNN_label[idx], ['toes movement']):
        temp_label = CNN_label[idx]
        temp_data = CNN_data[idx]
        input_data_no_cut[data_counter] = temp_data
        input_label_no_cut[data_counter] = temp_label
        data_counter += 1

mask = ~(input_data_no_cut == 0).all(axis=(1,2,3))
final_data = input_data_no_cut[mask]

mask = ~(input_label_no_cut == 0).all(axis = (1))
final_label = input_label_no_cut[mask]

len_of_cut_data = len(final_data)

convertedLabel = np.zeros(shape=(len_of_cut_data, 1))
for idx in range(len(final_label)):
    if np.array_equal(final_label[idx], ['complex leg mov']):
        convertedLabel[idx] = 0
    elif np.array_equal(final_label[idx], ['foot movement']):
        convertedLabel[idx] = 1
    elif np.array_equal(final_label[idx], ['toes movement']):
        convertedLabel[idx] = 2

# convertedLabel = np.repeat(convertedLabel, 10000, axis=0)
# final_label = np.repeat(final_data, 10000, axis=0)


idx = np.arange(len(convertedLabel))
np.random.shuffle(idx) # in-placeo operation
train_idx = idx[:int(len(idx) * 4/5)]
test_idx = idx[int(len(idx) * 4/5):]
train_data = final_data[train_idx,:,:,:]
train_label = convertedLabel[train_idx]
test_data = final_data[test_idx,:,:,:]
test_label = convertedLabel[test_idx]


# %% purely keras syntac, CNN
batch_size = 500
num_classes = 3
epochs = 100

# input dimensions
num_chans = layers

# convert class vectors to binary class matrices
train_label = keras.utils.to_categorical(train_label, num_classes)
test_label = keras.utils.to_categorical(test_label, num_classes)

# clear existing model
# del model
# K.clear_session()

model = Sequential()
model.add(Conv2D(64, kernel_size=(2),
                 activation='relu', padding='same',
                 input_shape=(num_chans, dataRow ,dataColumn)))  # tensorflow format, "channels_last"
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

cnnTrain = model.fit(train_data, train_label,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(test_data, test_label))
score = model.evaluate(x=test_data, y=test_label, batch_size=batch_size, verbose=0)
print('Test accuracy:', score[1])
# summarize history for accuracy
plt.plot(cnnTrain.history['acc'])
plt.plot(cnnTrain.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

print('Test loss:', score[0])
# summarize history for loss
plt.plot(cnnTrain.history['loss'])
plt.plot(cnnTrain.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights('model.h5')



## later...
# 
## load json and create model
#json_file = open('model.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#loaded_model = model_from_json(loaded_model_json)
## load weights into new model
#loaded_model.load_weights("model.h5")
#print("Loaded model from disk")
# 
## evaluate loaded model on test data
#loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#score = loaded_model.evaluate(X, Y, verbose=0)
#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


