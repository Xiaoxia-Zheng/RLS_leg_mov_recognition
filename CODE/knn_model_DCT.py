from scipy import signal
from scipy import fftpack
import numpy as np
import csv
import matplotlib.pyplot as plt
import json
import re, os
import glob


#Default DIR
DIR = '/Users/zhengli/Desktop/Projects/MS-thesis/leg_Gesture_Recognition/analysis/'

#patient name:
PATIENT =['PED1V1_2017_01_27']
          # 'PED2V1_2017_03_13',
          # 'RZ0015_V1N1 - 2017_06_08',
          # 'RZ0016_V1N1 -2017_06_29',
          # 'RZ0017_V1N1-2017_07_10',
          # 'RZ0018_V1N1-2017_11_07']

#Sub-folder of the split data
RIGHT_FOOT = '/split/right/'
LEFT_FOOT = '/split/left/'

#Dir for the tag json file
SAVED_TAG_JSON = '/saves/tags/'


# temp dataset to store the band's csv data
#dataRow = 11
#dataColumn = 11
#data_from_csv = np.zeros(shape=(1500000, dataRow, dataColumn), dtype=np.float64)
#label_from_comment = np.zeros(shape=(1500000, 1), dtype=object)
#win_size = 100 #window size
#counter = 0 #counter for counting how many spectrograms have been generate

win_size = 100  # window size
array_len = 100
# temp dataset to store the band's csv data
data_from_csv = np.zeros(shape=(150000, array_len), dtype=np.float64)
label_from_comment = np.zeros(shape=(150000, 1), dtype=object)
counter = 0 #counter for counting how many spectrograms have been generate




for patient in PATIENT:
    DATA_PATH = []
    patient_dir = DIR + patient
    json_file_dir = patient_dir + SAVED_TAG_JSON
    
    
    #   Get left and right raw data's path        
    for file in glob.glob(patient_dir + RIGHT_FOOT + '*[0-9].*'):
        # print(file)
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
#        print(tag)
#        print('\n''\n')
        
   
        
        for file_num in range(len(DATA_PATH)):
            # print(DATA_PATH[file_num])
            # read split raw data from the band
    #        print("file num:", file_num)
            with open(DATA_PATH[file_num], 'r', newline='') as raw_data:
                csv_data = csv.reader(raw_data)
                sensor_data_list = list(csv_data)
                colunm1 = [item[1] for item in sensor_data_list]     #rextract data from column 1
                sensor_data_array = np.asarray(sensor_data_list, dtype=np.float64)
                sensor_data_array_column1 = np.asarray(colunm1, dtype=np.float64)
                print(len(sensor_data_array_column1))
            
            START = 0
            STOP = 0
            spec_num = 0
            
            
            for i in tag_times:
                for j in i:
                    start = int(j['start'])
                    stop = int(j['stop'])
                    
                    comment = str(j['comment'])
                    
                    if comment == 'toes wiggling' or \
                                    comment =='toe movement' or \
                                    comment =='points right toes down':
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
                                column_array = sensor_data_array_column1[START: (START + win_size)]
                                data_from_csv[counter] = column_array
                                label_from_comment[counter] = comment
                                counter += 1
                            else:
                                column_array = sensor_data_array_column1[START: STOP]
                                num = 0
                                while num < spec_num:
                                    column_sub_array = column_array[(num * win_size): (num * win_size + win_size)]
                                    Zxx = fftpack.dct(column_sub_array, type=3)                                    # if file_num < 6:
                                    data_from_csv[counter] = Zxx.flatten()
                                    label_from_comment[counter] = comment
                                    counter += 1
                                    num += 1
                            break


with open('train_dataset.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile)

with open('train_labelset.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile)



# remove all 0 rows of the data set
mask = ~(data_from_csv == 0).all(axis=(1))
data_from_csv = data_from_csv[mask]

#remove all 0 rows of the label set
mask = ~(label_from_comment == 0).all(axis = (1))
label_from_comment = label_from_comment[mask]


len_of_data = len(data_from_csv)

#
input_data_without_cut = np.zeros(shape=(len_of_data, array_len))
input_label_without_cut = np.zeros(shape=(len_of_data, 1),dtype=object)
data_counter = 0

#find out the 3 type of data that we need: complex, foot jolt and toes wiggling
for idx in range(len(label_from_comment)):
    if np.array_equal(label_from_comment[idx], ['complex leg mov']) or \
                     np.array_equal(label_from_comment[idx], ['foot movement']) or \
                                   np.array_equal(label_from_comment[idx], ['toes movement']):
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
train_idx = idx[:int(len(idx)*4/5)]
test_idx = idx[int(len(idx)*4/5):]
train_data = final_data[train_idx,:]
train_label = convertedLabel[train_idx]
test_data = final_data[test_idx,:]
test_label = convertedLabel[test_idx]


# train knn model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

knn = KNeighborsClassifier(n_neighbors=3)

pred = knn.fit(train_data, train_label.ravel()).predict(test_data)
accuracy_knn = accuracy_score(test_label, pred)
print('knn accuracy:', accuracy_knn)

import itertools
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
class_names = ['toe', 'feet', 'leg']

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(test_label, pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()


 # #train LR model
 # from sklearn.linear_model import LogisticRegression
 #
 # logClassifier = LogisticRegression()
 # logClassifier.fit(train_data, train_label.ravel())
 # pred = logClassifier.predict(test_data)
 # accuracy_lr = accuracy_score(test_label, pred)
 # print('lr accuracy:', accuracy_lr)
 #
 # #train svm
 # from sklearn import svm
 # svmClassifier = svm.SVC()
 # svmClassifier.fit(train_data, train_label.ravel())
 # pred = svmClassifier.predict(test_data)
 # accuracy_svm = accuracy_score(test_label, pred)
 # print('svm accuracy:', accuracy_svm)
