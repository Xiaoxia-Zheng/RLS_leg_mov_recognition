import numpy as np
import pandas as pd
import scipy.stats
import pywt
import json
import sys
import itertools
from enum import Enum
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from pywt import wavedec
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from scipy import fftpack


from experimental_data import *

DEMO_MODE = False
mode = pywt.Modes.smooth
mode = pywt.Modes.periodization
# np.random.seed(0)  #TODO remove setting seed when working with real data

FeatureType = Enum("FeatureType", "STD WAVE MULTI DCT")
ActiveFeatureType = FeatureType.DCT

win_size = 500


# Generate non-movement data
def gen_non_movement_data(df, yes_times, no_times, all_times, time_column_name):
    i = 0
    while i < len(all_times):
        found_match = False
        rnd_start_time = np.random.uniform(df[time_column_name].iloc[0], df[time_column_name].iloc[-1])
        rnd_duration = np.abs(np.random.normal(np.mean(yes_times), np.std(yes_times)))

        if rnd_duration < 0.5:
            rnd_duration += 0.5
        rnd_stop_time = rnd_start_time + rnd_duration
        # check if start time matches any existing movement
        for t in all_times:
            if not ((rnd_start_time < t[0] and rnd_stop_time < t[0]) or (rnd_start_time > t[1] and rnd_stop_time > t[1])):
                # print("Found match", i, (rnd_start_time, rnd_start_time + rnd_duration), t)
                found_match = True
                break

        if not found_match:
            no_times.append((rnd_start_time, rnd_stop_time, 'no'))
            i += 1

    if len(no_times) != len(all_times):
        print("Warning, length of no_times does not match length of all_times")

    for nt in no_times:
        all_times.append(nt)



def compute_wave_feature(df, features_list, ind, label=None):
    feature = []
    for col in df:
        # if col in ["Capacitor_1", "Capacitor_2", "Capacitor_3"]:
        if col in ["Capacitor_2"]:  # use these columns to compute features
            data = df[col].iloc[ind]
            if DEMO_MODE:
                print("Data length", len(data))

            # compute wavelet coefs
            w = pywt.Wavelet('haar')
            max_level = pywt.dwt_max_level(len(data), w.dec_len)
            if max_level < 2:
                import sys
                print("Max level %i is < 2" % max_level)
                print(t)
                sys.exit(0)
                continue
            max_level = 2
            if DEMO_MODE and col == "Capacitor_2":
                plot_signal_decomp(data, w, max_level, "DWT: %s - Haar" % col)
            coeffs = wavedec(data, w, level=max_level)
            fv = []
            for i in range(len(coeffs) - 1, -1, -1):
                c = coeffs[i]
                if i == len(coeffs) - 1:
                    coef_description = "cA[%i]" % max_level
                else:
                    coef_description = "cD[%i]" % (i + 1)
                # compute features for column
                f = []
                # f.append(np.average(c))
                f.append(np.std(c))
                f.append(scipy.stats.skew(c))
                f.append(scipy.stats.kurtosis(c))
                # f.append(np.sqrt(np.mean(np.square(c))))  # RMS
                # f.append(np.sqrt(np.sum(np.array(coeffs[i]) ** 2)) / len(coeffs[i]))  # Energy of wavelet
                fv += f
                if DEMO_MODE:
                    print("Features")
                    print("Type:", t[2])
                    # print("Average %s:" % coef_description, f[0])
                    # print("StdDev %s:" % coef_description, f[1])
                    # print("Skew %s:" % coef_description, f[2])
                    # print("Kurtosis %s:" % coef_description, f[3])
                    # print("RMS %s:" % coef_description, f[4])
                    # print("Energy %s:" % coef_description, f[5])
            if DEMO_MODE:
                print("Feature vector", fv)
            # append to form final feature vector for particular label
            feature += fv
    # append feature with label
    if label is not None:
        features_list.append(feature + [label])
    else:
        features_list.append(feature)



def compute_dct_feature(df, features_list, ind, label=None):
    # skip time period if it does not exist in data
    # skip time period if it does not exist in data
    if ind[0].shape[0] == 0:
        return
    feature = []
    for col in df:
        if col in ["Capacitor_1", "Capacitor_2", "Capacitor_3"]:
            data = df[col].iloc[ind]
            data = data.values.astype(np.uint16)

            Zxx = fftpack.dct(data, type=3)
            dct_flat = Zxx.flatten()

            std = np.std(dct_flat)
            if np.isnan(std):
                print("Std is nan", ind, data)
                sys.exit(0)
            feature.append(std)
    if label is not None:
        features_list.append(feature + [label])
    else:
        features_list.append(feature)


    # for col in df:
    #     if col in ["Capacitor_1", "Capacitor_2", "Capacitor_3"]:
    #         data = df[col].iloc[ind]
    #         data = data.values.astype(np.uint16)
    #
    #         Zxx = fftpack.dct(data, type=3)
    #         dct_flat = Zxx.flatten()
    #
    #
    #         dct_flat_len = dct_flat.size
    #
    #         if dct_flat_len > win_size:
    #             sub_feature_num = dct_flat_len / win_size
    #             num = 0
    #             while num < sub_feature_num:
    #                 feature = []
    #                 sub_dct_feature = dct_flat[(num * win_size): (num * win_size + win_size)]
    #                 num += 1
    #                 feature.append(sub_dct_feature)
    #
    #                 if label is not None:
    #                     features_list.append(feature + [label])
    #                 else:
    #                     features_list.append(feature)
    #                 # dct = np.delete(dct_flat, np.arangenge(50, len(dct_flat)))
    #         else:
    #             feature = []
    #             dct = np.zeros(win_size)
    #             dct[0: len(dct_flat)] = dct_flat
    #
    #             if label is not None:
    #                 features_list.append(feature + [label])
    #             else:
    #                 features_list.append(feature)



def compute_std_feature(df, features_list, ind, label=None):
    # skip time period if it does not exist in data
    if ind[0].shape[0] == 0:
        return
    feature = []
    for col in df:
        if col in ["Capacitor_1", "Capacitor_2", "Capacitor_3"]:
            data = df[col].iloc[ind]
            data = data.values.astype(np.uint16)
            std = np.mean(data)
            if np.isnan(std):
                print("Std is nan", ind, data)
                sys.exit(0)
            feature.append(std)
    if label is not None:
        features_list.append(feature + [label])
    else:
        features_list.append(feature)


def compute_multi_feature(df, features_list, ind, compute_average=False, compute_std=False, compute_skew=False, compute_kurtosis=False, compute_rms=False, label=None):
    # skip time period if it does not exist in data
    if ind[0].shape[0] == 0:
        return
    feature = []
    for col in df:
        if col in ["Capacitor_1", "Capacitor_2", "Capacitor_3"]:
            data = df[col].iloc[ind]
            data = data.values.astype(np.uint16)
            avg = np.average(data)
            std = np.std(data)
            skew = scipy.stats.skew(data)
            kurtosis = scipy.stats.kurtosis(data)
            rms = np.sqrt(np.mean(np.square(data)))

            if np.isnan(std):
                print("Std is nan", ind, data)
                sys.exit(0)
            if compute_average:
                feature.append(avg)
            if compute_std:
                feature.append(std)
            if compute_skew:
                feature.append(skew)
            if compute_kurtosis:
                feature.append(kurtosis)
            if compute_rms:
                feature.append(rms)
    if label is not None:
        features_list.append(feature + [label])
    else:
        features_list.append(feature)

def create_svm_model(features, C=100, gamma=0.001, kernel='rbf'):
    # encode labels
    labels = features[:, -1]  # no motion, motion
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    # print(le.classes_)
    encoded_labels = le.transform(labels)
    # print(encoded_labels)
    features[:, -1] = encoded_labels  # replace human readable labels with encoded labels
    print("features shape: ", features.shape)

    # shuffle data and split into a training and testing sets
    data = features[:, :-1]
    labels = features[:, -1]
    data, labels = shuffle(data, labels, random_state=0)
    # data_subset, data_test_subset = np.split(data, 2)
    # labels_subset, labels_test_subset = np.split(labels, 2)
    data_subset, data_test_subset, labels_subset, labels_test_subset = train_test_split(data, labels, test_size=0.4, random_state=0)

    train_dataset = np.zeros(shape = (len(data_subset), len(data_subset[0][0])), dtype = np.float64)

    for idx in range(len(data_subset)):
        train_dataset[idx, :] = data_subset[idx][0]

    test_dataset = np.zeros(shape=(len(data_test_subset), len(data_test_subset[0][0])), dtype=np.float64)
    for idx in range(len(data_test_subset)):
        test_dataset[idx, :] = data_test_subset[idx][0]

    train_labels = np.asarray(labels_subset, dtype=np.float64)
    test_labels = np.asarray(labels_test_subset, dtype=np.float64)


    clf = svm.SVC(C=C, gamma=gamma, kernel=kernel)  # classifier, must be fitted to a model
    # clf = svm.SVC()  # classifier, must be fitted to a model
    clf.fit(train_dataset, train_labels)

    report = calculate_accuracy(clf, test_dataset, test_labels)

    model_file_name = "motion_detection_svm_" + kernel + "_C%i" % C
    if kernel == 'rbf':
        model_file_name += "_gamma%.4f" % gamma
    model_file_name += ".pkl"

    print(model_file_name)
    joblib.dump(clf, model_file_name)  # save model
    return report



def create_knn_model(features):
    # encode labels
    labels = features[:, -1]  # no motion, motion
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    # print(le.classes_)
    encoded_labels = le.transform(labels)
    # print(encoded_labels)
    features[:, -1] = encoded_labels  # replace human readable labels with encoded labels
    # print(features)

    # shuffle data and split into a training and testing sets
    data = features[:, :-1]
    labels = features[:, -1]

    data, labels = shuffle(data, labels, random_state=0)
    # data_subset, data_test_subset = np.split(data, 2)
    # labels_subset, labels_test_subset = np.split(labels, 2)

    data_subset, data_test_subset, labels_subset, labels_test_subset = train_test_split(data, labels, test_size=0.6, random_state=0)

    print("shape:", data_subset.shape)

    train_dataset = np.zeros(shape=(len(data_subset), len(data_subset[0][0])), dtype=np.float64)

    for idx in range(len(data_subset)):
        train_dataset[idx, :] = data_subset[idx][0]

    print("new shape:", train_dataset.shape)

    test_dataset = np.zeros(shape=(len(data_test_subset), len(data_test_subset[0][0])), dtype=np.float64)
    for idx in range(len(data_test_subset)):
        test_dataset[idx, :] = data_test_subset[idx][0]

    train_labels = np.asarray(labels_subset, dtype=np.float64)
    test_labels = np.asarray(labels_test_subset, dtype=np.float64)

    # train knn model
    knn = KNeighborsClassifier(n_neighbors=9)
    knn.fit(train_dataset, train_labels.ravel())

    report = calculate_accuracy(knn, test_dataset, test_labels)

    model_file_name = "motion_detection_knn"
    model_file_name += ".pkl"

    print(model_file_name)
    joblib.dump(knn, model_file_name)  # save model
    return report


def create_lr_model(features):
    # encode labels
    labels = features[:, -1]  # no motion, motion
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    # print(le.classes_)
    encoded_labels = le.transform(labels)
    # print(encoded_labels)
    features[:, -1] = encoded_labels  # replace human readable labels with encoded labels
    # print(features)

    # shuffle data and split into a training and testing sets
    data = features[:, :-1]
    labels = features[:, -1]
    data, labels = shuffle(data, labels, random_state=0)
    # data_subset, data_test_subset = np.split(data, 2)
    # labels_subset, labels_test_subset = np.split(labels, 2)
    data_subset, data_test_subset, labels_subset, labels_test_subset = train_test_split(data, labels, test_size=0.6, random_state=0)

    # #train LR model
    lr = LogisticRegression()
    lr.fit(data_subset, labels_subset)

    data_test_subset = data_test_subset.astype(np.float64)
    print(data_test_subset.dtype)
    report = calculate_accuracy(lr, data_test_subset, labels_test_subset)

    model_file_name = "motion_detection_knn"
    model_file_name += ".pkl"

    print(model_file_name)
    joblib.dump(lr, model_file_name)  # save model
    return report



def calculate_accuracy(clf, data_test_subset, labels_test_subset):
    class_names = ["mov", "none_mov"]
    # print(clf)
    # print("support vectors shape", clf.support_vectors_.shape)

    pred = clf.predict(data_test_subset)
    # print("Predicted labels", pred)
    # print("True labels", labels_test_subset)

    score = accuracy_score(labels_test_subset, pred) * 100
    print("Score:", score)

    y_true = labels_test_subset
    y_pred = pred
    cm = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cm, classes=class_names,
                          title='Confusion matrix, without normalization')
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cm, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')
    plt.show()

    tp = float(cm[0][0]) / np.sum(cm[0])
    tn = float(cm[1][1]) / np.sum(cm[1])

    print("True positive:", tp)
    print("True negative:", tn)

    print(classification_report(y_true, y_pred))
    return tp,tn,score

def grid_search(features):
    X = features[:, :-1]
    y = features[:, -1]

    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.6, random_state=0)

    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=5,
                           scoring='%s_macro' % score)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()


def read_labeled_data_and_calc_features(f_data, f_labels, time_column_name, features_list):
    print('data: ' + f_data)
    print('label: ' + f_labels)
    df = pd.read_csv(f_data, delimiter=",")

    # Collect all save files
    saveFiles = []
    tags = []
    if f_labels is not None:
        for (dirpath, dirnames, filenames) in os.walk(f_labels):
            for f_data in filenames:
                if f_data.endswith('.json'):
                    saveFiles.append(os.path.join(dirpath, f_data))

        # Use most recent save file
        saveFiles.sort()
        with open(saveFiles[-1], 'r') as f_data:
            tags = json.load(f_data)

    all_times = []
    yes_times = []
    no_times = []

    for tag in tags:
        label = tag["tagName"]
        for t in tag["times"]:
            if label == 'yes':
                yes_times.append(t['stop'] - t['start'])
            # filter out any events before capsense data
            if t['start'] >= df[time_column_name].iloc[0]:
                all_times.append((t['start'], t['stop'], label))
    total_duration = df[time_column_name].iloc[-1] - df[time_column_name].iloc[0]
    print("Total duration (hours):", total_duration/3600)
    print("Up time (%):", 100*np.sum(yes_times)/total_duration)

    if len(yes_times) > 0:
        # Output some stats about the motion data, such as count, mean, histogram and PDF
        print("Yes label count:", len(yes_times))
        print("Yes average duration (sec):", np.mean(yes_times))
        print("Yes stdDev:", np.std(yes_times))
        if False:
            # http://stackoverflow.com/questions/20011494/plot-normal-distribution-with-matplotlib
            h = sorted(yes_times)
            fit = scipy.stats.norm.pdf(h, np.mean(h), np.std(h))  # this is a fitting indeed
            plt.plot(h, fit, '-o', label='norm pdf')
            plt.hist(h, normed=True)  # use this to draw histogram of your data
            plt.legend(loc='best', frameon=False)
            plt.xlabel('Motion duration (sec)')
            plt.show()

    gen_non_movement_data(df, yes_times, no_times, all_times, time_column_name)

    # debug plotting
    if False:
        plt.figure()
        plt.plot(df[time_column_name], df["Capacitor_1"])
        plt.plot(df[time_column_name], df["Capacitor_2"])
        plt.plot(df[time_column_name], df["Capacitor_3"])
        plt.legend()

        for t in all_times:
            if t[2] == 'no':
                ind = np.where((df[time_column_name] > t[0]) & (df[time_column_name] < t[1]))  # find indices in time range
                plt.axvspan(t[0], t[1], color='blue', alpha=0.3)
            if t[2] == 'yes':
                ind = np.where((df[time_column_name] > t[0]) & (df[time_column_name] < t[1]))  # find indices in time range
                # plt.plot(df["Relative_Time_sec"].iloc[ind], df["Capacitor_3"].iloc[ind])
                plt.axvspan(t[0], t[1], color='red', alpha=0.3)
        plt.title("Cap")
        plt.show()

        plt.figure()
        plt.plot(df[time_column_name], df["Acc_X"])
        plt.plot(df[time_column_name], df["Acc_Y"])
        plt.plot(df[time_column_name], df["Acc_Z"])
        plt.legend()

        for t in all_times:
            if t[2] == 'no':
                ind = np.where((df[time_column_name] > t[0]) & (df[time_column_name] < t[1]))  # find indices in time range
                # plt.plot(df["Relative_Time_sec"].iloc[ind], df["Acc_X"].iloc[ind])
                plt.axvspan(t[0], t[1], color='blue', alpha=0.3)
            if t[2] == 'yes':
                ind = np.where((df[time_column_name] > t[0]) & (df[time_column_name] < t[1]))  # find indices in time range
                # plt.plot(df["Relative_Time_sec"].iloc[ind], df["Acc_X"].iloc[ind])
                plt.axvspan(t[0], t[1], color='red', alpha=0.3)
        plt.title("Acc")
        plt.show()

    # compute features
    if ActiveFeatureType is FeatureType.WAVE:
        # calculate features based on wavelet coeffs
        for t in all_times:
            ind = np.where((df[time_column_name] > t[0]) & (df[time_column_name] < t[1]))  # find indices in time range
            compute_wave_feature(ind, label=t[2])
    elif ActiveFeatureType is FeatureType.STD:
        # calculate features based on std
        for t in all_times:
            ind = np.where((df[time_column_name] > t[0]) & (df[time_column_name] < t[1]))  # find indices in time range
            compute_std_feature(df, features_list, ind, label=t[2])
    elif ActiveFeatureType is FeatureType.MULTI:
        # calculate features based on multiple statistical measures
        for t in all_times:
            ind = np.where((df[time_column_name] > t[0]) & (df[time_column_name] < t[1]))  # find indices in time range
            compute_average, compute_std, compute_skew, compute_kurtosis, compute_rms = MULTI_FEATURE_SELECTION
            compute_multi_feature(df, features_list, ind,
                                  compute_average=compute_average,
                                  compute_std=compute_std,
                                  compute_skew=compute_skew,
                                  compute_kurtosis=compute_kurtosis,
                                  compute_rms=compute_rms,
                                  label=t[2])

    elif ActiveFeatureType is FeatureType.DCT:
        # calculate features based on dct
        for t in all_times:
            ind = np.where((df[time_column_name] > t[0]) & (df[time_column_name] < t[1]))  # find indices in time range
            compute_dct_feature(df, features_list, ind, label=t[2])


def read_data():
    features_list = []

    # # PED1V1
    # f_data = PED1V1["right"]
    # f_labels = PED1V1["marks"]
    # time_column_name = PED1V1["time_column_name"]
    #
    # print("PED1V1, right:")
    # read_labeled_data_and_calc_features(f_data, f_labels, time_column_name, features_list)
    #
    # f_data = PED1V1["left"]
    # f_labels = PED1V1["marks"]
    # time_column_name = PED1V1["time_column_name"]
    #
    # print("PED1V1, left:")
    # read_labeled_data_and_calc_features(f_data, f_labels, time_column_name, features_list)
    #
    # # PED2V1
    # f_data = PED2V1["right"]
    # f_labels = PED2V1["marks"]
    # time_column_name = PED2V1["time_column_name"]
    #
    # print("PED2V1, right:")
    # read_labeled_data_and_calc_features(f_data, f_labels, time_column_name, features_list)
    #
    # f_data = PED2V1["left"]
    # f_labels = PED2V1["marks"]
    # time_column_name = PED2V1["time_column_name"]
    #
    # print("PED2V1, left:")
    # read_labeled_data_and_calc_features(f_data, f_labels, time_column_name, features_list)

    # RZ0015_V1N1
    f_data = RZ0015_V1N1["right"]
    f_labels = RZ0015_V1N1["marks"]
    time_column_name = RZ0015_V1N1["time_column_name"]

    read_labeled_data_and_calc_features(f_data, f_labels, time_column_name, features_list)

    f_data = RZ0015_V1N1["left"]
    f_labels = RZ0015_V1N1["marks"]
    time_column_name = RZ0015_V1N1["time_column_name"]

    read_labeled_data_and_calc_features(f_data, f_labels, time_column_name, features_list)

    # RZ0016_V1N1
    f_data = RZ0016_V1N1["right"]
    f_labels = RZ0016_V1N1["marks"]
    time_column_name = RZ0016_V1N1["time_column_name"]

    read_labeled_data_and_calc_features(f_data, f_labels, time_column_name, features_list)

    f_data = RZ0016_V1N1["left"]
    f_labels = RZ0016_V1N1["marks"]
    time_column_name = RZ0016_V1N1["time_column_name"]

    read_labeled_data_and_calc_features(f_data, f_labels, time_column_name, features_list)

    # RZ0017_V1N1
    f_data = RZ0017_V1N1["right"]
    f_labels = RZ0017_V1N1["marks"]
    time_column_name = RZ0017_V1N1["time_column_name"]

    read_labeled_data_and_calc_features(f_data, f_labels, time_column_name, features_list)

    f_data = RZ0017_V1N1["left"]
    f_labels = RZ0017_V1N1["marks"]
    time_column_name = RZ0017_V1N1["time_column_name"]

    read_labeled_data_and_calc_features(f_data, f_labels, time_column_name, features_list)

    # RZ0018_V1N1
    f_data = RZ0018_V1N1["right"]
    f_labels = RZ0018_V1N1["marks"]
    time_column_name = RZ0018_V1N1["time_column_name"]

    read_labeled_data_and_calc_features(f_data, f_labels, time_column_name, features_list)

    f_data = RZ0018_V1N1["left"]
    f_labels = RZ0018_V1N1["marks"]
    time_column_name = RZ0018_V1N1["time_column_name"]

    read_labeled_data_and_calc_features(f_data, f_labels, time_column_name, features_list)


    print("Number of features:", len(features_list))
    features = np.array(features_list)
    print("Example feature", features[0])
    # i = 0
    # while i < len(features_list):
    #     print("featires:", features_list[i])
    #     i += 1
    return features


def train_model(features):
    # create_svm_model(features, C=100, gamma=0.001, kernel='rbf')
    # create_svm_model(features, C=1000, gamma=0.001, kernel='rbf')
    # create_svm_model(features, C=10000, gamma=0.001, kernel='rbf')
    # create_svm_model(features, C=1, kernel='linear')
    # create_svm_model(features, C=10, kernel='linear')
    # create_svm_model(features, C=100, kernel='linear') # smallest support vectors
    # report = create_svm_model(features, C=100, gamma=0.001, kernel='rbf')
    report = create_knn_model(features)
    # report = create_lr_model(features)
    return report


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




# train model based on stdev
# ActiveFeatureType = FeatureType.STD
features = read_data()
# np.savez_compressed("features.npz", features)
train_model(features)
# grid_search(features)

# sys.exit(0)
# compare features
# tf_set = set()
# size = 5
# while len(tf_set) < 2**size:
#     t = tuple(np.random.randint(2, size=size, dtype=np.bool))
#     tf_set.add(t)
# ActiveFeatureType = FeatureType.MULTI

# tf_set = [
# (True, True, False, True, True),
# (True, True, True, False, True),  # freezes up
# (True, False, False, False, False),
# (False, False, True, True, True),
# (True, False, True, False, False),
# (True, False, False, True, False),
# (False, False, False, True, True),
# (False, True, False, False, False),
# (False, True, True, True, False),
# (False, True, True, False, False),
# (False, True, False, True, False),
# (True, True, True, False, False),
# (True, True, False, True, False),
# (True, True, False, False, False),
# (True, True, False, False, True),
# (True, False, False, True, True),
# (False, False, True, False, False),
# (True, False, False, False, True),
# (False, False, False, True, False),
# (True, False, True, True, True),
# (False, False, True, True, False),
#(False, False, False, False, False),
# (False, True, False, True, True),
# (True, True, True, True, True),
# (False, True, True, True, True),
# (False, True, False, False, True),
# ]

# reports = []
# for s in tf_set:
#     if s != (False, False, False, False, False):
#         MULTI_FEATURE_SELECTION = s
#         print(MULTI_FEATURE_SELECTION)
#         features = read_data()
#         report = train_model(features)
#         reports.append((s,report))
#
#     for r in reports:
#         print("%s & %s & %s & %s & %s & %.2f & %.2f & %.2f\\\\" % (*r[0], *r[1]))


