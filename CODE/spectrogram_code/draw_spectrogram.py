from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from itertools import islice
from mpl_toolkits.axes_grid1 import make_axes_locatable

DIR = '/Users/zhengli/Desktop/Projects/MS-thesis/Data/PED2V1_2017_03_13/splitData/right/leg_'
SENSOR = 'Capacitor_3'
CSV_DIR = DIR + SENSOR + '.csv'

output = []

with open(CSV_DIR, 'r', newline='') as f:
    reader = csv.reader(f)
    output = list(islice((row[1] for row in reader), 117647,117848))

csv_file = np.asarray(output, dtype=np.float64)

# print(csv_file)
windows = [30]
overlap = [10, 15, 20]
Save_dir = '/Users/zhengli/Desktop/Projects/MS-thesis/Data/PED2V1_2017_03_13/splitData/right/spectrogram'
COMPLEX = '/complex/'
FOOT = '/foot/'
TOE = '/toe/'

for win in windows:
    if win == 30:
        f, t, Sxx = signal.spectrogram(csv_file, fs=50, nperseg=win)
        fig = plt.pcolormesh(t, f, Sxx)
        plt.colorbar(fig)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        # plt.show()
        s_win = str(win)
        save_path = Save_dir + TOE + SENSOR + '_w' + s_win + '.png'
        plt.savefig(save_path)
        plt.clf()
        for ol in overlap:
            f, t, Sxx = signal.spectrogram(csv_file, fs=50, nperseg=win, noverlap=ol)
            fig = plt.pcolormesh(t, f, Sxx)
            plt.colorbar(fig)
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            # plt.show()
            s_win = str(win)
            s_ol = str(ol)
            save_path = Save_dir + TOE + SENSOR + '_w' + s_win + '_ol' + s_ol + '.png'
            plt.savefig(save_path)
            plt.clf()

    else:
        f, t, Sxx = signal.spectrogram(csv_file, fs=50, nperseg=win)
        fig = plt.pcolormesh(t, f, Sxx)
        plt.colorbar(fig)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        # plt.show()
        s_win = str(win)
        save_path = Save_dir + TOE + SENSOR + '_w' + s_win + '_ol0' + '.png'
        plt.savefig(save_path)
        plt.clf()


