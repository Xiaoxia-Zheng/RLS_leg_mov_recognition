import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

DIR = '/Users/zhengli/Desktop/Projects/MS-thesis/Data/PED2V1_2017_03_13/splitData/right/spectrogram'
COMPLEX = '/complex/'
FOOT = '/foot/'
TOE = '/toe/'
FOOT_DIR = DIR + FOOT
COMPLEX_DIR = DIR + COMPLEX
TOE_DIR = DIR + TOE
WIN = str(30)
OVERLAP = str(20)

Cap_foot1 = mpimg.imread(FOOT_DIR + 'Capacitor_1_w' + WIN + '_ol' + OVERLAP + '.png')
Cap_foot2 = mpimg.imread(FOOT_DIR + 'Capacitor_2_w' + WIN + '_ol' + OVERLAP + '.png')
Cap_foot3 = mpimg.imread(FOOT_DIR + 'Capacitor_3_w' + WIN + '_ol' + OVERLAP + '.png')

Cap_complx1 = mpimg.imread(COMPLEX_DIR + 'Capacitor_1_w' + WIN + '_ol' + OVERLAP + '.png')
Cap_complx2 = mpimg.imread(COMPLEX_DIR + 'Capacitor_2_w' + WIN + '_ol' + OVERLAP + '.png')
Cap_complx3 = mpimg.imread(COMPLEX_DIR + 'Capacitor_3_w' + WIN + '_ol' + OVERLAP + '.png')

Cap_toe1 = mpimg.imread(TOE_DIR + 'Capacitor_1_w' + WIN + '_ol' + OVERLAP + '.png')
Cap_toe2 = mpimg.imread(TOE_DIR + 'Capacitor_2_w' + WIN + '_ol' + OVERLAP + '.png')
Cap_toe3 = mpimg.imread(TOE_DIR + 'Capacitor_3_w' + WIN + '_ol' + OVERLAP + '.png')


fig = plt.figure()
a=fig.add_subplot(3,3,1)
plt.imshow(Cap_foot1)
a.set_title('Capacitor_1_foot', size='6', y=0.88)
a.axis('off')

a=fig.add_subplot(3,3,2)
plt.imshow(Cap_foot2)
a.set_title('Capacitor_2_foot', size='6', y=0.88)
a.axis('off')

a=fig.add_subplot(3,3,3)
plt.imshow(Cap_foot3)
a.set_title('Capacitor_3_foot', size='6', y=0.88)
a.axis('off')

a=fig.add_subplot(3,3,4)
plt.imshow(Cap_complx1)
a.set_title('Capacitor_1_complx', size='6', y=0.88)
a.axis('off')

a=fig.add_subplot(3,3,5)
plt.imshow(Cap_complx2)
a.set_title('Capacitor_2_complx', size='6', y=0.88)
a.axis('off')

a=fig.add_subplot(3,3,6)
plt.imshow(Cap_complx3)
a.set_title('Capacitor_3_complx', size='6', y=0.88)
a.axis('off')

a=fig.add_subplot(3,3,7)
plt.imshow(Cap_toe1)
a.set_title('Capacitor_1_toe', size='6', y=0.88)
a.axis('off')

a=fig.add_subplot(3,3,8)
plt.imshow(Cap_toe2)
a.set_title('Capacitor_2_toe', size='6', y=0.88)
a.axis('off')

a=fig.add_subplot(3,3,9)
plt.imshow(Cap_toe3)
a.set_title('Capacitor_3_toe', size='6', y=0.88)
a.axis('off')

# plt.show()
plt.subplots_adjust(wspace=0,hspace=0)
plt.savefig(DIR + '/Cap_compare_w' + WIN + '_ol' + OVERLAP + '.png', dpi=300)
plt.clf()



# Acc_foot1 = mpimg.imread(FOOT_DIR + 'Acc_X_w' + WIN + '_ol' + OVERLAP + '.png')
# Acc_foot2 = mpimg.imread(FOOT_DIR + 'Acc_Y_w' + WIN + '_ol' + OVERLAP + '.png')
# Acc_foot3 = mpimg.imread(FOOT_DIR + 'Acc_Z_w' + WIN + '_ol' + OVERLAP + '.png')
#
# Acc_complx1 = mpimg.imread(COMPLEX_DIR + 'Acc_X_w' + WIN + '_ol' + OVERLAP + '.png')
# Acc_complx2 = mpimg.imread(COMPLEX_DIR + 'Acc_Y_w' + WIN + '_ol' + OVERLAP + '.png')
# Acc_complx3 = mpimg.imread(COMPLEX_DIR + 'Acc_Z_w' + WIN + '_ol' + OVERLAP + '.png')
#
# Acc_toe1 = mpimg.imread(TOE_DIR + 'Acc_X_w' + WIN + '_ol' + OVERLAP + '.png')
# Acc_toe2 = mpimg.imread(TOE_DIR + 'Acc_Y_w' + WIN + '_ol' + OVERLAP + '.png')
# Acc_toe3 = mpimg.imread(TOE_DIR + 'Acc_Z_w' + WIN + '_ol' + OVERLAP + '.png')
#
# fig = plt.figure()
# a=fig.add_subplot(2,3,1)
# plt.imshow(Acc_foot1)
# a.set_title('Acc_X_foot', size='6', y=0.88)
# a.axis('off')
#
# a=fig.add_subplot(2,3,2)
# plt.imshow(Acc_foot2)
# a.set_title('Acc_Y_foot', size='6', y=0.88)
# a.axis('off')
#
# a=fig.add_subplot(2,3,3)
# plt.imshow(Acc_foot3)
# a.set_title('Acc_Z_foot', size='6', y=0.88)
# a.axis('off')
#
# a=fig.add_subplot(2,3,4)
# plt.imshow(Acc_complx1)
# a.set_title('Acc_X_complx', size='6', y=0.88)
# a.axis('off')
#
# a=fig.add_subplot(2,3,5)
# plt.imshow(Acc_complx2)
# a.set_title('Acc_Y_complx', size='6', y=0.88)
# a.axis('off')
#
# a=fig.add_subplot(2,3,6)
# plt.imshow(Acc_complx3)
# a.set_title('Acc_Z_complx', size='6', y=0.88)
# a.axis('off')
#
# a=fig.add_subplot(3,3,7)
# plt.imshow(Acc_toe1)
# a.set_title('Acc_X_toe', size='6', y=0.88)
# a.axis('off')
#
# a=fig.add_subplot(3,3,8)
# plt.imshow(Acc_toe2)
# a.set_title('Acc_Y_toe', size='6', y=0.88)
# a.axis('off')
#
# a=fig.add_subplot(3,3,9)
# plt.imshow(Acc_toe3)
# a.set_title('Acc_Z_toe', size='6', y=0.88)
# a.axis('off')
#
# # plt.show()
# plt.subplots_adjust(wspace=0,hspace=0)
# plt.savefig(DIR + '/Acc_compare_w' + WIN + '_ol' + OVERLAP + '.png', dpi=300)
# plt.clf()
#
#
#
#
# Gyro_foot1 = mpimg.imread(FOOT_DIR + 'Gyro_X_w' + WIN + '_ol' + OVERLAP + '.png')
# Gyro_foot2 = mpimg.imread(FOOT_DIR + 'Gyro_Y_w' + WIN + '_ol' + OVERLAP + '.png')
# Gyro_foot3 = mpimg.imread(FOOT_DIR + 'Gyro_Z_w' + WIN + '_ol' + OVERLAP + '.png')
#
# Gyro_complx1 = mpimg.imread(COMPLEX_DIR + 'Gyro_X_w' + WIN + '_ol' + OVERLAP + '.png')
# Gyro_complx2 = mpimg.imread(COMPLEX_DIR + 'Gyro_Y_w' + WIN + '_ol' + OVERLAP + '.png')
# Gyro_complx3 = mpimg.imread(COMPLEX_DIR + 'Gyro_Z_w' + WIN + '_ol' + OVERLAP + '.png')
#
# Gyro_toe1 = mpimg.imread(TOE_DIR + 'Gyro_X_w' + WIN + '_ol' + OVERLAP + '.png')
# Gyro_toe2 = mpimg.imread(TOE_DIR + 'Gyro_Y_w' + WIN + '_ol' + OVERLAP + '.png')
# Gyro_toe3 = mpimg.imread(TOE_DIR + 'Gyro_Z_w' + WIN + '_ol' + OVERLAP + '.png')
#
# fig = plt.figure()
# a=fig.add_subplot(2,3,1)
# plt.imshow(Gyro_foot1)
# a.set_title('Gyro_X_foot', size='6', y=0.88)
# a.axis('off')
#
# a=fig.add_subplot(2,3,2)
# plt.imshow(Gyro_foot2)
# a.set_title('Gyro_Y_foot', size='6', y=0.88)
# a.axis('off')
#
# a=fig.add_subplot(2,3,3)
# plt.imshow(Gyro_foot3)
# a.set_title('Gyro_Z_foot', size='6', y=0.88)
# a.axis('off')
#
# a=fig.add_subplot(2,3,4)
# plt.imshow(Gyro_complx1)
# a.set_title('Gyro_X_complx', size='6', y=0.88)
# a.axis('off')
#
# a=fig.add_subplot(2,3,5)
# plt.imshow(Gyro_complx2)
# a.set_title('Gyro_Y_complx', size='6', y=0.88)
# a.axis('off')
#
# a=fig.add_subplot(2,3,6)
# plt.imshow(Gyro_complx3)
# a.set_title('Gyro_Z_complx', size='6', y=0.88)
# a.axis('off')
#
# a=fig.add_subplot(3,3,7)
# plt.imshow(Gyro_toe1)
# a.set_title('Gyro_X_toe', size='6', y=0.88)
# a.axis('off')
#
# a=fig.add_subplot(3,3,8)
# plt.imshow(Gyro_toe2)
# a.set_title('Gyro_Y_toe', size='6', y=0.88)
# a.axis('off')
#
# a=fig.add_subplot(3,3,9)
# plt.imshow(Gyro_toe3)
# a.set_title('Gyro_Z_toe', size='6', y=0.88)
# a.axis('off')
#
# plt.subplots_adjust(wspace=0,hspace=0)
# plt.savefig(DIR + '/Gyro_compare_w' + WIN + '_ol' + OVERLAP + '.png', dpi=300)
# plt.clf()