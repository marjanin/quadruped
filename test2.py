import os
import pickle
from matplotlib import pyplot as plt
import tensorflow as tf
import multiprocessing as mp
from all_functions import *

attempted_kinematics = create_p2p_movements_fcn(number_of_steps = 10, attempt_length = 10, dt=0.05, filtfilt_N=4)
print(attempted_kinematics.shape)
plt.plot(attempted_kinematics[:,:])

# x = attempted_kinematics[:,1]
# N=2
# b=np.ones(N)/N
# y = signal.filtfilt(b,1,x)

# plt.plot(x)
# plt.plot(y)



plt.show(block=True)


