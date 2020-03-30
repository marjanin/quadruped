import os
import pickle
from matplotlib import pyplot as plt
import tensorflow as tf
from all_functions import *

experiment_ID = 'exp2_2'
if experiment_ID == 'exp1_2':
	use_sensory = True
else:
	use_sensory = False
Mj_render = False
dt = .01
number_of_all_runs = 50

attempt_kinematics = create_cyclical_movements_fcn(omega = 3, attempt_length = 10, timestep = 0.01)
MuJoCo_model_name = "tendon_quadruped_ws_heavyonfloor.xml"
performances=[]
for run_no in range(number_of_all_runs):
	print(run_no)
	[returned_kinematics, returned_sensorreads, returned_est_activations ] = run_activations_ws_cl_sepANNs_fcn(
	MuJoCo_model_name, attempt_kinematics, log_address="./log/{}/{}/".format(experiment_ID,run_no), timestep=dt, use_sensory=use_sensory, Mj_render=Mj_render) # this should be cl
	RMSE = np.sqrt(np.mean(np.square((returned_kinematics[-5000:,:8]-attempt_kinematics[-5000:,:8]))))
	performances.append(RMSE)
print("mean:", np.mean(performances))
print("std:", np.std(performances))

#on air:
# exp1_2:
# mean: 0.2400054126581516
# std: 0.01953127372249579
# exp2_2:
# mean: 0.2609049716072669
# std: 0.015688751511170444

#on floor:
# exp1_2:
# mean: 0.36153386590006853
# std: 0.02076464625008372
# exp2_2:
# mean: 0.3584563929198147
# std: 0.012293704820328823

# #on heavy20%:
# exp1_2:
# mean: 0.40585830096453007
# std: 0.02369045150546554
# exp2_2:
# mean: 0.394236967669215
# std: 0.016023563178252855

# #on heavy10%:
# exp1_2:
# mean: 0.3845226383978554
# std: 0.023950545850844437
# exp2_2:
# mean: 0.375854134820465
# std: 0.012927174315521837

# #on heavy minus 10%:
# exp1_2:
# mean: 0.3222816390324178
# std: 0.020760629025793075
# exp2_2:
# mean: 0.32681647039001005
# std: 0.015691114693232793
