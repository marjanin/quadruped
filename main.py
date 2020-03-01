from matplotlib import pyplot as plt
import sklearn.model_selection
from babbling_functions import *
from all_functions import *


## initialization
dt=0.01 # time step
signal_duration_in_seconds=1*60 # babbling duration
np.random.seed(0) # setting the seed for numpy's random number generator

## generating babbling data
babbling_signals = babbling_input_gen_fcn(
	number_of_signals=8,
	signal_duration_in_seconds=signal_duration_in_seconds,
	pass_chance=dt,
	max_in=1,
	min_in=-1,
	dt=dt)
# plotting the generated babbling data (optional)
plt.figure()
plt.plot(babbling_signals)
plt.title('generated babbling inputs')
plt.xlabel('sample #')
plt.show(block=False)

## running the babbling data through the plant
est_activations = babbling_signals
MuJoCo_model_name = "tendon_quadruped_onair.xml"
[babbling_kinematics, babbling_activations] = run_activations_fcn(MuJoCo_model_name, est_activations, timestep=0.01, Mj_render=False)
# kinematics = NxM N number of samples, M = DoFs x 3 (pos x DoFs, vel x DoFs, acc x DoF)
# plotting
plt.figure()
plt.plot(babbling_kinematics[:,:8])
plt.title('generated babbling inputs')
plt.xlabel('sample #')
plt.show(block=False)

# training the neural network
model = inverse_mapping_fcn(babbling_kinematics, babbling_activations, log_address="./log/save", early_stopping=False)
est_activations=model.predict(babbling_kinematics)
plt.figure()
plt.plot(est_activations[:])
plt.show(block=False)

#import pdb; pdb.set_trace()

attempt_kinematics = create_sin_cos_kinematics_fcn(attempt_length = 5 , number_of_cycles = 4, timestep = 0.01)
est_activations=model.predict(attempt_kinematics)

[returned_kinematics, returned_est_activations] = run_activations_fcn(MuJoCo_model_name, est_activations, timestep=0.01, Mj_render=True)

# kinematics = dummy_plant_fcn(activations)
# plt.figure()
# plt.plot(kinematics[:,0])
# plt.plot(kinematics[:,3])
# plt.legend(['q0','q1'])
# plt.title('plant joint locations (babbling)')
# plt.xlabel('sample #')
# plt.show(block=False)

# ## creating the inverse map (training the NN model)
# model=inverse_mapping_fcn(kinematics, activations, log_address='./log', early_stopping=False)

# ## running the inverse model (optional)
# est_activations=model.predict(kinematics)

# ## compare desired and estimateds (optional)
# fig, axs = plt.subplots(1, 3, figsize=(9, 3))
# axs[0].plot(np.linspace(0,signal_duration_in_seconds,num=int(signal_duration_in_seconds/dt)),
# 	activations[:,0],
# 	np.linspace(0,signal_duration_in_seconds,int(signal_duration_in_seconds/dt)),
# 	est_activations[:,0])
# axs[1].plot(np.linspace(0,signal_duration_in_seconds,int(signal_duration_in_seconds/dt)),
# 	activations[:,1],
# 	np.linspace(0,signal_duration_in_seconds,int(signal_duration_in_seconds/dt)),
# 	est_activations[:,1])
# axs[2].plot(np.linspace(0,signal_duration_in_seconds,int(signal_duration_in_seconds/dt)),
# 	activations[:,2],
# 	np.linspace(0,signal_duration_in_seconds,int(signal_duration_in_seconds/dt)),
# 	est_activations[:,2])
# axs[0].legend(['Activation','Estimation'])
# for axes_index in range(3):
# 	axs[axes_index].set_xlabel('time(s)')
# fig.suptitle('Activations vs. Estimations (babbling)')
# plt.subplots_adjust(bottom=0.15)
# plt.show(block=False)

# ## testing a 2-DOF system with cyclical and point-to-point movements
# # sin_cos
# q0_sincos, q1_sincos = create_sin_cos_positions_fcn(number_of_cycles=10, duration_of_each_cycle=3, dt=0.01)
# desired_sincos_attempt_kinematics = positions_to_kinematics_fcn(q0_sincos, q1_sincos, dt = 0.01)
# # ploting (optional)
# plt.figure()
# plt.plot(desired_sincos_attempt_kinematics)
# plt.title('desired sin-cos kinematics')
# plt.xlabel('sample #')
# plt.show(block=False)
# #p2p
# q0_p2p, q1_p2p = create_point2point_positions_fcn(number_of_positions=10, duration_of_each_position=3, dt=0.01)
# # LP FILTER (FIR-MA)
# q0_p2p_filtered = LP_filt(50, q0_p2p)
# q1_p2p_filtered = LP_filt(50, q1_p2p)
# # positions to kinematics
# desired_p2p_attempt_kinematics = positions_to_kinematics_fcn(q0_p2p_filtered, q1_p2p_filtered, dt = 0.01)
# # plotting (optional)
# plt.figure()
# plt.plot(desired_p2p_attempt_kinematics)
# plt.title('desired p2p kinematics')
# plt.xlabel('sample #')
# plt.show(block=False)

# ## Testing systems performance on the tasks
# est_sincos_activations=model.predict(desired_sincos_attempt_kinematics)
# plant_sincos_attempt_kinematics = dummy_plant_fcn(est_sincos_activations)
# error_q0_sincos = error_cal_fcn(q0_sincos, plant_sincos_attempt_kinematics[:,0], disregard_error_percentage = 0)
# error_q1_sincos = error_cal_fcn(q1_sincos, plant_sincos_attempt_kinematics[:,3], disregard_error_percentage = 0)
# plt.figure()
# plt.plot(np.linspace(0,desired_sincos_attempt_kinematics.shape[0]*dt,desired_sincos_attempt_kinematics.shape[0]),
# 	desired_sincos_attempt_kinematics[:,0],color='C0')
# plt.plot(np.linspace(0,desired_sincos_attempt_kinematics.shape[0]*dt,desired_sincos_attempt_kinematics.shape[0]),
# 	plant_sincos_attempt_kinematics[:,0],linestyle='--',color='C0')
# plt.plot(np.linspace(0,desired_sincos_attempt_kinematics.shape[0]*dt,desired_sincos_attempt_kinematics.shape[0]),
# 	desired_sincos_attempt_kinematics[:,3],color='C1')
# plt.plot(np.linspace(0,desired_sincos_attempt_kinematics.shape[0]*dt,desired_sincos_attempt_kinematics.shape[0]),
# 	plant_sincos_attempt_kinematics[:,3],linestyle='--',color='C1')
# plt.xlabel('time (s)')
# plt.legend(['desired q0','plant q0','desired q1','plant q1'])
# plt.title('sin-cos task results')
# plt.show(block=False)

# est_p2p_activations=model.predict(desired_p2p_attempt_kinematics)
# plant_p2p_attempt_kinematics = dummy_plant_fcn(est_p2p_activations)
# error_q0_p2p = error_cal_fcn(desired_p2p_attempt_kinematics[:,0], plant_p2p_attempt_kinematics[:,0], disregard_error_percentage = 0)
# error_q1_p2p = error_cal_fcn(q1_p2p, plant_p2p_attempt_kinematics[:,3], disregard_error_percentage = 0)

# plt.figure()
# plt.plot(np.linspace(0,desired_p2p_attempt_kinematics.shape[0]*dt,desired_p2p_attempt_kinematics.shape[0]),
# 	desired_p2p_attempt_kinematics[:,0],color='C0')
# plt.plot(np.linspace(0,desired_p2p_attempt_kinematics.shape[0]*dt,desired_p2p_attempt_kinematics.shape[0]),
# 	plant_p2p_attempt_kinematics[:,0],linestyle='--',color='C0')
# plt.plot(np.linspace(0,desired_p2p_attempt_kinematics.shape[0]*dt,desired_p2p_attempt_kinematics.shape[0]),
# 	desired_p2p_attempt_kinematics[:,3],color='C1')
# plt.plot(np.linspace(0,desired_p2p_attempt_kinematics.shape[0]*dt,desired_p2p_attempt_kinematics.shape[0]),
# 	plant_p2p_attempt_kinematics[:,3],linestyle='--',color='C1')
# plt.xlabel('time (s)')
# plt.legend(['desired q0','plant q0','desired q1','plant q1'])
# plt.title('P2P task results')
# plt.show()
#import pdb; pdb.set_trace()
