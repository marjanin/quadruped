import os
import pickle
from matplotlib import pyplot as plt
import tensorflow as tf
from all_functions import *

experiment_ID = 'test'
number_of_all_runs = 50
np.zeros(number_of_all_runs,)
all_performances=[]
dt=.01 # time step
np.random.seed(0) # setting the seed for numpy's random number generator
for run_no in range(number_of_all_runs):
	## initialization

	babbling = True
	use_sensory = True
	number_of_legs = 4
	ANNs = number_of_legs*[None]
	#phase 1 - babbling on air
	MuJoCo_model_name = "tendon_quadruped_ws_onair.xml"
	babbling_signal_duration_in_seconds=1*60 # babbling duration
	## generating babbling data
	babbling_signals = babbling_input_gen_fcn(
		number_of_signals=8,
		signal_duration_in_seconds=babbling_signal_duration_in_seconds,
		pass_chance=dt,
		max_in=1,
		min_in=-1,
		dt=dt)
	## running the babbling data through the plant
	est_activations = babbling_signals
	[babbling_kinematics_p1, real_attempt_sensorreads_p1, babbling_activations_p1] = run_activations_ws_ol_fcn(
	MuJoCo_model_name, est_activations, timestep=0.01, Mj_render=False) # this should be ol
	# training the neural network
	Inverse_ANN_models = inverse_mapping_ws_sepANNs_fcn(
		babbling_kinematics_p1, real_attempt_sensorreads_p1, babbling_activations_p1, epochs=10, log_address="./log/{}/{}/".format(experiment_ID,run_no), use_sensory=use_sensory, use_prior_model=False) #
	# phase 2 - babblin on floor
	MuJoCo_model_name = "tendon_quadruped_ws_onfloor.xml"
	babbling_signal_duration_in_seconds=.25*60 # babbling duration
	## generating babbling data
	babbling_signals = babbling_input_gen_fcn(
		number_of_signals=8,
		signal_duration_in_seconds=babbling_signal_duration_in_seconds,
		pass_chance=dt,
		max_in=1,
		min_in=-1,
		dt=dt)
	## running the babbling data through the plant
	est_activations = babbling_signals
	[babbling_kinematics_p2, real_attempt_sensorreads_p2, babbling_activations_p2] = run_activations_ws_ol_fcn(
	MuJoCo_model_name, est_activations, timestep=dt, Mj_render=False) # this should be ol
	# concatinating babbling data from two phases
	babbling_kinematics = np.concatenate((babbling_kinematics_p1, babbling_kinematics_p2),axis=0)
	babbling_sensorreads = np.concatenate((real_attempt_sensorreads_p1, real_attempt_sensorreads_p2),axis=0)
	babbling_activations = np.concatenate((babbling_activations_p1, babbling_activations_p2),axis=0)
	# training the neural network
	Inverse_ANN_models = inverse_mapping_ws_sepANNs_fcn(
		babbling_kinematics, babbling_sensorreads, babbling_activations, epochs=10, log_address="./log/{}/{}/".format(experiment_ID,run_no), use_sensory=use_sensory, use_prior_model=True) #
	# creating the cyclical movement kinematics
	MuJoCo_model_name = "tendon_quadruped_ws_onfloor.xml"
	attempt_kinematics = create_cyclical_movements_fcn(omega = 3, attempt_length = 10, timestep = dt)

	# running the activations created for the cyclical movements
	[returned_kinematics, returned_sensorreads, returned_est_activations ] = run_activations_ws_cl_sepANNs_fcn(
	MuJoCo_model_name, attempt_kinematics, log_address="./log/{}/{}/".format(experiment_ID,run_no), timestep=dt, use_sensory=use_sensory, Mj_render=False) # this should be cl
	# calculating RMSE
	RMSE = np.sqrt(np.mean(np.square((returned_kinematics[:,:8]-attempt_kinematics[:,:8]))))
	print("RMSE:", RMSE)
	# concatingating babbling and run data
	kinematics_all=babbling_kinematics
	sensory_all=babbling_sensorreads
	est_activations_all=babbling_activations
	performances=[RMSE]
	# the L2 loop
	for ii in range(10):
		kinematics_all = np.concatenate((kinematics_all,returned_kinematics),axis=0)
		sensory_all = np.concatenate((sensory_all,returned_sensorreads),axis=0)
		est_activations_all = np.concatenate((est_activations_all,returned_est_activations),axis=0)

		Inverse_ANN_models = inverse_mapping_ws_sepANNs_fcn(
		kinematics_all, sensory_all, est_activations_all, epochs=5, log_address="./log/{}/{}/".format(experiment_ID,run_no), use_sensory=use_sensory, use_prior_model=True) #
		Mj_render = False
		if ii == 11:
			Mj_render=True
		[returned_kinematics, returned_sensorreads, returned_est_activations ] = run_activations_ws_cl_sepANNs_fcn(
		MuJoCo_model_name, attempt_kinematics, log_address="./log/{}/{}/".format(experiment_ID,run_no), timestep=dt, use_sensory=use_sensory, Mj_render=Mj_render) # this should be cl
		RMSE = np.sqrt(np.mean(np.square((returned_kinematics[:,:8]-attempt_kinematics[:,:8]))))
		print("Run #:", ii)
		print("RMSE:", RMSE)
		performances.append(RMSE)
	print(performances)
	# plt.plot(performances)
	# plt.show(block=True)
	all_performances.append(performances)
np.save('./results/{}_results'.format(experiment_ID),all_performances) 
#import pdb; pdb.set_trace()
