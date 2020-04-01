import os
import pickle
from matplotlib import pyplot as plt
import tensorflow as tf
from all_functions import *

def babble_and_refine(MuJoCo_model_name, experiment_ID, run_no, kinematics_all, sensory_all, activations_all, number_of_refinements, use_sensory=True):
	dt=.01 # time step
	babbling = True
	number_of_legs = 4
	ANNs = number_of_legs*[None]
	babbling_signal_duration_in_seconds= 1*80
	refinement_duration_in_seconds = 10
	babbling_signals = babbling_input_gen_fcn(
		number_of_signals=8,
		signal_duration_in_seconds=babbling_signal_duration_in_seconds,
		pass_chance=dt,
		max_in=1,
		min_in=-1,
		dt=dt)
	est_activations = babbling_signals
	attempt_kinematics = create_cyclical_movements_fcn(omega = 3, attempt_length = refinement_duration_in_seconds, timestep = dt)

	[babbling_kinematics, babbling_sensorreads, babbling_activations] = run_activations_ws_ol_fcn(
	MuJoCo_model_name, est_activations, timestep=0.01, Mj_render=False) # this should be ol
	if kinematics_all == []: # initialization
		kinematics_all = babbling_kinematics
		sensory_all = babbling_sensorreads
		activations_all = babbling_activations
		use_prior_model_babbling = False
	else:
		kinematics_all = np.concatenate((kinematics_all,babbling_kinematics),axis=0)
		sensory_all = np.concatenate((sensory_all,babbling_sensorreads),axis=0)
		activations_all = np.concatenate((activations_all,babbling_activations),axis=0)
		use_prior_model_babbling = True

	Inverse_ANN_models = inverse_mapping_ws_sepANNs_fcn(
	kinematics_all, sensory_all, activations_all, epochs=25, log_address="./log/{}/{}/".format(experiment_ID,run_no), use_sensory=use_sensory, use_prior_model=use_prior_model_babbling) #
	
	errors = []
	for ii in range(number_of_refinements):
		[returned_kinematics, returned_sensorreads, returned_est_activations ] = run_activations_ws_cl_sepANNs_fcn(
			MuJoCo_model_name, attempt_kinematics, log_address="./log/{}/{}/".format(experiment_ID,run_no), timestep=dt, use_sensory=use_sensory, Mj_render=False) # this should be cl
		RMSE = np.sqrt(np.mean(np.square((returned_kinematics[int(returned_kinematics.shape[0]/2):,:8]-attempt_kinematics[int(attempt_kinematics.shape[0]/2):,:8])))) # RMSE on the last half of the trial
		print("Run #:", ii+1)
		print("RMSE:", RMSE)
		errors.append(RMSE)

		kinematics_all = np.concatenate((kinematics_all,returned_kinematics),axis=0)
		sensory_all = np.concatenate((sensory_all,returned_sensorreads),axis=0)
		activations_all = np.concatenate((activations_all,returned_est_activations),axis=0)

		Inverse_ANN_models = inverse_mapping_ws_sepANNs_fcn(
		kinematics_all, sensory_all, activations_all, epochs=5, log_address="./log/{}/{}/".format(experiment_ID,run_no), use_sensory=use_sensory, use_prior_model=True) #
	# test_run (no-training or storing the data)
	[returned_kinematics, returned_sensorreads, returned_est_activations ] = run_activations_ws_cl_sepANNs_fcn(
		MuJoCo_model_name, attempt_kinematics, log_address="./log/{}/{}/".format(experiment_ID,run_no), timestep=dt, use_sensory=use_sensory, Mj_render=False) # this should be cl
	RMSE = np.sqrt(np.mean(np.square((returned_kinematics[int(returned_kinematics.shape[0]/2):,:8]-attempt_kinematics[int(attempt_kinematics.shape[0]/2):,:8]))))
	print("Run #:", number_of_refinements+1)
	print("RMSE:", RMSE)
	errors.append(RMSE)
	return errors, kinematics_all, sensory_all, activations_all

def test_a_task(MuJoCo_model_name, experiment_ID, run_no, use_sensory=True, Mj_render=False):
	refinement_duration_in_seconds = 10
	dt = .01
	attempt_kinematics = create_cyclical_movements_fcn(omega = 3, attempt_length = refinement_duration_in_seconds, timestep = dt)
	[returned_kinematics, returned_sensorreads, returned_est_activations ] = run_activations_ws_cl_sepANNs_fcn(
			MuJoCo_model_name, attempt_kinematics, log_address="./log/{}/{}/".format(experiment_ID,run_no), timestep=dt, use_sensory=use_sensory, Mj_render=Mj_render) # this should be cl
	RMSE = np.sqrt(np.mean(np.square((returned_kinematics[int(returned_kinematics.shape[0]/2):,:8]-attempt_kinematics[int(attempt_kinematics.shape[0]/2):,:8])))) # RMSE on the last half of the trial
	return RMSE

## main code
experiment_ID_base = 'cur3_test_'
all_sensory_cases = [True, False]
for use_sensory in all_sensory_cases:
	np.random.seed(0)
	if use_sensory:
		experiment_ID = experiment_ID_base+"w_sensory"
	else:
		experiment_ID = experiment_ID_base+"wo_sensory"
	number_of_all_runs = 50
	MuJoCo_model_names = ["tendon_quadruped_ws_onair.xml", "tendon_quadruped_ws_onfloor.xml", "tendon_quadruped_ws_onfloorloaded.xml"]
	number_of_refinements = 8
	errors_all = np.zeros((number_of_all_runs, len(MuJoCo_model_names), number_of_refinements+1))
	task_errors = np.zeros((number_of_all_runs, len(MuJoCo_model_names)))
	for run_no in range(number_of_all_runs):
		kinematics_all = sensory_all = activations_all = []
		for MuJoCo_model_name, ii in zip(MuJoCo_model_names, range(len(MuJoCo_model_names))):
			[errors, kinematics_all, sensory_all, activations_all] = babble_and_refine(MuJoCo_model_name, experiment_ID, run_no, kinematics_all, sensory_all, activations_all, number_of_refinements, use_sensory=use_sensory)
			errors_all[run_no,ii,:] = errors
		for MuJoCo_model_name , ii in zip(MuJoCo_model_names, range(len(MuJoCo_model_names))):
			task_errors[run_no, ii] = test_a_task(MuJoCo_model_name, experiment_ID, run_no, use_sensory=use_sensory)
	np.save('./results/{}_babble_and_refine_results'.format(experiment_ID),errors_all)
	np.save('./results/{}_task_results'.format(experiment_ID),task_errors)

#import pdb; pdb.set_trace()
