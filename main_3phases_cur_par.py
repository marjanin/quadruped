import os
import pickle
from matplotlib import pyplot as plt
import tensorflow as tf
import multiprocessing as mp
from all_functions import *


## main code
experiment_ID_base = 'cur3_par_test_'
all_sensory_cases = [True, False]
for use_sensory in all_sensory_cases:
	np.random.seed(0)
	if use_sensory:
		experiment_ID = experiment_ID_base+"w_sensory"
	else:
		experiment_ID = experiment_ID_base+"wo_sensory"
	number_of_all_runs = 2
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
