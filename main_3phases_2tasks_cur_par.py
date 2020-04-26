import os
import pickle
from matplotlib import pyplot as plt
import tensorflow as tf
import multiprocessing as mp
from all_functions import *


def L2_learn_quadruped_experiment(run_no):
	experiment_ID_base = 'cur3_V3_3_test_'
# Create target Directory if don't exist
	if not os.path.exists('./results/'+experiment_ID_base):
		os.mkdir('./results/'+experiment_ID_base)
	all_sensory_cases = [True, False]
	for use_sensory in all_sensory_cases:
		np.random.seed(run_no)
		if use_sensory:
			experiment_ID_base2 = experiment_ID_base+"w_sensory_"
		else:
			experiment_ID_base2 = experiment_ID_base+"wo_Sensory_"
		# MuJoCo_model_names =\
		# 	["tendon_quadruped_ws_onair.xml",
		# 	"tendon_quadruped_ws_onfloor.xml",
		# 	"tendon_quadruped_ws_onfloorloaded.xml"]

		MuJoCo_model_names =\
			["tendon_quadruped_ws_onfloorloaded.xml",
			"tendon_quadruped_ws_onfloor.xml",
			"tendon_quadruped_ws_inair.xml"]
		task_types = ["cyclical", "p2p"]
		number_of_refinements = 8
		for task_type in task_types:
			experiment_ID = experiment_ID_base2+task_type
			learning_errors = np.zeros((len(MuJoCo_model_names), number_of_refinements+1))
			task_errors = np.zeros(len(MuJoCo_model_names))
			kinematics_all = sensory_all = activations_all = []
			for MuJoCo_model_name, ii in zip(MuJoCo_model_names, range(len(MuJoCo_model_names))):
				[errors, kinematics_all, sensory_all, activations_all] = \
					babble_and_refine(
						MuJoCo_model_name,
						experiment_ID,
						run_no,
						kinematics_all,
						sensory_all,
						activations_all,
						number_of_refinements,
						use_sensory=use_sensory,
						task_type=task_type)
				learning_errors[ii,:] = errors
			for MuJoCo_model_name , ii in zip(MuJoCo_model_names, range(len(MuJoCo_model_names))):
				task_errors[ii] = test_a_task(MuJoCo_model_name, experiment_ID, run_no, use_sensory=use_sensory, task_type=task_type)
			np.save('./results/{}/MC{}_{}_babble_and_refine_results'.format(experiment_ID_base, run_no, experiment_ID),learning_errors)
			np.save('./results/{}/MC{}_{}_task_results'.format(experiment_ID_base, run_no, experiment_ID),task_errors)
## main code
pool = mp.Pool(mp.cpu_count())
print(mp.cpu_count())
number_of_all_runs = 50
pool.map_async(L2_learn_quadruped_experiment, [run_no for run_no in range(number_of_all_runs)])
pool.close()
pool.join()

#import pdb; pdb.set_trace()
#L2_learn_quadruped_experiment(0)