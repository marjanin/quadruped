import os
import pickle
from matplotlib import pyplot as plt
import tensorflow as tf
import multiprocessing as mp
from all_functions import *

def L2_learn_quadruped_experiment(run_no):
	experiment_ID_base = 'cur3_V5_TD_test66'
# Create target Directory if don't exist
	dt=.005
	if not os.path.exists('./results/'+experiment_ID_base):
		os.mkdir('./results/'+experiment_ID_base)
	# all_sensory_cases = [True, False]
	# curriculums = ["_E2H", "_H2E"]
	# ANN_structures = ["S","M"]
	all_sensory_cases = [True, False]
	all_feedback_cases = [True, False]
	use_acc = True
	normalize = True
	curriculums = ["_E2H"]
	ANN_structures = ["S"]
	actuation_type = "TD"
	number_of_refinements = 8
	random_seed=0
	for cur in curriculums:
		for ANN_structure in ANN_structures:
			for use_sensory in all_sensory_cases:
				for use_feedback in all_feedback_cases:
					np.random.seed(random_seed)
					if cur == "_E2H":
						MuJoCo_model_names =\
							["tendon_quadruped_ws_inair.xml",
							"tendon_quadruped_ws_onfloor.xml",
							"tendon_quadruped_ws_onfloorloaded.xml",
							"tendon_quadruped_ws_onfloorloadedheavy.xml"]
					elif cur == "_H2E":
						MuJoCo_model_names =\
							["tendon_quadruped_ws_onfloorloaded.xml",
							"tendon_quadruped_ws_onfloor.xml",
							"tendon_quadruped_ws_inair.xml"]
					task_types = ["cyclical","p2p"]
					for task_type in task_types:
						if use_sensory:
							if use_feedback:
								experiment_ID = "w_sensory_"+"w_feedback_"+ANN_structure+"_ANN_"+task_type+cur
							else:
								experiment_ID = "w_sensory_"+"wo_feedback_"+ANN_structure+"_ANN_"+task_type+cur
						else:
							if use_feedback:
								experiment_ID = "wo_sensory_"+"w_feedback_"+ANN_structure+"_ANN_"+task_type+cur
							else:
								experiment_ID = "wo_sensory_"+"wo_feedback_"+ANN_structure+"_ANN_"+task_type+cur
						learning_errors = np.zeros((len(MuJoCo_model_names), number_of_refinements+2))
						task_errors = np.zeros(len(MuJoCo_model_names))
						kinematics_all = []
						sensory_all = []
						activations_all = []
						for MuJoCo_model_name, ii in zip(MuJoCo_model_names, range(len(MuJoCo_model_names))):
							save_log_path = experiment_ID_base+"/"+experiment_ID
							[errors, kinematics_all, sensory_all, activations_all] = \
								babble_and_refine(
									MuJoCo_model_name,
									save_log_path,
									run_no,
									kinematics_all,
									sensory_all,
									activations_all,
									number_of_refinements,
									random_seed=random_seed,
									use_sensory=use_sensory,
									use_feedback=use_feedback,
									normalize=normalize,
									task_type=task_type,
									ANN_structure=ANN_structure,
									actuation_type=actuation_type,
									use_acc=use_acc,
									dt=dt,)
							if ii ==0:
								learning_errors[ii,1:] = errors
							else:
								learning_errors[ii,:] = errors
						for MuJoCo_model_name , ii in zip(MuJoCo_model_names, range(len(MuJoCo_model_names))):
							task_errors[ii] = test_a_task(MuJoCo_model_name, save_log_path, run_no, random_seed=random_seed, use_sensory=use_sensory, use_feedback=use_feedback, normalize=normalize, plot_position_curves=False, task_type=task_type, ANN_structure=ANN_structure, dt=dt, actuation_type=actuation_type, use_acc=use_acc)
						np.save('./results/{}/MC{}_{}_babble_and_refine_results'.format(experiment_ID_base, run_no, experiment_ID),learning_errors)
						np.save('./results/{}/MC{}_{}_task_results'.format(experiment_ID_base, run_no, experiment_ID),task_errors)
# main code
# pool = mp.Pool(mp.cpu_count())
# print(mp.cpu_count())
# number_of_all_runs = 16+14
# pool.map_async(L2_learn_quadruped_experiment, [run_no for run_no in range(number_of_all_runs)])
# pool.close()
# pool.join()

#import pdb; pdb.set_trace()
L2_learn_quadruped_experiment(0)
