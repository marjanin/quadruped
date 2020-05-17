import os
import pickle
from matplotlib import pyplot as plt
import tensorflow as tf
import multiprocessing as mp
from all_functions import *

def L2_learn_quadruped_experiment(run_no):
	experiment_ID_base = 'cur3_V5_TD_test_1'
# Create target Directory if don't exist
	dt=.005
	if not os.path.exists('./results/'+experiment_ID_base):
		os.mkdir('./results/'+experiment_ID_base)
	all_sensory_cases = [True, False]
	curriculums = ["_E2H", "_H2E"]
	ANN_structures = ["S","M"]
	actuation_type = "TD"
	number_of_refinements = 8
	for cur in curriculums:
		for ANN_structure in ANN_structures:
			for use_sensory in all_sensory_cases:
				np.random.seed(run_no)
				if cur == "_E2H":
					MuJoCo_model_names =\
						["tendon_quadruped_ws_inair.xml",
						"tendon_quadruped_ws_onfloor.xml",
						"tendon_quadruped_ws_onfloorloaded.xml"]
				elif cur == "_H2E":
					MuJoCo_model_names =\
						["tendon_quadruped_ws_onfloorloaded.xml",
						"tendon_quadruped_ws_onfloor.xml",
						"tendon_quadruped_ws_inair.xml"]
				task_types = ["cyclical", "p2p"]
				for task_type in task_types:
					if use_sensory:
						experiment_ID = "w_sensory_"+ANN_structure+"_ANN_"+task_type+cur
					else:
						experiment_ID = "wo_Sensory_"+ANN_structure+"_ANN_"+task_type+cur
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
								use_sensory=use_sensory,
								task_type=task_type,
								ANN_structure=ANN_structure,
								actuation_type=actuation_type,
								dt=dt)
						if ii ==0:
							learning_errors[ii,1:] = errors
						else:
							learning_errors[ii,:] = errors
					for MuJoCo_model_name , ii in zip(MuJoCo_model_names, range(len(MuJoCo_model_names))):
						task_errors[ii] = test_a_task(MuJoCo_model_name, save_log_path, run_no, use_sensory=use_sensory, task_type=task_type, ANN_structure=ANN_structure, dt=dt, actuation_type=actuation_type)
					np.save('./results/{}/MC{}_{}_babble_and_refine_results'.format(experiment_ID_base, run_no, experiment_ID),learning_errors)
					np.save('./results/{}/MC{}_{}_task_results'.format(experiment_ID_base, run_no, experiment_ID),task_errors)
# main code
pool = mp.Pool(mp.cpu_count())
print(mp.cpu_count())
number_of_all_runs = 50
pool.map_async(L2_learn_quadruped_experiment, [run_no for run_no in range(16,number_of_all_runs)])
pool.close()
pool.join()

#import pdb; pdb.set_trace()
#L2_learn_quadruped_experiment(0)
