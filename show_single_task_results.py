import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats
from all_functions import *
from ch5_visu_functions import *

experiment_ID_base = 'cur4_xmlVer11_TD_V4'#mc1 error -> replaced with MC0 copy


curriculums = ["_E2H", "_H2E"]
ANN_structures = ["S","M"]
task_types = ["cyclical", "p2p"]
all_sensory_cases = [0,1]
all_feedback_cases = [0,1]
use_acc = 1
normalize = 1

random_seed = 0
run_no = random_seed

# use_sensory = 0
# use_feedback = 1

# ANN_structure = ANN_structures[1]
task_type = task_types[0]
curriculum = curriculums[0]
number_of_refinements = 6+1
#number_of_all_runs = 60


for use_sensory in all_sensory_cases:
	for use_feedback in all_feedback_cases:
		for ANN_structure in ANN_structures:
			dt=.0025
			actuation_type = "TD"

			Mj_render = 0
			if use_sensory:
				if use_feedback==True:
					experiment_ID = "w_sensory_"+"w_feedback_"+ANN_structure+"_ANN_"+task_type+curriculum
				else:
					experiment_ID = "w_sensory_"+"wo_feedback_"+ANN_structure+"_ANN_"+task_type+curriculum
			else:
				if use_feedback==True:
					experiment_ID = "wo_sensory_"+"w_feedback_"+ANN_structure+"_ANN_"+task_type+curriculum
				else:
					experiment_ID = "wo_sensory_"+"wo_feedback_"+ANN_structure+"_ANN_"+task_type+curriculum
			save_log_path = experiment_ID_base+"/"+experiment_ID
			MuJoCo_model_names = ["tendon_quadruped_ws_inair.xml", "tendon_quadruped_ws_onfloor.xml", "tendon_quadruped_ws_onfloorloaded1000.xml","tendon_quadruped_ws_onfloorloaded2000.xml"]
			for MuJoCo_model_name , ii in zip(MuJoCo_model_names, range(len(MuJoCo_model_names))):
				test_run_RMSE, attempt_kinematics, returned_kinematics=\
					test_a_task(MuJoCo_model_name, save_log_path, run_no, Mj_render=Mj_render, random_seed=random_seed, use_sensory=use_sensory, use_feedback=use_feedback, normalize=normalize, return_kinematics=True, task_type=task_type, ANN_structure=ANN_structure, dt=dt, actuation_type=actuation_type, use_acc=use_acc)
				# np.save('./tmp/test_run_RMSE.npy',test_run_RMSE)
				# np.save('./tmp/attempt_kinematics.npy',attempt_kinematics)
				# np.save('./tmp/returned_kinematics.npy',returned_kinematics)
				# test_run_RMSE=np.load('./tmp/test_run_RMSE.npy')
				# attempt_kinematics=np.load('./tmp/attempt_kinematics.npy')
				# returned_kinematics=np.load('./tmp/returned_kinematics.npy')
				print(MuJoCo_model_name,"RMSE: " ,test_run_RMSE*180/np.pi)
				# import pdb; pdb.set_trace()
				fig_name_base="./results/"+experiment_ID_base+"/figures/"+"rrleg_"+experiment_ID+"_runno_"+str(run_no)+"_"+MuJoCo_model_name[:-4]
				task_plots_fcn(test_run_RMSE, attempt_kinematics, returned_kinematics, dt, fig_name_base, in_degree=1, save_figures=1)
				task_animation_fcn(test_run_RMSE, attempt_kinematics, returned_kinematics, fig_name_base)
			#import pdb; pdb.set_trace()
			# if plot_position_curves:
			# import pdb; pdb.set_trace()
			print("Experiment_ID: "+experiment_ID)