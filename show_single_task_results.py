import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats
from all_functions import *

experiment_ID_base = 'cur4_xmlVer11_TD_V4'#mc1 error -> replaced with MC0 copy


curriculums = ["_E2H", "_H2E"]
ANN_structures = ["S","M"]
task_types = ["cyclical", "p2p"]

all_sensory_cases = [True, False]
use_feedback = True
use_acc = True
normalize = True


Mj_render = True
random_seed = 0

task_type = task_types[1]
curriculum = curriculums[0]
ANN_structure = ANN_structures[1]
number_of_refinements = 6+1
number_of_all_runs = 60

dt=.0025
use_sensory = True
actuation_type = "TD"

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
run_no = 0
MuJoCo_model_names = ["tendon_quadruped_ws_inair.xml", "tendon_quadruped_ws_onfloor.xml", "tendon_quadruped_ws_onfloorloaded1000.xml","tendon_quadruped_ws_onfloorloaded2000.xml"]
if show_video:
	for MuJoCo_model_name , ii in zip(MuJoCo_model_names, range(len(MuJoCo_model_names))):
		test_run_RMSE = test_a_task(MuJoCo_model_name, save_log_path, run_no, Mj_render=Mj_render, random_seed=random_seed, use_sensory=use_sensory, use_feedback=use_feedback, normalize=normalize, plot_position_curves=True, task_type=task_type, ANN_structure=ANN_structure, dt=dt, actuation_type=actuation_type, use_acc=use_acc)
		print(MuJoCo_model_name,"RMSE: " ,test_run_RMSE)

#import pdb; pdb.set_trace()

