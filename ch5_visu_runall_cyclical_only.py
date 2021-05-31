import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats
import time
from os import mkdir, path
from ch5_visu_functions import *
#from all_functions import *
experiment_ID_base = 'cur4_xmlVer11_TD_V4'
if not path.exists("./results/{}/figures/".format(experiment_ID_base)):
	mkdir("./results/{}/figures/".format(experiment_ID_base))
number_of_refinements = 6+1
number_of_all_runs = 64
curriculum="_E2H"

all_cases={
"ANN_structures":["S","M"],
"feedback_cases":[0,1],
"tactile_cases":[0,1]}

case_to_run_1={
"ANN_structures":[],
"feedback_cases":[],
"tactile_cases":[]}

case_to_run_2={
"ANN_structures":[],
"feedback_cases":[],
"tactile_cases":[]}

case_to_run_name={
"ANN_structures":[],
"feedback_cases":[],
"tactile_cases":[]}
#import pdb; pdb.set_trace()	
for current_key, current_value in all_cases.items():
	other_keys=[]
	case_to_run_1[current_key]=current_value[0]
	case_to_run_2[current_key]=current_value[1]
	if current_key == "feedback_cases":
		labels=["w/o fb", "w fb"]
	elif current_key == "tactile_cases":
		labels=["w/o tac.", "w tac."]
	elif current_key == "task_types":
		labels=["cycl", "p2p"]
	elif current_key == "ANN_structures":
		labels=["sin.", "mul."]
	case_to_run_name[current_key]="var"
	for key, value in all_cases.items():
		if key != current_key:
			other_keys.append(key)
	for item_1 in all_cases[other_keys[0]]:
		for item_2 in all_cases[other_keys[1]]:
			case_to_run_1[other_keys[0]]=item_1
			case_to_run_1[other_keys[1]]=item_2
			case_to_run_2[other_keys[0]]=item_1
			case_to_run_2[other_keys[1]]=item_2
			case_to_run_name[other_keys[0]]=str(item_1)
			case_to_run_name[other_keys[1]]=str(item_2)

			comparison_name="task_{}_cur_{}_stru_{}_fb_{}_sensory_{}".format(
			"cyclical",
			curriculum,case_to_run_name["ANN_structures"],
			case_to_run_name["feedback_cases"],
			case_to_run_name["tactile_cases"])
			
			[learning_errors_all_1, task_errors_all_1] =\
			loading_plotting_data_fcn(
				experiment_ID_base=experiment_ID_base,
				use_sensory=case_to_run_1["tactile_cases"],
				use_feedback=case_to_run_1["feedback_cases"],
				curriculum=curriculum,
				task_type="cyclical",
				ANN_structure=case_to_run_1["ANN_structures"],
				number_of_refinements=number_of_refinements,
				number_of_all_runs=number_of_all_runs)
			[learning_errors_all_2, task_errors_all_2] =\
			loading_plotting_data_fcn(
				experiment_ID_base=experiment_ID_base,
				use_sensory=case_to_run_2["tactile_cases"],
				use_feedback=case_to_run_2["feedback_cases"],
				curriculum=curriculum,
				task_type="cyclical",
				ANN_structure=case_to_run_2["ANN_structures"],
				number_of_refinements=number_of_refinements,
				number_of_all_runs=number_of_all_runs)
			# import pdb; pdb.set_trace()
			fig1=compare_learning_error_plots_fcn(learning_errors_all_1, learning_errors_all_2, labels, curriculum="_E2H")
			fig2=compare_task_error_plots_fcn(task_errors_all_1, task_errors_all_2, labels)
			#import pdb; pdb.set_trace()
			# plt.show(block=1)
			save_figures = True
			if save_figures:
				dpi = 150
				# fig1.subplots_adjust(left=.06, bottom=.12, right=.96, top=.92, wspace=.30, hspace=.20)
				fig1.savefig("./results/{}/figures/{}_figure1.png".format(experiment_ID_base,comparison_name), dpi=dpi)
				#fig2.subplots_adjust(bottom=.12, top=.92)
				fig2.savefig("./results/{}/figures/{}_figure2.png".format(experiment_ID_base,comparison_name), dpi=dpi)
				# plt.show(block=1)
				plt.close(fig1)
				plt.close(fig2)
#import pdb; pdb.set_trace()
