import numpy as np
from matplotlib import pyplot as plt
from all_functions import *

experiment_ID_base = 'cur3_V5_TD_test44'
show_video = True

all_sensory_cases = [True, False]
use_feedback =True
use_acc=False
normalize=True
cur = "_E2H"
ANN_structure = "S"
actuation_type = "TD"
task_type = "cyclical"
number_of_refinements = 4+1
number_of_all_runs = 1

fig1, axes1 = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
fig2, axes2 = plt.subplots(nrows=1, ncols=1, figsize=(6, 4.2))

for use_sensory in all_sensory_cases:
	if use_sensory:
		linestyle = ".:"
		hatch = ""
		color = "C0"
		if use_feedback:
			experiment_ID = "w_sensory_"+"w_feedback_"+ANN_structure+"_ANN_"+task_type+cur
		else:
			experiment_ID = "w_sensory_"+"wo_feedback_"+ANN_structure+"_ANN_"+task_type+cur
	else:
		linestyle = ".:"
		hatch = "//"
		color= "C1"
		if use_feedback:
			experiment_ID = "wo_sensory_"+"w_feedback_"+ANN_structure+"_ANN_"+task_type+cur
		else:
			experiment_ID = "wo_sensory_"+"wo_feedback_"+ANN_structure+"_ANN_"+task_type+cur

	if cur == "_E2H":
		MuJoCo_model_names = ["tendon_quadruped_ws_inair.xml", "tendon_quadruped_ws_onfloor.xml", "tendon_quadruped_ws_onfloorloaded.xml"]
		MuJoCo_model_names_short = ["In Air", "On Floor", "On Floor With Load"]
	elif cur == "_H2E":
		MuJoCo_model_names = ["tendon_quadruped_ws_onfloorloaded.xml", "tendon_quadruped_ws_onfloor.xml", "tendon_quadruped_ws_inair.xml"]
		MuJoCo_model_names_short = ["On Floor with Load", "On Floor", "In Air"]
	else:
		ValueError("unacceptable curriculum")

	# save_log_path = experiment_ID_base+"/"+experiment_ID
	learning_errors_all = np.zeros((number_of_all_runs, len(MuJoCo_model_names), number_of_refinements+1))
	task_errors_all = np.zeros((number_of_all_runs, len(MuJoCo_model_names)))

	for run_no in range(number_of_all_runs):
		learning_errors_all[run_no,:,:]=np.load('./results/{}/MC{}_{}_babble_and_refine_results.npy'.format(experiment_ID_base, run_no, experiment_ID))
		task_errors_all[run_no,:]=np.load('./results/{}/MC{}_{}_task_results.npy'.format(experiment_ID_base, run_no, experiment_ID))
	# errors_all = learning_errors_all
	# task_errors_all = task_errors_all_all

	learning_errors_all_mean = np.mean(learning_errors_all, axis=0)
	learning_errors_all_std = np.std(learning_errors_all, axis=0)
	task_errors_all_mean = np.mean(task_errors_all, axis=0)
	task_errors_all_std = np.std(task_errors_all, axis=0)

	for ii in range(3):
		if ii == 0:
			#import pdb; pdb.set_trace()
			axes1[ii].errorbar(x=np.arange(1,number_of_refinements+1), y=learning_errors_all_mean[ii,1:], yerr=learning_errors_all_std[ii,1:], capsize=2, animated=True, alpha=.4, color=color)
			axes1[ii].plot(np.arange(1,number_of_refinements+1), learning_errors_all_mean[ii,1:],linestyle, alpha=.8, color=color)
		else:
			axes1[ii].errorbar(x=np.arange(number_of_refinements+1), y=learning_errors_all_mean[ii], yerr=learning_errors_all_std[ii], capsize=2, animated=True, alpha=.4, color=color)
			if use_sensory:
				line1, = axes1[ii].plot(np.arange(number_of_refinements+1), learning_errors_all_mean[ii],linestyle, alpha=.8, color=color)
			else:
				line2, = axes1[ii].plot(np.arange(number_of_refinements+1), learning_errors_all_mean[ii],linestyle, alpha=.8, color=color)

		line3 = axes1[ii].axvline(x=.5, color='r', linestyle='dashdot', linewidth=1.5, alpha=.25)
		axes1[ii].set_title(MuJoCo_model_names_short[ii])
		axes1[ii].set_xlabel('Refinement #')
		axes1[ii].set_ylabel('RMSE')
		axes1[ii].set_ylim(0.0, 1)
		axes1[ii].set_xlim(-0.50, 9.5)
		axes1[ii].grid(color='k', linestyle=':', linewidth=.5)

	fig1.suptitle('RMSE vs. Refinement # (learning)', fontsize=16)
	fig1.subplots_adjust(left=0.06, bottom=0.12, right=.95, top=.85, wspace=.25, hspace=.20)
	if use_sensory:
		x_shift=.35
	else:
		x_shift=0
	# for ii in range(3):
	# 	axes2.errorbar(ii+x_shift, y=task_errors_all_mean[ii], yerr=task_errors_all_std[ii], capsize=2, animated=True, alpha=.3, color=color)
	# 	axes2.bar(ii+x_shift, task_errors_all_mean[ii], width=0.45,alpha=.6, color=color, hatch=hatch)
	# 	axes2.set_title('RMSE vs. Task (test)')
	# 	axes2.set_xlabel('Task')
	# 	axes2.set_ylabel('RMSE')
	# 	axes2.grid(color='k', linestyle=':', linewidth=.5)
	# fig2.subplots_adjust(bottom=0.12, top=.92)
	positions=np.arange(3)+x_shift
	axes2.boxplot(task_errors_all, positions=positions)
axes2.set_title('RMSE vs. Task (test)')
axes2.set_xlabel('Task')
axes2.set_xticklabels(["w. sen.","w. sen.","w. sen.","w/o. sen.","w/o. sen.","w/o. sen."])
# Rotate the tick labels and set their alignment.
plt.setp(axes2.get_xticklabels(), rotation=-45, ha="left",
         rotation_mode="anchor")
axes2.set_ylabel('RMSE')
axes2.grid(color='k', linestyle=':', linewidth=.5)
axes2.set_ylim(0.0, .7)
fig2.subplots_adjust(bottom=0.15, top=.92)
axes1[2].legend((line1,line2),('with sensory','without sensory'))


save_figures = True
if save_figures:
	dpi = 600
	# fig1.subplots_adjust(left=.06, bottom=.12, right=.96, top=.92, wspace=.30, hspace=.20)
	fig1.savefig("./results/{}/{}_figure1.png".format(experiment_ID_base,experiment_ID), dpi=dpi)
	#fig2.subplots_adjust(bottom=.12, top=.92)
	fig2.savefig("./results/{}/{}_figure2.png".format(experiment_ID_base,experiment_ID), dpi=dpi)
plt.show(block=True)

if show_video:
	dt=.005
	use_sensory = True
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
	save_log_path = experiment_ID_base+"/"+experiment_ID
	run_no = 0
	MuJoCo_model_names = ["tendon_quadruped_ws_inair.xml", "tendon_quadruped_ws_onfloor.xml", "tendon_quadruped_ws_onfloorloaded.xml"]
	for MuJoCo_model_name , ii in zip(MuJoCo_model_names, range(len(MuJoCo_model_names))):
		test_run_RMSE = test_a_task(MuJoCo_model_name, save_log_path, run_no, Mj_render=False, use_sensory=use_sensory, use_feedback=use_feedback, normalize=normalize, plot_position_curves=True, task_type=task_type, ANN_structure=ANN_structure, dt=dt, actuation_type=actuation_type, use_acc=use_acc)
		print(MuJoCo_model_name,"RMSE: " ,test_run_RMSE)
#import pdb; pdb.set_trace()
