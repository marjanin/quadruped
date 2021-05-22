import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt

def loading_plotting_data_fcn(experiment_ID_base, use_sensory, use_feedback, curriculum, task_type, ANN_structure, number_of_refinements, number_of_all_runs):
	if use_sensory:
		linestyle = ".:"
		hatch = ""
		if use_feedback==True:
			experiment_ID = "w_sensory_"+"w_feedback_"+ANN_structure+"_ANN_"+task_type+curriculum
		else:
			experiment_ID = "w_sensory_"+"wo_feedback_"+ANN_structure+"_ANN_"+task_type+curriculum
	else:
		linestyle = ".:"
		hatch = "//"
		if use_feedback==True:
			experiment_ID = "wo_sensory_"+"w_feedback_"+ANN_structure+"_ANN_"+task_type+curriculum
		else:
			experiment_ID = "wo_sensory_"+"wo_feedback_"+ANN_structure+"_ANN_"+task_type+curriculum

	if curriculum == "_E2H":
		MuJoCo_model_names = ["tendon_quadruped_ws_inair.xml", "tendon_quadruped_ws_onfloor.xml", "tendon_quadruped_ws_onfloorloaded1000.xml","tendon_quadruped_ws_onfloorloaded3000.xml"]
		MuJoCo_model_names_short = ["In Air", "On Floor", "On Floor With Load",  "On Floor With Heavy Load"]
	elif curriculum == "_H2E":
		MuJoCo_model_names = ["tendon_quadruped_ws_onfloorloaded3000.xml","tendon_quadruped_ws_onfloorloaded1000.xml", "tendon_quadruped_ws_onfloor.xml", "tendon_quadruped_ws_inair.xml"]
		MuJoCo_model_names_short = ["On Floor With Heavy Load", "On Floor with Load", "On Floor", "In Air"]
	else:
		ValueError("unacceptable curriculum")

	# save_log_path = experiment_ID_base+"/"+experiment_ID
	learning_errors_all = np.zeros((number_of_all_runs, len(MuJoCo_model_names), number_of_refinements+1))
	task_errors_all = np.zeros((number_of_all_runs, len(MuJoCo_model_names)))
	for run_no in range(number_of_all_runs):
		learning_errors_all[run_no,:,:]=np.load('./results/{}/MC{}_{}_babble_and_refine_results.npy'.format(experiment_ID_base, run_no, experiment_ID))
		task_errors_all[run_no,:]=np.load('./results/{}/MC{}_{}_task_results.npy'.format(experiment_ID_base, run_no, experiment_ID))
	return learning_errors_all, task_errors_all

def compare_learning_error_plots_fcn(learning_errors_all_1, learning_errors_all_2, labels, curriculum, in_degree=True):
	if in_degree:
		unit_exchange_ratio=180/np.pi
	else:
		unit_exchange_ratio=1
	learning_errors_all_1=unit_exchange_ratio*learning_errors_all_1
	learning_errors_all_2=unit_exchange_ratio*learning_errors_all_2

	number_of_refinements = learning_errors_all_1.shape[2]-1
	fig1, axes1 = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))
	color_1 = [31./255, 119./255, 180./255]#"C0"
	color_2 = [255./255, 128./255,14./255]#"C1"
	linestyle = ".:"
	hatch = ""
	learning_errors_all_mean_1 = np.mean(learning_errors_all_1, axis=0)
	learning_errors_all_std_1 = np.std(learning_errors_all_1, axis=0)
	learning_errors_all_mean_2 = np.mean(learning_errors_all_2, axis=0)
	learning_errors_all_std_2 = np.std(learning_errors_all_2, axis=0)

	if curriculum == "_E2H":
		MuJoCo_model_names = ["tendon_quadruped_ws_inair.xml", "tendon_quadruped_ws_onfloor.xml", "tendon_quadruped_ws_onfloorloaded1000.xml","tendon_quadruped_ws_onfloorloaded3000.xml"]
		MuJoCo_model_names_short = ["In Air", "On Floor", "On Floor With Load",  "On Floor With Heavy Load"]
	elif curriculum == "_H2E":
		MuJoCo_model_names = ["tendon_quadruped_ws_onfloorloaded3000.xml","tendon_quadruped_ws_onfloorloaded1000.xml", "tendon_quadruped_ws_onfloor.xml", "tendon_quadruped_ws_inair.xml"]
		MuJoCo_model_names_short = ["On Floor With Heavy Load", "On Floor with Load", "On Floor", "In Air"]
	else:
		ValueError("unacceptable curriculum")

	for ii in range(4):
		if ii == 0:
			#import pdb; pdb.set_trace()
			axes1[ii].errorbar(x=np.arange(1,number_of_refinements+1), y=learning_errors_all_mean_1[ii,1:], yerr=learning_errors_all_std_1[ii,1:], capsize=2, animated=True, alpha=.4, color=color_1)
			axes1[ii].plot(np.arange(1,number_of_refinements+1), learning_errors_all_mean_1[ii,1:],linestyle, alpha=.8, color=color_1)
			axes1[ii].errorbar(x=np.arange(1,number_of_refinements+1), y=learning_errors_all_mean_2[ii,1:], yerr=learning_errors_all_std_2[ii,1:], capsize=2, animated=True, alpha=.4, color=color_2)
			axes1[ii].plot(np.arange(1,number_of_refinements+1), learning_errors_all_mean_2[ii,1:],linestyle, alpha=.8, color=color_2)
			if in_degree:
				axes1[ii].set_ylabel('angle (degrees)')
			else:
				axes1[ii].set_ylabel('RMSE')
		else:
			axes1[ii].errorbar(x=np.arange(number_of_refinements+1), y=learning_errors_all_mean_1[ii], yerr=learning_errors_all_std_1[ii], capsize=2, animated=True, alpha=.4, color=color_1)
			line1, = axes1[ii].plot(np.arange(number_of_refinements+1), learning_errors_all_mean_1[ii],linestyle, alpha=.8, color=color_1)
			axes1[ii].errorbar(x=np.arange(number_of_refinements+1), y=learning_errors_all_mean_2[ii], yerr=learning_errors_all_std_2[ii], capsize=2, animated=True, alpha=.4, color=color_2)
			line2, = axes1[ii].plot(np.arange(number_of_refinements+1), learning_errors_all_mean_2[ii],linestyle, alpha=.8, color=color_2)
		line3 = axes1[ii].axvline(x=.5, color='r', linestyle='dashdot', linewidth=1.5, alpha=.25)
		axes1[ii].set_title(MuJoCo_model_names_short[ii])
		axes1[ii].set_xlabel('Refinement #')
		axes1[ii].set_ylim(0, 0.25*unit_exchange_ratio)
		axes1[ii].set_xlim(-0.50, 9.5)
		# axes1[ii].grid(color='k', linestyle=':', linewidth=.5)
	fig1.suptitle('position error vs. Refinement # (forward learning)', fontsize=16)
	fig1.subplots_adjust(left=0.06, bottom=0.12, right=.95, top=.85, wspace=.25, hspace=.20)
	axes1[3].legend((line1,line2),(labels[0],labels[1]))
	return fig1

def compare_task_error_plots_fcn(task_errors_all_1, task_errors_all_2, labels, in_degree=True):
	if in_degree:
		unit_exchange_ratio=180/np.pi
	else:
		unit_exchange_ratio=1
	task_errors_all_1=unit_exchange_ratio*task_errors_all_1
	task_errors_all_2=unit_exchange_ratio*task_errors_all_2

	fig2, axes2 = plt.subplots(nrows=1, ncols=1, figsize=(6, 5.1))
	x_shift_1=0.0
	x_shift_2=0.35
	positions_1=np.arange(4)+x_shift_1
	positions_2=np.arange(4)+x_shift_2

	task_errors_all_mean_1 = np.mean(task_errors_all_1, axis=0)
	task_errors_all_std_1 = np.std(task_errors_all_1, axis=0)
	task_errors_all_mean_2 = np.mean(task_errors_all_2, axis=0)
	task_errors_all_std_2 = np.std(task_errors_all_2, axis=0)

	axes2.boxplot(task_errors_all_1, positions=positions_1)
	axes2.boxplot(task_errors_all_2, positions=positions_2)
	stat_sig_star_positions=np.arange(4)+x_shift_2/2
	F_values=np.zeros(4)
	p_values=np.zeros(4)
	for ii in range(4):
		F_values[ii],p_values[ii]=stats.f_oneway(task_errors_all_2[:,ii],task_errors_all_1[:,ii])
		if p_values[ii]<0.01:
			axes2.plot([stat_sig_star_positions[ii]],[.205*unit_exchange_ratio],'k*')
		if p_values[ii]<0.05:
			axes2.plot([stat_sig_star_positions[ii]],[.2*unit_exchange_ratio],'k*')
	print(p_values)
	axes2.set_title('position error vs. test case (backward generalization)')
	axes2.set_xlabel('test case')
	axes2.set_xticklabels([labels[0], labels[0], labels[0], labels[0], labels[1], labels[1], labels[1], labels[1]])
	# Rotate the tick labels and set their alignment.
	plt.setp(axes2.get_xticklabels(), rotation=-45, ha="left",
	         rotation_mode="anchor")
	if in_degree:
		axes2.set_ylabel('angle (degrees)')
	else:
		axes2.set_ylabel('RMSE')
	axes2.grid(color='k', linestyle=':', linewidth=.5)
	axes2.set_ylim(0, 0.25*unit_exchange_ratio)
	fig2.subplots_adjust(bottom=0.15, top=.92)
	return fig2