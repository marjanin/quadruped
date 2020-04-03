import numpy as np
from matplotlib import pyplot as plt
from all_functions import *

fig1, axes1 = plt.subplots(nrows=1, ncols=3, figsize=(12, 3.5))
fig2, axes2 = plt.subplots(nrows=1, ncols=1, figsize=(6, 3.5))
experiment_ID_base = 'cur3_1_'
all_sensory_cases = [True, False]
for use_sensory in all_sensory_cases:
	np.random.seed(0)
	if use_sensory:
		color='C0'
		experiment_ID = experiment_ID_base+"w_sensory"
	else:
		experiment_ID = experiment_ID_base+"wo_sensory"
		color="C1"
	errors_all = np.load('./results/{}_babble_and_refine_results.npy'.format(experiment_ID))
	task_errors = np.load('./results/{}_task_results.npy'.format(experiment_ID))
	errors_all_mean = np.mean(errors_all, axis=0)
	errors_all_std = np.std(errors_all, axis=0)
	task_errors_mean = np.mean(task_errors, axis=0)
	task_errors_std = np.std(task_errors, axis=0)

	for ii in range(3):
		axes1[ii].errorbar(x=np.arange(9), y=errors_all_mean[ii], yerr=errors_all_std[ii], capsize=2, animated=True, alpha=.3, color=color)
		axes1[ii].plot(np.arange(9), errors_all_mean[ii],'--', alpha=.7, color=color)
		if ii == 1:
			axes1[ii].set_title('RMSE vs. Refinement # (learning)')
		axes1[ii].set_xlabel('Refinement #')
		axes1[ii].set_ylabel('RMSE')
		
	if use_sensory:
		x_shift=.25
	else:
		x_shift=0
	for ii in range(3):
		axes2.errorbar(ii+x_shift, y=task_errors_mean[ii], yerr=task_errors_std[ii], capsize=2, animated=True, alpha=.3, color=color)
		axes2.bar(ii+x_shift, task_errors_mean[ii], width=0.45,alpha=.6, color=color)
		axes2.set_title('RMSE vs. Task (test)')
		axes2.set_xlabel('Task')
		axes2.set_ylabel('RMSE')

save_figures = False
if save_figures:
	dpi = 600
	fig1.subplots_adjust(left=.06, bottom=.12, right=.96, top=.92, wspace=.30, hspace=.20)
	fig1.savefig("./results/{}_figure1.png".format(experiment_ID_base), dpi=dpi)
	fig2.subplots_adjust(bottom=.12, top=.92)
	fig2.savefig("./results/{}_figure2.png".format(experiment_ID_base), dpi=dpi)

show_video = False
#use_sensory = False
experiment_ID = experiment_ID_base+"wo_sensory"
run_no = 0
MuJoCo_model_names = ["tendon_quadruped_ws_onair.xml", "tendon_quadruped_ws_onfloor.xml", "tendon_quadruped_ws_onfloorloaded.xml"]
if show_video:
	for MuJoCo_model_name , ii in zip(MuJoCo_model_names, range(len(MuJoCo_model_names))):
		tmp = test_a_task(MuJoCo_model_name, experiment_ID, run_no, use_sensory=use_sensory,  Mj_render=True)
plt.show(block=True)


#import pdb; pdb.set_trace()