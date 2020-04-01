import numpy as np
from matplotlib import pyplot as plt
from all_functions import *

def test_a_task(MuJoCo_model_name, experiment_ID, run_no, use_sensory=True, Mj_render=False):
	refinement_duration_in_seconds = 10
	dt = .01
	attempt_kinematics = create_cyclical_movements_fcn(omega = 3, attempt_length = refinement_duration_in_seconds, timestep = dt)
	[returned_kinematics, returned_sensorreads, returned_est_activations ] = run_activations_ws_cl_sepANNs_fcn(
			MuJoCo_model_name, attempt_kinematics, log_address="./log/{}/{}/".format(experiment_ID,run_no), timestep=dt, use_sensory=use_sensory, Mj_render=Mj_render) # this should be cl
	RMSE = np.sqrt(np.mean(np.square((returned_kinematics[int(returned_kinematics.shape[0]/2):,:8]-attempt_kinematics[int(attempt_kinematics.shape[0]/2):,:8])))) # RMSE on the last half of the trial
	return RMSE

fig1, axes1 = plt.subplots(nrows=1, ncols=3, figsize=(9, 3))
fig2, axes2 = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))
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
		axes1[ii].set_title('Task RMSE')
		axes1[ii].set_xlabel('Task #')
		axes1[ii].set_ylabel('RMSE')


	# axes1[1,1].errorbar(x=np.arange(3), y=task_errors_mean, yerr=task_errors_std, capsize=2, animated=True, alpha=.3, color=color)
	# axes1[1,1].plot(np.arange(3), task_errors_mean,'--', alpha=.7, color=color)
	# axes1[1,1].set_title('Task RMSE')
	# axes1[1,1].set_xlabel('Task #')
	# axes1[1,1].set_ylabel('RMSE')
	
	axes2.errorbar(x=np.arange(3), y=task_errors_mean, yerr=task_errors_std, capsize=2, animated=True, alpha=.3, color=color)
	axes2.plot(np.arange(3), task_errors_mean,'--', alpha=.7, color=color)
	axes2.set_title('Task RMSE')
	axes2.set_xlabel('Task #')
	axes2.set_ylabel('RMSE')

show_video = True
use_sensory = False
experiment_ID = experiment_ID_base+"wo_sensory"
run_no = 0
MuJoCo_model_names = ["tendon_quadruped_ws_onair.xml", "tendon_quadruped_ws_onfloor.xml", "tendon_quadruped_ws_onfloorloaded.xml"]
if show_video:
	for MuJoCo_model_name , ii in zip(MuJoCo_model_names, range(len(MuJoCo_model_names))):
		tmp = test_a_task(MuJoCo_model_name, experiment_ID, run_no, use_sensory=use_sensory,  Mj_render=True)

plt.show(block=True)
#import pdb; pdb.set_trace()


# y=np.mean(all_performances,0)
# yerr=np.std(all_performances,0)
# #import pdb; pdb.set_trace()
# axes1.errorbar(x=x, y=y, yerr=yerr, capsize=2, animated=True, alpha=.3, color='C2')
# axes1.plot(x, y,'--', alpha=.7, color='C0')
# axes1.set_title('Error vs. run')
# axes1.set_xlabel('run #')
# axes1.set_ylabel('Error')
# plt.plot()

# experiment_ID = 'exp2'
# all_performances = np.load('./results/{}_results.npy'.format(experiment_ID))
# y=np.mean(all_performances,0)
# yerr=np.std(all_performances,0)
# #import pdb; pdb.set_trace()
# axes1.errorbar(x=x, y=y, yerr=yerr, capsize=2, animated=True, alpha=.3, color='C3')
# axes1.plot(x, y,'--', alpha=.7, color='C1')
# axes1.set_title('Error vs. run')
# axes1.set_xlabel('run #')
# axes1.set_ylabel('Error')
# plt.plot()

#plt.show(block=True)
# w
# np.mean(task_errors,axis=0)                                    │
# array([0.14201016, 0.37790977, 0.45373478]) 
# wo
# np.mean(task_errors,axis=0)                                    │
# array([0.17151521, 0.45107131, 0.51196521]) 