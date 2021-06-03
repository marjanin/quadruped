import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from time import time

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

def task_plots_fcn(test_run_RMSE, attempt_kinematics, returned_kinematics, dt, fig_name_base, in_degree=1, save_figures=1):
	if in_degree:
		unit_exchange_ratio=180./np.pi
		attempt_kinematics=attempt_kinematics*unit_exchange_ratio
		returned_kinematics=returned_kinematics*unit_exchange_ratio
	refinement_duration_in_seconds = 10.
	number_of_all_samples=refinement_duration_in_seconds/dt
	number_of_all_samples_to_plot=number_of_all_samples/2
	
	# figure 1 - temporal plots
	x_axis_plot=np.linspace(0,5,int(number_of_all_samples_to_plot))
	fig1, axes1 = plt.subplots(nrows=2, ncols=1, figsize=(6, 4.2))
	axes1[0].plot(x_axis_plot,attempt_kinematics[int(attempt_kinematics.shape[0]/2):,0], x_axis_plot, returned_kinematics[int(returned_kinematics.shape[0]/2):,0])
	# axes1[0].plot(x_axis_plot, attempt_kinematics[:,6])
	axes1[0].set_title('proximal')
	axes1[0].set_ylabel('angle (degrees)')
	axes1[1].plot(x_axis_plot, attempt_kinematics[int(attempt_kinematics.shape[0]/2):,1], x_axis_plot, returned_kinematics[int(returned_kinematics.shape[0]/2):,1])
	# axes1[1].plot(x_axis_plot, attempt_kinematics[:,7])
	fig1.subplots_adjust(bottom=.145, hspace=0.5)
	axes1[1].set_title('distal')
	axes1[1].set_ylabel('angle (degrees)')
	axes1[1].set_xlabel('time (seconds)\nRMSE (degrees): {:.2f}'.format(test_run_RMSE*180/np.pi))
	# plt.show(block=True)

	# figure 2 - joint space representation
	fig2, axes2 = plt.subplots(nrows=1, ncols=1, figsize=(4.2, 4.2))
	axes2.plot(
		attempt_kinematics[int(attempt_kinematics.shape[0]/2):,0],
		attempt_kinematics[int(attempt_kinematics.shape[0]/2):,1],
		returned_kinematics[int(returned_kinematics.shape[0]/2):,0],
		returned_kinematics[int(attempt_kinematics.shape[0]/2):,1])
	axes2.set_aspect('equal', 'box')
	axes2.set_title('Joint angle space')
	axes2.set_xlabel('proximal\nRMSE (degrees): {:.2f}'.format(test_run_RMSE*180/np.pi))
	axes2.set_ylabel('distal')
	axes2.legend(['desired (degrees)','performed (degrees)'],prop={'size': 8})
	fig2.subplots_adjust(left=.14, bottom=None, right=None, top=.917, wspace=None, hspace=None)
	plt.show(block=0)
	if save_figures:
		dpi = 600
		# fig1.subplots_adjust(left=.06, bottom=.12, right=.96, top=.92, wspace=.30, hspace=.20)
		fig1.savefig(fig_name_base+"_task_figure_1.png", dpi=dpi)
		#fig2.subplots_adjust(bottom=.12, top=.92)
		fig2.savefig(fig_name_base+"_task_figure_2.png", dpi=dpi)
		# plt.show(block=1)
		plt.close(fig1)
		plt.close(fig2)
def task_animation_fcn(test_run_RMSE, attempt_kinematics, returned_kinematics, fig_name_base):
	downsampling_factor=10
	qd0=attempt_kinematics[int(attempt_kinematics.shape[0]/2)::downsampling_factor,6]-(-0.5235987756)+(np.pi/2)+np.pi/32
	qd1=attempt_kinematics[int(attempt_kinematics.shape[0]/2)::downsampling_factor,7]+np.pi/16
	qp0=returned_kinematics[int(returned_kinematics.shape[0]/2)::downsampling_factor,6]-(-0.5235987756)+(np.pi/2)+np.pi/32
	qp1=returned_kinematics[int(returned_kinematics.shape[0]/2)::downsampling_factor,7]+np.pi/16

	t_max=5 #seconds
	fs_original=400

	fs=int(fs_original/downsampling_factor)
	interval=1000*(1/fs)# in ms
	total_frames=t_max*fs
	t=np.linspace(0,t_max,fs)
	l0=.133
	l1=.106
	# w=1
	theta1_offset=0
	# theta0=2*np.pi*w*t
	# theta1=theta1_offset+2*np.pi*w*t
	thetad0=qd0
	thetad1=qd1

	thetap0=qp0
	thetap1=qp1

	xd0=np.cos(thetad0)*l0
	yd0=-np.sin(thetad0)*l0
	xd1=np.cos(thetad0+thetad1)*l1+xd0
	yd1=-np.sin(thetad0+thetad1)*l1+yd0

	xp0=np.cos(thetap0)*l0
	yp0=-np.sin(thetap0)*l0
	xp1=np.cos(thetap0+thetap1)*l1+xp0
	yp1=-np.sin(thetap0+thetap1)*l1+yp0
	#set up figure
	fig = plt.figure(figsize=(5.15, 5.5))
	ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
	                     xlim=(-.1, .15), ylim=(-.25, 0.025))
	ax.set_xlabel('meters\nRMSE (degrees): {:.2f}'.format(test_run_RMSE*180/np.pi))
	ax.set_ylabel('meters')
	ax.grid()
	line1, = ax.plot([], [], 'o-', lw=2)
	line2, = ax.plot([], [], 'o-', lw=2)
	# time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
	# energy_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)

	# animation
	def init():
	    """initialize animation"""
	    line1.set_data([], [])
	    line2.set_data([], [])
	    # time_text.set_text('aa')
	    # energy_text.set_text('bb')
	    return line1,line2#, time_text, energy_text

	def animate(i):
	    """perform animation step"""    
	    line1.set_data([0,-xd0[i],-xd1[i]],[0,yd0[i],yd1[i]])
	    line2.set_data([0,-xp0[i],-xp1[i]],[0,yp0[i],yp1[i]])

	#    line.set_data([x0[i],x1[i]],[y0[i],y1[i]])
	    # time_text.set_text('time = %.1f' % pendulum.time_elapsed)
	    # energy_text.set_text('energy = %.3f J' % pendulum.energy())
	    return line1,line2#, time_text, energy_text

	# animate(0)
	ani = animation.FuncAnimation(fig, animate, frames=total_frames,
	                              interval=10, blit=False, init_func=init)
	ani.save(fig_name_base+"_task_video.mp4", fps=fs*0.5) # fps=fs*0.5: 0.5 x speed
	# plt.show()
	plt.close(fig)