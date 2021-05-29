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
number_of_all_runs = 32
curriculum="_E2H"
comparison_name="test_fig"
labels=["worst","best"]

[learning_errors_all_1, task_errors_all_1] =\
loading_plotting_data_fcn(
	experiment_ID_base=experiment_ID_base,
	use_sensory=0,
	use_feedback=0,
	curriculum=curriculum,
	task_type="cyclical",
	ANN_structure="S",
	number_of_refinements=number_of_refinements,
	number_of_all_runs=number_of_all_runs)
[learning_errors_all_2, task_errors_all_2] =\
loading_plotting_data_fcn(
	experiment_ID_base=experiment_ID_base,
	use_sensory=1,
	use_feedback=0,
	curriculum=curriculum,
	task_type="cyclical",
	ANN_structure="S",
	number_of_refinements=number_of_refinements,
	number_of_all_runs=number_of_all_runs)
# import pdb; pdb.set_trace()
fig1=compare_learning_error_plots_fcn(learning_errors_all_1, learning_errors_all_2, labels, curriculum="_E2H")
fig2=compare_task_error_plots_fcn(task_errors_all_1, task_errors_all_2, labels)
#import pdb; pdb.set_trace()
plt.show(block=1)
save_figures = 0
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
