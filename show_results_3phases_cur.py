import numpy as np
from matplotlib import pyplot as plt
experiment_ID_base = 'cur3_1_'
all_sensory_cases = [True, False]
for use_sensory in all_sensory_cases:
	np.random.seed(0)
	if use_sensory:
		experiment_ID = experiment_ID_base+"w_sensory"
	else:
		experiment_ID = experiment_ID_base+"wo_sensory"
	errors_all = np.load('./results/{}_babble_and_refine_results.npy'.format(experiment_ID))
	task_errors = np.load('./results/{}_task_results.npy'.format(experiment_ID))
	import pdb; pdb.set_trace()





# y=np.mean(all_performances,0)
# yerr=np.std(all_performances,0)
# #import pdb; pdb.set_trace()
# axes.errorbar(x=x, y=y, yerr=yerr, capsize=2, animated=True, alpha=.3, color='C2')
# axes.plot(x, y,'--', alpha=.7, color='C0')
# axes.set_title('Error vs. run')
# axes.set_xlabel('run #')
# axes.set_ylabel('Error')
# plt.plot()

# experiment_ID = 'exp2'
# all_performances = np.load('./results/{}_results.npy'.format(experiment_ID))
# y=np.mean(all_performances,0)
# yerr=np.std(all_performances,0)
# #import pdb; pdb.set_trace()
# axes.errorbar(x=x, y=y, yerr=yerr, capsize=2, animated=True, alpha=.3, color='C3')
# axes.plot(x, y,'--', alpha=.7, color='C1')
# axes.set_title('Error vs. run')
# axes.set_xlabel('run #')
# axes.set_ylabel('Error')
# plt.plot()

#plt.show(block=True)

