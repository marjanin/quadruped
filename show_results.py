import numpy as np
from matplotlib import pyplot as plt

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))
x=np.arange(0,11)

experiment_ID = 'exp1'
all_performances = np.load('./results/{}_results.npy'.format(experiment_ID))
y=np.mean(all_performances,0)
yerr=np.std(all_performances,0)
#import pdb; pdb.set_trace()
axes.errorbar(x=x, y=y, yerr=yerr, capsize=2, animated=True, alpha=.3, color='C2')
axes.plot(x, y,'--', alpha=.7, color='C0')
axes.set_title('Error vs. run')
axes.set_xlabel('run #')
axes.set_ylabel('Error')
plt.plot()

experiment_ID = 'exp2'
all_performances = np.load('./results/{}_results.npy'.format(experiment_ID))
y=np.mean(all_performances,0)
yerr=np.std(all_performances,0)
#import pdb; pdb.set_trace()
axes.errorbar(x=x, y=y, yerr=yerr, capsize=2, animated=True, alpha=.3, color='C3')
axes.plot(x, y,'--', alpha=.7, color='C1')
axes.set_title('Error vs. run')
axes.set_xlabel('run #')
axes.set_ylabel('Error')
plt.plot()

plt.show(block=True)

