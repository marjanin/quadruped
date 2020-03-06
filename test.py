import numpy as np
from matplotlib import pyplot as plt
from all_functions import sinusoidal_CPG

q0=sinusoidal_CPG_fcn(w = 2, phi = np.pi, lower_band = -10, upper_band = 12, attempt_length = 3 , timestep = 0.01)
plt.plot(q0)
plt.show()