import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import time
#data
t_max=1 #seconds
fs=100
interval=1000*(1/fs)# in ms
total_frames=t_max*fs
t=np.linspace(0,t_max,fs)
l0=1
l1=1
w=1
theta1_offset=0
theta0=2*np.pi*w*t
theta1=theta1_offset+2*np.pi*w*t
x0=np.cos(theta0)*l0
y0=-np.sin(theta0)*l0
x1=np.cos(theta0+theta1)*l1+x0
y1=-np.sin(theta0+theta1)*l1+y0
#set up figure
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-2, 2), ylim=(-2, 2))
ax.grid()

line1, = ax.plot([], [], 'o-', lw=2)
line2, = ax.plot([], [], 'o-', lw=2)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
energy_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)

# animation

def init():
    """initialize animation"""
    line1.set_data([], [])
    line2.set_data([], [])
    time_text.set_text('')
    energy_text.set_text('')
    return line1,line2, time_text, energy_text

def animate(i):
    """perform animation step"""    
    line1.set_data([0,x0[i],x1[i]],[0,y0[i],y1[i]])
    line2.set_data([0,-x0[i],-x1[i]],[0,y0[i],y1[i]])

#    line.set_data([x0[i],x1[i]],[y0[i],y1[i]])
    # time_text.set_text('time = %.1f' % pendulum.time_elapsed)
    # energy_text.set_text('energy = %.3f J' % pendulum.energy())
    return line1,line2#, time_text, energy_text

# animate(0)
ani = animation.FuncAnimation(fig, animate, frames=total_frames,
                              interval=10, blit=False, init_func=init)

plt.show()