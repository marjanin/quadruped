import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import time

#data
test_run_RMSE=np.load('./tmp/test_run_RMSE.npy')
attempt_kinematics=np.load('./tmp/attempt_kinematics.npy')
returned_kinematics=np.load('./tmp/returned_kinematics.npy')

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
ani.save('./tmp/testvid.mp4', fps=fs*0.5) # fps=fs*0.5: 0.5 x speed
# plt.show()