# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 12:17:09 2023

@author: pgomez
"""
import numpy as np
from matplotlib.animation import FuncAnimation

framerate = 20
signal = "/path/"

fig, ax = plt.subplots()
xdata, ydata = [], []
line, = ax.plot([], [], color="white")
ax.axhline(y=0, color='white', linewidth=0.5)

# Set the background color to black
fig.patch.set_facecolor('black')
ax.set_facecolor('black')

# Set the axis tick labels and axis label color to white
ax.tick_params(colors='white')
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')


#Set label names
xlabels = ['0', '10', '20', '30', '40', '50', '60']
ax.set_xticklabels(xlabels)
ax.set_ylabel("'ΔF/F")
ax.set_xlabel("Time(s)")

#Function to determine graph size
def init():
    ax.set_ylim(signal.min(), signal.max())
    ax.set_xlim(0, 60*framerate) #showing axis of 1 min at a time
    return line,

#Function to update the plot to animate
def animate(i):
    xdata.append(i)
    ydata.append(signal[i])
    line.set_data(np.array(xdata[-60*framerate:])-i+60*framerate, ydata[-60*framerate:]) #showing window of 1 min
    return line,

#Animation
ani = FuncAnimation(fig, animate, frames=len(signal), interval=(1/framerate)*1000, init_func=init)
plt.show()