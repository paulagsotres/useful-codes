# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 17:48:10 2023

@author: pgomez
"""

import cv2

import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from scipy import signal as ss
from scipy.signal import find_peaks
from scipy.signal import butter
from scipy.signal import filtfilt
from scipy.integrate import simps
from matplotlib.animation import FuncAnimation



file_path = Path(r"I:\Experiments\2023\20230302_Test fiber photometry\20230306_mitoGcAMP_odors day 2 - odor stranger\2_mitoGcAMP_06.03.2023_test odors_000_001.mat")
filepathbehav = Path(r"I:\Experiments\2023\20230302_Test fiber photometry\20230306_mitoGcAMP_odors day 2 - odor stranger\2_mitoGcAMP_06.03.2023_test odors_000_001_BEHAVIOR_No focal subject.csv")
video_path = r"H:\2023\20230411_MitoGcamP6S Batch 2\MP1_HABITUATION_DAY 1-04112023162321.avi"
filename = file_path.stem
filename_analysis = file_path.stem + ("_analyzed.csv")
filename_extracted = file_path.stem + ("_extracted.csv")
save_path_extracted = Path (file_path.parent.joinpath(filename_extracted))
save_path_analysis = Path (file_path.parent.joinpath(filename_analysis)) #takes the parent, and joins the new created path for the filename
glob_path = file_path.glob("*.mat")
filter_window = 10 #average moving window to calculate Fmean in frames
framerate = 20 # Sampling frequency in Hz
photobleach = 1200#time to remove because of the photobleaching
baseline_start = 1300
baseline_end = 2300
#---------------------

def open_data (file_path): #fix so it takes the path from dictionary 
    with h5py.File (file_path) as f:
            signal = f['sig'][()]
    return signal #specific to open keys from a h5py

signal = open_data (file_path)
isosbestic = signal[:, 0] 
raw_signal = signal[:, 1]

#--------------generate timestamp--------------

def timestamp (signal, framerate, photobleach):
    """
    This function generates two columns that we might find useful later, one with the timestamp in frames
    and the other one with the timestamp equivalent in minutes,  without counting the photobleaching period

    Parameters
    ----------
    signal : np.array
        
    framerate : integer
        framerate in which the recording was acquired
        
    photobleaching : integer
        

    Returns
    -------
    timestamp_frames : np.array 
        np.array with the same number of rows as the original signal, starts at 0 frames
    timestamp_seconds : np.array
       same but in seconds

    """
    timestamp_list = []
    for i in range(len(signal)):
        timestamp_list.append (i)
    timestamp_list = timestamp_list [photobleach:]
    timestamp_frames = np.array(timestamp_list)
    timestamp_seconds =  timestamp_frames/framerate
    return timestamp_frames, timestamp_seconds

timestamp_frame, timestamp_seconds= timestamp (raw_signal, framerate, photobleach)

#-------------remove period of photobleaching at the beginning
def remove_bleaching (signal, photobleach):
    isosbestic_new = signal[:, 0][photobleach:]
    rawsignal_new = signal[:, 1][photobleach:]
    return isosbestic_new, rawsignal_new

isosbestic, raw_signal = remove_bleaching (signal, photobleach)
#updated signals without photobleaching


#------- smoothing of the signal

def smoothing (isosbestic, rawsignal, filter_window):
    b = np.divide(np.ones((filter_window,)), filter_window)
    a = 1
    control_smooth = ss.filtfilt(b, a, isosbestic)
    signal_smooth = ss.filtfilt(b, a, rawsignal)
    return control_smooth, signal_smooth

isosbestic_smooth, raw_signal_smooth = smoothing (isosbestic, raw_signal, filter_window)

#---------------------fitting

# Fit the control channel to the signal channel using a least squares polynomial fit of degree 1.
def controlfit(isosbesticsmooth, rawsignalsmooth):
    p = np.polyfit(isosbesticsmooth, rawsignalsmooth, 1)
    fitted_isosbestic = np.polyval(p, isosbesticsmooth)
    return fitted_isosbestic

isosbestic_fitted = controlfit (isosbestic_smooth, raw_signal_smooth)

#------------------calculate dff after fitting


def compute_delta_f_over_f(fitted_control_channel, signal_channel):
    """
    Computes ΔF/F by subtracting the fitted control channel from the signal channel and dividing by the fitted control channel.

    Parameters:
    control_channel (numpy array): The control channel signal.
    signal_channel (numpy array): The signal channel signal.

    Returns:
    delta_f_over_f (numpy array): The ΔF/F signal.
    """

    # Compute ΔF/F by subtracting the fitted control channel from the signal channel, and dividing by the fitted control channel.
    delta_f_over_f = (signal_channel - fitted_control_channel) / fitted_control_channel
    
    percdff = delta_f_over_f*100
    
    return delta_f_over_f, percdff

dff, dff100 = compute_delta_f_over_f(isosbestic_fitted, raw_signal_smooth)

#------------------detect transients 

def transients (signal, window_size):
    # Calculate the median absolute deviation (MAD) in the moving window
    kernel = np.ones(window_size) / window_size
    
    # Calculate the median absolute deviation (MAD) in the moving window
    mad = np.zeros(signal.shape)
    for i in range(signal.shape[0]):
        start = max(0, i - window_size // 2)
        end = min(signal.shape[0], i + window_size // 2)
        mad[i] = np.median(np.abs(signal[start:end] - np.median(signal[start:end])))
    
    # Convolve the signal with the kernel to obtain the moving average
    moving_avg = np.convolve(signal, kernel, mode='same')
    # Use the MAD as a threshold to detect calcium events in the signal
    threshold = 2 * mad
    peaks, properties = find_peaks(signal, prominence=threshold)
    
    # Return the indices and values of the detected calcium events
    event_values = signal[peaks]
    
    plt.plot (peaks, signal [peaks], "r+")
    plt.plot (signal)
    plt.show()
    
    return peaks, event_values
    
peaksindex, peakvalue = transients (dff100, 2400)

#----------------------------bandpass filter
# Define the filter parameters
lowcut = 0.015 # Lower cutoff frequency in Hz
highcut = 5# Upper cutoff frequency in Hz
order = 4  # Filter order
t = timestamp_seconds
# Define the filter function
def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs  # Nyquist frequency
    low = lowcut/nyq  # Normalized lower cutoff frequency
    high = highcut/nyq  # Normalized upper cutoff frequency
    b, a = butter(order, [low, high], btype='band')  # Compute filter coefficients
    return b, a

def apply_butterpass (signal, lowcut, highcut, framerate, order):
    b, a = butter_bandpass(lowcut, highcut, framerate, order)
    filtered = filtfilt(b, a, signal)
    return filtered
    
filtered = apply_butterpass (dff100, lowcut, highcut, framerate, order)


#------------calculate zscore based on a baseline period 

def baseline_z_score(signal, baseline_start, baseline_end, photobleach):
    """
    Calculates the baseline z-score of a signal by subtracting the mean and dividing by the standard deviation of
    a designated baseline period.

    Args:
        signal (numpy array): The signal to calculate the baseline z-score of.
        baseline_start (int): The starting index of the baseline period in the signal.
        baseline_end (int): The ending index of the baseline period in the signal.

    Returns:
        numpy array: The baseline z-scored signal.
    """
    baseline_signal = signal[baseline_start:baseline_end]
    mean_baseline = np.mean(baseline_signal)
    std_baseline = np.std(baseline_signal)
    baseline_z_score = (signal - mean_baseline) / std_baseline

    return baseline_z_score

zscore = baseline_z_score(filtered, baseline_start, baseline_end, photobleach)


#--------- to save processed data in csv 

def data_processed (signal, photobleach, framerate, filter_window, save_path, lowcut, highcut, order):
    timestamp_s= timestamp (signal, framerate, photobleach)[1]
    isosbestic_raw, raw_signal = remove_bleaching (signal, photobleach)
    isosbestic_smooth, raw_signal_smooth = smoothing (isosbestic, raw_signal, filter_window)
    isosbestic_fitted = controlfit (isosbestic_smooth, raw_signal_smooth)
    dff, dff100 = compute_delta_f_over_f(isosbestic_fitted, raw_signal_smooth)
    filtered = apply_butterpass (dff, lowcut, highcut, framerate, order)
    z_score = baseline_z_score(filtered, baseline_start, baseline_end, photobleach)
    all_data = np.vstack ((timestamp_s, isosbestic_raw, raw_signal, isosbestic_smooth, raw_signal_smooth, isosbestic_fitted, dff, dff100, filtered, z_score)).T
    all_data_csv = pd.DataFrame(all_data).to_csv (save_path, index="Timestamp", header =["Seconds (starting after photobleach period", "Isosbestic raw", "Signal raw", "Isosbestic smoothed", "Signal smoothed", "Isosbestic fitted", "Normalized AF/F", "% Normalized AF/F", "Filtered", "Zscore"])
    return all_data, all_data_csv

all_data, all_data_csv = data_processed (signal, photobleach, framerate, filter_window, save_path_analysis, lowcut, highcut, order) #saves csv in folder of origin with the same name and *analyzed




#------------PLOT SIGNAL REAL TIME 

plt.plot (filtered)
signal = filtered

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



xlabels = ['0', '10', '20', '30', '40', '50', '60']
ax.set_xticklabels(xlabels)
ax.set_ylabel("'ΔF/F")
ax.set_xlabel("Time(s)")

def init():
    ax.set_ylim(signal.min(), signal.max())
    ax.set_xlim(0, 1200)
    return line,

def animate(i):
    xdata.append(i)
    ydata.append(signal[i])
    line.set_data(np.array(xdata[-1200:])-i+1200, ydata[-1200:])
    return line,

ani = FuncAnimation(fig, animate, frames=len(signal), interval=50, init_func=init)
plt.show()
