# -*- coding: utf-8 -*-
"""
Created on Fri May  5 14:53:42 2023

@author: pgomez
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = Path(r"C:\Users\pgomez\Desktop\NOR heatmap\MP6_NEDLC_dlcrnetms5_NOR track test Mar2shuffle1_100000_el.csv") #change to point to folder
# Set the path to the directory containing the files
dir_path = Path(r"C:\Users\pgomez\Desktop\NOR heatmap") #change to point to folder


    

def get_data(path):
    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(path)

    # Find the columns that contain the word "nose"
    nose_cols = [col for col in df.columns if 'Nose' in df[col].values]

    # Find the first row that contains the word "nose"
    nose_row = df.isin(['Nose']).any(axis=1).idxmax()

    # Get the rows below the nose_row and the columns that contain "nose"
    nose_data = df.loc[nose_row+2:, nose_cols].values

    
    nose_array = np.array (nose_data)

    # Print the resulting DataFrame
    return nose_array

def filter_coordinates(coordinates):
    x_diff = np.diff(coordinates[:, 0])
    y_diff = np.diff(coordinates[:, 1])
    distance = np.sqrt(x_diff**2 + y_diff**2)
    distance_threshold = 40
    idx_to_remove = np.where(distance > distance_threshold)[0] + 1
    filtered_coordinates = np.delete(coordinates, idx_to_remove, axis=0)
    return filtered_coordinates


def take_all_files (path):
    file_dict = {}
    files = Path(dir_path).glob("*shuffle*.csv")
    for i, file_path in enumerate(files):
        key = f"file_{i}"
        file_dict[key] = filter_coordinates (get_data(file_path).astype(float))
    return file_dict    

dict_results = take_all_files (dir_path)

def average_dict(dict_array):
    # Combine all arrays into one array
    all_arrays = np.concatenate([arr for arr in dict_array.values()])

    # Calculate the mean of each column
    means = np.nanmean(all_arrays, axis=0)

    # Return means as a 2D array
    return means.reshape(1, -1)

coordinates = average_dict_arrays(dict_results)

def plot_heatmap(coord, bins):
    # Convert x and y to float arrays
    x = coord[:, 0]
    y=coord[:, 1]
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    # Define the bin edges
    x_edges = np.linspace(x.min(), x.max(), bins + 1)
    y_edges = np.linspace(y.min(), y.max(), bins + 1)

    # Compute the 2D histogram
    H, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    
    # Find the maximum frequency
    max_freq = H.max()

    # Find the indices of the cells with maximum frequency
    max_freq_indices = np.where(H == max_freq)

    # If there is only one cell with maximum frequency, remove it from the heatmap
    if len(max_freq_indices[0]) == 1:
        H[max_freq_indices] = 0 
        
    H_max = H.max()

    # Normalize the histogram
    H_norm = H / H_max

    # Plot the heatmap
    plt.imshow(H_norm.T, origin='lower', cmap='jet', extent=[x.min(), x.max(), y.min(), y.max()])
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

plot_heatmap(coor, 100)
