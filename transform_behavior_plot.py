# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 11:58:26 2023

@author: pgomez
"""

from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

file_path = r"I:\Gomez-Sotres et al 2023 - Data\Fig 2\CB1 KO.xlsx"
save_path = Path(r"I:\Gomez-Sotres et al 2023 - Data\Fig 2")



def open_behavior (file_path):
    """
    Opens the behavior from a boris csv file 
    """
    excel_file = pd.ExcelFile(file_path)

    # Get the sheet names
    sheet_names = excel_file.sheet_names
    
    # Create an empty dictionary to store the DataFrames
    dataframes = {}
    
    # Iterate over the sheet names
    for sheet_name in sheet_names:
        # Read each sheet as a DataFrame
        dataframe = excel_file.parse(sheet_name)
        
        # Store the DataFrame in the dictionary
        dataframes[sheet_name] = dataframe
    
    # Access the DataFrames by sheet name
    df1 = dataframes['CB1 KO STRESS']
    df2 = dataframes['CB1 KO NO STRESS']
    return df1, df2 

micuko, micuwt = open_behavior (file_path)

def plot_all_behaviors(dataframe, title):
    behaviors = ["sniff ano-gen", "sniff head-torso", "allogroom", "groom", "rear", "dig", "walk", "sit", "fight"]

    def detect_behaviors(dataframe, behaviors):
        result_dict = {}
        
        for column in dataframe.columns:
            column_dict = {}
            
            for behavior in behaviors:
                behavior_series = (dataframe[column] == behavior)
                column_dict[behavior] = behavior_series
              
            result_dict[column] = column_dict
    
        return result_dict
    
    # Assuming you have a dataframe named 'dataframe' and it has 300 rows
    
    behavior_dict = detect_behaviors(dataframe, behaviors)
    
    color_ano = "cyan"
    color_head = "limegreen"
    color_allogroom = "navy"
    color_groom = "orange"
    color_rear = "thistle"
    color_dig = "saddlebrown"
    color_walk = "darkgreen"
    color_sit = "khaki"
    color_fight = "lightcoral"
    
    fig, ax = plt.subplots(figsize=(10, 3))
    
    y_ticks = []
    y_tick_labels = []
    for i, (key, value) in enumerate(behavior_dict.items()):
        y_ticks.append(i + 0.5)
        y_tick_labels.append(key)
        
        for behavior, values in value.items():
            events = np.nonzero(np.array(values))[0]
            
            for event in events:
                
                color = None
                if behavior == "sniff ano-gen":
                    color = color_ano
                elif behavior == "sniff head-torso":
                    color = color_head
                elif behavior == "allogroom":
                    color = color_allogroom
                elif behavior == "groom":
                    color = color_groom
                elif behavior == "rear":
                    color = color_rear
                elif behavior == "dig":
                    color = color_dig
                elif behavior == "walk":
                    color = color_walk
                elif behavior == "sit":
                    color = color_sit
                elif behavior == "fight":
                    color = color_fight
                    
                rect = plt.Rectangle((event, i), 1, 1, color=color)
                ax.add_patch(rect)
    
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Mouse")
    ax.set_ylim(0, len(behavior_dict))
    ax.set_xticks(np.arange(0, 301, 30))
    fig.suptitle(title, fontsize=14)
    plt.gca().invert_yaxis()
    plt.show()
    return fig

def plot_anogenital_behaviors(dataframe, title):
    behaviors = ["sniff ano-gen", "sniff head-torso", "allogroom", "groom", "rear", "dig", "walk", "sit", "fight"]

    def detect_behaviors(dataframe, behaviors):
        result_dict = {}
        
        for column in dataframe.columns:
            column_dict = {}
            
            for behavior in behaviors:
                behavior_series = (dataframe[column] == behavior)
                column_dict[behavior] = behavior_series
              
            result_dict[column] = column_dict
    
        return result_dict
    
    # Assuming you have a dataframe named 'dataframe' and it has 300 rows
    
    behavior_dict = detect_behaviors(dataframe, behaviors)
    
    color_ano = "cyan"
    color_head = "white"
    color_allogroom = "white"
    color_groom = "white"
    color_rear = "white"
    color_dig = "white"
    color_walk = "white"
    color_sit = "white"
    color_fight = "white"
    
    fig, ax = plt.subplots(figsize=(10, 3))
    
    y_ticks = []
    y_tick_labels = []
    for i, (key, value) in enumerate(behavior_dict.items()):
        y_ticks.append(i + 0.5)
        y_tick_labels.append(key)
        
        for behavior, values in value.items():
            events = np.nonzero(np.array(values))[0]
            
            for event in events:
                
                color = None
                if behavior == "sniff ano-gen":
                    color = color_ano
                elif behavior == "sniff head-torso":
                    color = color_head
                elif behavior == "allogroom":
                    color = color_allogroom
                elif behavior == "groom":
                    color = color_groom
                elif behavior == "rear":
                    color = color_rear
                elif behavior == "dig":
                    color = color_dig
                elif behavior == "walk":
                    color = color_walk
                elif behavior == "sit":
                    color = color_sit
                elif behavior == "fight":
                    color = color_fight
                    
                rect = plt.Rectangle((event, i), 1, 1, color=color)
                ax.add_patch(rect)
    
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Mouse")
    ax.set_ylim(0, len(behavior_dict))
    ax.set_xticks(np.arange(0, 301, 30))
    fig.suptitle(title, fontsize=14)
    plt.gca().invert_yaxis()
    plt.show()
    return fig

def detect_behaviors(dataframe, behaviors):
        result_dict = {}
        
        for column in dataframe.columns:
            column_dict = {}
            
            for behavior in behaviors:
                behavior_series = (dataframe[column] == behavior)
                column_dict[behavior] = behavior_series
              
            result_dict[column] = column_dict
    
        return result_dict

def plot_headtorso_behaviors(dataframe, title):
    behaviors = ["sniff ano-gen", "sniff head-torso", "allogroom", "groom", "rear", "dig", "walk", "sit", "fight"]
    
    # Assuming you have a dataframe named 'dataframe' and it has 300 rows
    
    behavior_dict = detect_behaviors(dataframe, behaviors)
    
    color_ano = "white"
    color_head = "limegreen"
    color_allogroom = "white"
    color_groom = "white"
    color_rear = "white"
    color_dig = "white"
    color_walk = "white"
    color_sit = "white"
    color_fight = "white"
    
    fig, ax = plt.subplots(figsize=(10, 3))
    
    y_ticks = []
    y_tick_labels = []
    for i, (key, value) in enumerate(behavior_dict.items()):
        y_ticks.append(i + 0.5)
        y_tick_labels.append(key)
        
        for behavior, values in value.items():
            events = np.nonzero(np.array(values))[0]
            
            for event in events:
                
                color = None
                if behavior == "sniff ano-gen":
                    color = color_ano
                elif behavior == "sniff head-torso":
                    color = color_head
                elif behavior == "allogroom":
                    color = color_allogroom
                elif behavior == "groom":
                    color = color_groom
                elif behavior == "rear":
                    color = color_rear
                elif behavior == "dig":
                    color = color_dig
                elif behavior == "walk":
                    color = color_walk
                elif behavior == "sit":
                    color = color_sit
                elif behavior == "fight":
                    color = color_fight
                    
                rect = plt.Rectangle((event, i), 1, 1, color=color)
                ax.add_patch(rect)
    
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Mouse")
    ax.set_ylim(0, len(behavior_dict))
    ax.set_xticks(np.arange(0, 301, 30))
    fig.suptitle(title, fontsize=14)
    plt.gca().invert_yaxis()
    plt.show()
    return fig


def calculate_freq (dataframe):
    behaviors = ["sniff ano-gen", "sniff head-torso", "allogroom", "groom", "rear", "dig", "walk", "sit", "fight"]
    data = detect_behaviors(dataframe, behaviors)
    result = {}
    
    for mouse, values in data.items():
        # Iterate over each behavioral event
        for event, values_dict in values.items():
            # Create a list to store frequency values per row
            frequency_list = []
            
            # Iterate over each row in the values_dict
            for row in range(len(values_dict)):
                # Count the number of True values for the event across all mice
                count = sum([data[mouse][event][row] for mouse in data])
                
                # Calculate the frequency by dividing the count by the number of mice
                frequency = count / len(data)
                
                # Append the frequency to the list
                frequency_list.append(frequency)
            
            # Create a DataFrame with the frequency list and store it in the result dictionary
            result[event] = pd.DataFrame(frequency_list, columns=[event])
    
    return result


def plot_frequency (dictionary):
    freq_dict = calculate_freq (dictionary)
    num_plots = len(freq_dict.keys())
    fig, axs = plt.subplots(num_plots, 1, figsize=(10, num_plots*2), sharex=True, sharey=True)
    fig.tight_layout(pad=2.0)
    plt.subplots_adjust(hspace=0.4)

    for i, (behavior, data) in enumerate(freq_dict.items()):
        normalized_data = data / 1

        axs[i].imshow(normalized_data.T, cmap='inferno', vmin=0, vmax=1, aspect='auto')
        axs[i].set_xlabel('Behaviors')
        axs[i].set_ylabel('Time (s)')
        axs[i].set_title(f'{behavior} Timeline')

    plt.show()

plot_frequency (micuko)
plt.savefig(save_path.joinpath(save_path.stem +("plot_frequency_CB1 KO - shock.svg")), format='svg')

plot_frequency (micuwt)
plt.savefig(save_path.joinpath(save_path.stem +("plot_frequency_CB1 KO - no_shock.svg")), format='svg')



plot_all_behaviors(micuko, title = "CB1 KO - Stress DEM")
plt.savefig(save_path.joinpath(save_path.stem +("all_shock.svg")), format='svg')       
plot_all_behaviors(micuwt, title ="CB1 KO - Neutral DEM") 
plt.savefig(save_path.joinpath(save_path.stem +("all_neutral.svg")), format='svg')   

plot_anogenital_behaviors(micuko, title = "Anogenital investigation of CB1 KO - Stress DEM")       
plot_anogenital_behaviors(micuwt, title ="Anogenital investigation of CB1 KO - Neutral DEM") 

plot_headtorso_behaviors(micuko, title = "Body exploration of CB1 KO - Stress DEM")       
plot_headtorso_behaviors(micuwt, title ="Body exploration of CB1 KO - Neutral DEM") 

