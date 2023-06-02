# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 13:06:57 2023

@author: pgomez
"""

from pathlib import Path
import pandas as pd
import numpy as np

# Set the path to the directory containing the files
dir_path = Path(r"I:\Experiments\CODING") #change to point to folder
path_summary = r"I:\Experiments\CODING\Summary.csv" #change to point to summary file
filename_analysis = dir_path.stem + ("_ALL_MICE_otherbehaviors_analyzed.csv")
save_path_analysis= Path (dir_path.joinpath(filename_analysis))

framerate = 20

def take_all_files (dir_path):
    """
    Function to take all the files that contain a certain word from a folder and extract all behaviors
    Files should be named: MOUSE NUMBER_CONDITION_whateverinfo_TEST.csv
    Returns a dictionary with the mouse number and the filepath for each file

    """
    def extract_behaviors_file(file_path):
        """
        Extracts all behaviors from each column and transforms them into independent variables
        """
        df = pd.read_csv(file_path)
        variables_dict = {}
        for col_name in df.columns:
            if not col_name or col_name.startswith("Unnamed"):
                continue
            # Create a new variable with the same name as the column, storing the data as a numpy array
            variables_dict[col_name] = np.array(df[col_name])
        return variables_dict
    
    # Create an empty dictionary to store the results
    results_dict = {}
    
    # Iterate over all files in the directory
    for file_path in Path(dir_path).glob("*TEST*.csv"):
        # Extract the mouse number from the file name
        mouse_num = file_path.name.split("_")[0]
        # Call the extract_behaviors_file function to get a dictionary of variables for this file
        variables_dict = extract_behaviors_file(str(file_path))
        # Check if the mouse number is already a key in the results dictionary
        if mouse_num in results_dict:
            # If it is, append the variables dictionary to the list of dictionaries for that mouse
            results_dict[mouse_num].append(variables_dict)
        else:
            # If it's not, create a new list with the variables dictionary and store it in the results dictionary
            results_dict[mouse_num] = [variables_dict]
            
    return results_dict
    

dic_allmice = take_all_files (dir_path)

def count_behavior(behavior, framerate):
    """
    Function that counts total time, number of events and mean latency between those events of a binary
    BORIS behavior exported file
    """
    #calculate total time
    time_behavior = float(sum(behavior)/framerate)
    
    #calculate number of events 
    in_behavioral_event = False
    behavioral_events = []
    for index, value in enumerate(behavior):
        if value == 1:
            if not in_behavioral_event:
                in_behavioral_event = True
                start_index = index
        else:
            if in_behavioral_event:
                in_behavioral_event = False
                end_index = index - 1
                behavioral_events.append((start_index, end_index))
                
    number_events =  float(len(behavioral_events))
    
    #calculate latency between events
    durations = []
    for i in range(len(behavioral_events) - 1):
        # Calculate the duration between the end of the current tuple and the start of the next tuple
        current_end = behavioral_events[i][1]
        next_start = behavioral_events[i + 1][0]
        duration = next_start - current_end
        durations.append(duration)
        
    durations_seconds = np.array(durations)/framerate
    
    average_latency =  float(np.mean(durations_seconds))

    return  time_behavior, number_events, average_latency

def behav_allmice (input_dict, framerate):
    """
    Function that runs the count_behavior function in all elements of the dictionary that stores the 
    values for the behavior of all mice and gives a dictionary with the resulting calculation

    """
    output_dict = {}
    # Loop through the input dictionary
    for key, value in input_dict.items():
        # Create an empty list to store the output for this key
        output_list = {}
        # Loop through the arrays for this key
        for array_dict in value:
            # Create a new dictionary to store the processed arrays
            processed_dict = {}
            # Loop through the arrays in the array dictionary
            for array_key, array_value in array_dict.items():
                # Process the array using the sample function
                time_behavior, number_events, average_latency = count_behavior(array_value, framerate)
                # Store the processed array in the new dictionary
                processed_dict[array_key] = {"total_time": time_behavior,
                                          "number_events": number_events,
                                          "average_latency": average_latency}
            
            # Append the new dictionary to the output list
            output_list.update(processed_dict)
        # Store the output list for this key in the output dictionary
        output_dict[key] = output_list
    return output_dict


def identify_novelfamiliar (path_summary, dic_all, framerate):
    """
    Function identifies which objects are novel or familiar based on a previously defined excel
    """
    dataframe = pd.read_csv(path_summary)
    dictionary = dict_all
    if path_summary is not None:    
        #FOR TEST 
        for key in dictionary.keys():
            for i in range(len(dataframe)):
                if dataframe.loc[i]["Novel"] == "Same":
                    pass
                else:
                    if dataframe.loc[i]['Mouse #'] == int(key):
                        if dataframe.loc[i]['Novel'] == 'Up':
                            dictionary[key]['turning_novel'] = dictionary[key].pop('turning_left')
                            dictionary[key]['turning_familiar'] = dictionary[key].pop('turning_right')
                            dictionary[key]['novel_object'] = dictionary[key].pop('left_object')
                            dictionary[key]['familiar_object'] = dictionary[key].pop('right_object')
                        else:
                            dictionary[key]['turning_novel'] = dictionary[key].pop('turning_right')
                            dictionary[key]['turning_familiar'] = dictionary[key].pop('turning_left')
                            dictionary[key]['novel_object'] = dictionary[key].pop('right_object')
                            dictionary[key]['familiar_object'] = dictionary[key].pop('left_object')
                        novel_obj = dictionary[key]['novel_object']['total_time']
                        fam_obj = dictionary[key]['familiar_object']['total_time']
                        discrimination_index = (novel_obj - fam_obj) / (novel_obj + fam_obj)
                        dictionary[key]['novel_object']['discrimination_index'] = discrimination_index
    return dictionary

processed_dict = identify_novelfamiliar (path_summary, dic_allmice, framerate)

def generate_summary (d, path_summary):
    """
    Creates summary file with all the parameters of the NOR 
    """
    summary = pd.read_csv(path_summary)
    dfs = []
    for key, value in d.items():
        novel_object = d[key]['novel_object']
        familiar_object = d[key]['familiar_object']
        column_names_novel = [f"{col}_Novel Object" for col in novel_object.keys()]
        df_novel = pd.DataFrame([novel_object.values()], columns=column_names_novel)
        column_names_familiar = [f"{col}_Familiar Object" for col in familiar_object.keys()]
        df_familiar = pd.DataFrame([familiar_object.values()], columns=column_names_familiar)
        df_summary = pd.concat([df_novel, df_familiar], axis=1).T.sort_index(ascending=False).T
        dfs.append(df_summary)
    final_summary = pd.concat(dfs, axis=0, ignore_index=True)
    all_mice = pd.concat ([summary, final_summary], axis = 1)
    all_mice.to_csv(path_summary)
    return all_mice

def generate_excel (d, save_path):
    """
    Takes the dictionary, transforms it into a dataframe with the name of the behaviors

    """
    data = {}
    for key in d:
        subdict = d[key]
        for event in subdict:
            if event == 'time':
                continue
            new_key = event + '_' + key
            event_dict = subdict[event]
            data.setdefault(new_key, {})
            for subkey in event_dict:
                data[new_key][subkey] = event_dict[subkey]
    df = pd.DataFrame.from_dict(data, orient='index').sort_index()
    df.to_csv(save_path, index=True)
    return df

    
if path_summary is not None:
    nor_summary = generate_summary (processed_dict, path_summary) 
    print (nor_summary)
else:
    full_dic = generate_excel (processed_dict, save_path_analysis)
    print (full_dic)




