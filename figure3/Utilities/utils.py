

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import pandas as pd
import ast
import os
import json
import seaborn as sns
import math
import statsmodels.api as sm
import pickle
import random



def determine_neuron_spike_pos(replay_motif_data,seq_type):
    neuron_spike_positions = {}
    for i in range(len(replay_motif_data)):

        if replay_motif_data.cluster_seq_type[i] == seq_type:

            replay_neurons = ast.literal_eval(replay_motif_data.cluster_neurons[i])
            replay_spikes = np.array(ast.literal_eval(replay_motif_data.cluster_spike_times[i]))

            norm_replay_spikes = normalize_times(replay_spikes)

            replay_neurons_fspike_only = replace_repeats_with_string(replay_neurons)

            for index,neur_id in enumerate(replay_neurons_fspike_only):
                if not neur_id == 'repeat':
                    if neur_id in neuron_spike_positions:
                        neuron_spike_positions[neur_id] += [norm_replay_spikes[index]]

                    else:
                        neuron_spike_positions[neur_id] = [norm_replay_spikes[index]]
    return neuron_spike_positions

def normalize_times(times):
    min_time = min(times)
    max_time = max(times)
    time_range = max_time - min_time

    normalized_times = [(t - min_time) / time_range for t in times]
    return normalized_times

def replace_repeats_with_string(numbers):
    seen = set()
    result = []

    for num in numbers:
        if num in seen:
            result.append('repeat')
        else:
            result.append(num)
            seen.add(num)

    return result


def find_closest_example(numbers, examples):
    # Initialize dictionaries to store the closest example and example totals
    closest_examples = {}
    example_totals = {example: 0 for example in examples}

    # Iterate over each number in the list
    for number in numbers:
        # Initialize a variable to keep track of the closest example
        closest_example = None
        min_distance = float('inf')  # Initialize the minimum distance to infinity

        # Compare the number with each example
        for example in examples:
            # Calculate the absolute difference between the number and example
            distance = abs(number - example)

            # Check if the current example is closer than the previous closest example
            if distance < min_distance:
                min_distance = distance
                closest_example = example

        # Update the closest example for the current number in the dictionary
        closest_examples[number] = closest_example

        # Increment the total count for the closest example
        example_totals[closest_example] += 1

    return closest_examples, example_totals

def relative_warp_values(e_f_warp_factors):
    rels = []
    for item in e_f_warp_factors:
        rels += [list(np.array(item)/sum(item))]
    return rels



def bin_data(x, y, n_bins):
   # Create bins for y values
    bin_edges = np.linspace(0, 1, n_bins)
    
    # Calculate bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Assign each data point to the closest bin
    bin_indices = np.argmin(np.abs(np.subtract.outer(y, bin_centers)), axis=1)
    
    # Initialize arrays to store binned data
    binned_x = [[] for _ in range(n_bins)]
    
    # Populate binned arrays
    for i in range(len(y)):
        bin_idx = bin_indices[i]
        binned_x[bin_idx].append(x[i])
    
    # Calculate mean and SEM for each bin
    bin_means = np.array([np.mean(b) for b in binned_x])
    bin_sems = np.array([np.std(b) for b in binned_x])
    
    return bin_means, bin_sems,bin_edges,binned_x

    
def plot_warps(e_f_warp_factors,e_r_warp_factors,ax,var_str,bins_):

    bin_labels = [item + 'x' for item in np.array(bins_).astype(str)]

    means = []
    sems = []
    data_out_f = []
    for item in conactinate_nth_items(e_f_warp_factors):
        means += [np.mean(item)]
        sems += [np.std(item)]
        data_out_f += [item]
    ax.plot(means,'-->', color = 'red', markersize = 8, label = 'forward')
    upper = np.array(means)+ sems
    lower = np.array(means)- sems
    ax.fill_between((range(len(bin_labels))),(lower),(upper),
        alpha=0.2, edgecolor='None', facecolor='red',
        linewidth=1, linestyle='dashdot', antialiased=True)

    means = []
    sems = []
    data_out_r = []
    for item in conactinate_nth_items(e_r_warp_factors):
        means += [np.mean(item)]
        sems += [np.std(item)]
        data_out_r += [item]
    ax.plot(means,'--<', color = 'blue', markersize = 8,label = 'reverse')
    upper = np.array(means)+ sems
    lower = np.array(means)- sems
    ax.fill_between((range(len(bin_labels))),(lower),(upper),
        alpha=0.2, edgecolor='None', facecolor='blue',
        linewidth=1, linestyle='dashdot', antialiased=True)
    ax.set_title(var_str)
    
    # Set the vertical labels
    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels, rotation=90)
    
    ax.set_ylim(0,0.40)

    ax.legend()
    
    return(data_out_f,data_out_r)

def conactinate_nth_items(startlist):
    concatinated_column_vectors = []
    for c in range(len(max(startlist, key=len))):
        column = []
        for t in range(len(startlist)):
            if c <= len(startlist[t])-1:
                column = column + [startlist[t][c]]
        concatinated_column_vectors.append(column)
    return concatinated_column_vectors

def poisson_probability(rate, k):
    """Calculate Poisson probability of k events occurring given the rate."""
    return math.exp(-rate) * (rate ** k) / math.factorial(k)

def calculate_probabilities(rate_per_second, event_length_s):
    # Convert event length from milliseconds to seconds
    
    # Calculate the rate per 0.3 seconds considering event length
    rate_per_0p3_seconds = rate_per_second *(0.3 + event_length_s)
    
    results = {}
    total_probability = 0  # Variable to store the total probability
    
    for k in range(1, 7):  # Include probabilities for 1 to 6 events
        probability = poisson_probability(rate_per_0p3_seconds, k)
        results[k] = probability
        total_probability += probability  # Add probability to the total
    
    # Calculate percentages
    percentages = {k: (probability / total_probability) * 100 for k, probability in results.items()}
    
    return rate_per_0p3_seconds, results, percentages

def bin_position_data(x, y, n_bins):
   # Create bins for y values
    bin_edges = np.linspace(0, 1, n_bins)
    
    # Calculate bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Assign each data point to the closest bin
    bin_indices = np.argmin(np.abs(np.subtract.outer(y, bin_centers)), axis=1)
    
    # Initialize arrays to store binned data
    binned_x = [[] for _ in range(n_bins)]
    
    # Populate binned arrays
    for i in range(len(y)):
        bin_idx = bin_indices[i]
        binned_x[bin_idx].append(x[i])
    
    # Calculate mean and SEM for each bin
    bin_means = np.array([np.mean(b) for b in binned_x])
    bin_sems = np.array([np.std(b) for b in binned_x])
    
    return bin_means, bin_sems,bin_edges,binned_x

def return_binned_neuron_awake_sleep_rel(awake_dat,sleep_dat):
    awake_sleep_relationship = {}
    for key in awake_dat:
        for index, awake_mouse_data in enumerate(awake_dat[key]):
            
            awake = awake_mouse_data
            sleep = np.array(sleep_dat[key][index])

            min_neurons_filter = 0
            bins = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]    

            for bin_ in bins:
                mask = np.round(awake,1) == bin_
                if len(sleep[mask]) > min_neurons_filter:
                    if bin_ in awake_sleep_relationship:
                        awake_sleep_relationship[bin_] += [np.mean(sleep[mask])]
                    else:
                        awake_sleep_relationship[bin_] = [np.mean(sleep[mask])]

    return awake_sleep_relationship

def plot_start_end_times(e_all_chunk_reverse_start_mean,e_all_chunk_forward_start_mean,e_all_chunk_reverse_end_mean,e_all_chunk_forward_end_mean,ax,ax2,var_str):
        
    ## plot forward start and ends

    ax.plot(np.array(e_all_chunk_reverse_start_mean),np.ones(len(e_all_chunk_reverse_start_mean))*0.3,'o', color = 'red', alpha = 0.5,markeredgewidth = 0, markersize = 9)

    ax.plot(np.array(e_all_chunk_reverse_end_mean),np.ones(len(e_all_chunk_reverse_end_mean))*0.7,'o', color = 'red', alpha = 0.5,markeredgewidth = 0, markersize = 9)


    groups =  ['starts'] * len(e_all_chunk_reverse_start_mean) + (['ends'] * len(e_all_chunk_reverse_end_mean)) 
    data =  e_all_chunk_reverse_start_mean +e_all_chunk_reverse_end_mean

    if len(data) > 0:
        forward_plt_df = pd.DataFrame({'group':groups,'distances (%)': data })
        ax=sns.boxplot( x = 'distances (%)', y = 'group', data = forward_plt_df, color = 'blue', width = .2, zorder = 10,\
                    showcaps = True, boxprops = {'facecolor':'none', "zorder":10},\
                    showfliers=False, whiskerprops = {'linewidth':2, "zorder":10},\
                       saturation = 1, orient = 'h',ax = ax)
        ax.set_xlabel('realtive start point')
        ax.set_title(var_str + '    reverse')

    ax.set_xlim(0,100)

    ###########

    ax2.plot(np.array(e_all_chunk_forward_start_mean),np.ones(len(e_all_chunk_forward_start_mean))*0.3,'o', color = 'red', alpha = 0.5,markeredgewidth = 0, markersize = 9)

    ax2.plot(np.array(e_all_chunk_forward_end_mean),np.ones(len(e_all_chunk_forward_end_mean))*0.7,'o', color = 'red', alpha = 0.5,markeredgewidth = 0, markersize = 9)


    groups =  ['starts'] * len(e_all_chunk_forward_start_mean) + (['ends'] * len(e_all_chunk_forward_end_mean)) 
    data =  e_all_chunk_forward_start_mean +e_all_chunk_forward_end_mean

    if len(data) > 0:
        forward_plt_df = pd.DataFrame({'group':groups,'distances (%)': data })
        ax=sns.boxplot( x = 'distances (%)', y = 'group', data = forward_plt_df, color = 'blue', width = .2, zorder = 10,\
                    showcaps = True, boxprops = {'facecolor':'none', "zorder":10},\
                    showfliers=False, whiskerprops = {'linewidth':2, "zorder":10},\
                       saturation = 1, orient = 'h',ax = ax2)


        ax2.set_xlabel('realtive start point')
        ax2.set_title(var_str + '    forward')


    ax2.set_xlim(0,100)
    
import scipy 
def bin_data(x, y, n_bins):
   # Create bins for y values
    bin_edges = np.linspace(min(x), max(x), n_bins)
    
#     # Calculate bin centers
#     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Assign each data point to the closest bin
    bin_indices = np.argmin(np.abs(np.subtract.outer(x, bin_edges)), axis=1)
    
    # Initialize arrays to store binned data
    binned_y = [[] for _ in range(n_bins)]
    
    # Populate binned arrays
    for i in range(len(x)):
        bin_idx = bin_indices[i]
        binned_y[bin_idx].append(y[i])
    
    # Calculate mean and SEM for each bin
    bin_means = np.array([np.mean(b) for b in binned_y])
    bin_sems = np.array([scipy.stats.sem(b) for b in binned_y])
    
    return bin_means, bin_sems,bin_edges,binned_y

# ## set ppseq file
def find_example_file(PP_PATH, example = '178_1_7'):
    for file_ in os.listdir(PP_PATH):
        if example in file_:
            file = file_
    return file 

def Load_example_data(pp_path, file, tracking_path, dat_path, mouse_session_recording = '178_1_7'):
    # Load processed spike data
    latent_event_history_df_split, spikes_seq_type_adjusted, neuron_order, ordered_preferred_type, neuron_index, config = load_processed_spike_data(pp_path, file)
    
    # Load DLC tracking data
    print("\nLOADING DLC TRACKING DATA")
    back_head_centre, back_ports = load_tracking_data(tracking_path)
    
    # Load the timespan used for pppseq
    behav_time_interval_start = load_input_params(pp_path, file, mouse_session_recording)
    
    # Load behaviour data
    behav_sync, transitions, poke_in_times, ports,behav_mask = load_behav_data(dat_path, behav_time_interval_start)
    
    
    neuron_response_df = pd.read_csv(pp_path + file + r"\neuron_response.csv")
    bkgd_log_proportions_array = pd.read_csv(pp_path + file + r"\bkgd_log_proportions_array.csv")

    # Return all loaded data for further processing if needed
    return {
        "latent_event_history_df_split": latent_event_history_df_split,
        "spikes_seq_type_adjusted": spikes_seq_type_adjusted,
        "neuron_order": neuron_order,
        "ordered_preferred_type": ordered_preferred_type,
        "neuron_index": neuron_index,
        "config": config,
        "back_head_centre": back_head_centre,
        "back_ports": back_ports,
        "behav_sync": behav_sync,
        "transitions": transitions,
        "poke_in_times": poke_in_times,
        "ports": ports,
        "neuron_response_df" : neuron_response_df,
        "bkgd_log_proportions_array" : bkgd_log_proportions_array,
        "behav_time_interval_start" : behav_time_interval_start,
        "behav_mask" : behav_mask
    }

def load_processed_spike_data(pp_path, file):
    print("\nLOADING processed_spike_data")
    analysis_path = os.path.join(pp_path, file, "analysis_output")
    
    latent_event_history_df_split = load_pickle(os.path.join(analysis_path, "latent_event_history_df_split.pickle"))
    spikes_seq_type_adjusted = load_pickle(os.path.join(analysis_path, "spikes_seq_type_adjusted.pickle"))
    neuron_order = np.load(os.path.join(analysis_path, 'neuron_order.npy'))
    ordered_preferred_type = np.load(os.path.join(analysis_path, 'ordered_preferred_type.npy'))
    neuron_index = np.load(os.path.join(analysis_path, 'neuron_index.npy'))

    config = eval(load_json(os.path.join(pp_path, file, 'config_file.json')))

    return latent_event_history_df_split, spikes_seq_type_adjusted, neuron_order, ordered_preferred_type, neuron_index, config

def load_pickle(file_path):
    with open(file_path, "rb") as input_file:
        return pickle.load(input_file)


def load_json(file_path):
    with open(file_path) as f:
        return json.load(f)
    
def load_tracking_data(tracking_path):
    back_head_centre = load_H5_bodypart(tracking_path, 'back', 'head_centre')
    back_p1, back_p2, back_p3, back_p4, back_p5 = load_H5_ports(tracking_path, 'back')
    return back_head_centre, (back_p1, back_p2, back_p3, back_p4, back_p5)

def load_H5_bodypart(tracking_path,video_type, tracking_point):

    # Load in all '.h5' files for a given folder:
    TFiles_unsort = list_files(tracking_path, 'h5')

    for file in TFiles_unsort:
        print(file)
        if video_type in file:
            if 'task' in file:
                back_file = pd.read_hdf(tracking_path + file)     
                
    # drag data out of the df
    scorer = back_file.columns.tolist()[0][0]
    body_part = back_file[scorer][tracking_point]
    
    parts=[]
    for item in list(back_file[scorer]):
        parts+=[item[0]]
    print(np.unique(parts))
    
    # clean and interpolate frames with less than 98% confidence
    clean_and_interpolate(body_part,0.98)
    
    return(body_part)
  
def load_H5_ports(tracking_path,video_type):

    # Load in all '.h5' files for a given folder:
    TFiles_unsort = list_files(tracking_path, 'h5')

    for file in TFiles_unsort:
        print(file)
        if video_type in file:
            if 'port' in file:
                back_ports_file = pd.read_hdf(tracking_path + file)

    ## same for the ports:
    scorer = back_ports_file.columns.tolist()[0][0]
        
    if video_type == 'back':
        port1 =back_ports_file[scorer]['port2']
        port2 =back_ports_file[scorer]['port1']
        port3 =back_ports_file[scorer]['port6']
        port4 =back_ports_file[scorer]['port3']
        port5 =back_ports_file[scorer]['port7']
    else:
        port1 =back_ports_file[scorer]['Port2']
        port2 =back_ports_file[scorer]['Port1']
        port3 =back_ports_file[scorer]['Port6']
        port4 =back_ports_file[scorer]['Port3']
        port5 =back_ports_file[scorer]['Port7']

    clean_and_interpolate(port1,0.98)
    clean_and_interpolate(port2,0.98)
    clean_and_interpolate(port3,0.98)
    clean_and_interpolate(port4,0.98)
    clean_and_interpolate(port5,0.98)
    
    return(port1,port2,port3,port4,port5)


def list_files(directory, extension):
    return (f for f in os.listdir(directory) if f.endswith('.' + extension))

def clean_and_interpolate(head_centre,threshold):
    bad_confidence_inds = np.where(head_centre.likelihood.values<threshold)[0]
    newx = head_centre.x.values
    newx[bad_confidence_inds] = 0
    newy = head_centre.y.values
    newy[bad_confidence_inds] = 0

    start_value_cleanup(newx)
    interped_x = interp_0_coords(newx)

    start_value_cleanup(newy)
    interped_y = interp_0_coords(newy)
    
    head_centre['interped_x'] = interped_x
    head_centre['interped_y'] = interped_y
    
def start_value_cleanup(coords):
    # This is for when the starting value of the coords == 0; interpolation will not work on these coords until the first 0 
    #is changed. The 0 value is changed to the first non-zero value in the coords lists
    for index, value in enumerate(coords):
        working = 0
        if value > 0:
            start_value = value
            start_index = index
            working = 1
            break
    if working == 1:
        for x in range(start_index):
            coords[x] = start_value
            
def interp_0_coords(coords_list):
    #coords_list is one if the outputs of the get_x_y_data = a list of co-ordinate points
    for index, value in enumerate(coords_list):
        if value == 0:
            if coords_list[index-1] > 0:
                value_before = coords_list[index-1]
                interp_start_index = index-1
                #print('interp_start_index: ', interp_start_index)
                #print('interp_start_value: ', value_before)
                #print('')

        if index < len(coords_list)-1:
            if value ==0:
                if coords_list[index+1] > 0:
                    interp_end_index = index+1
                    value_after = coords_list[index+1]
                    #print('interp_end_index: ', interp_end_index)
                    #print('interp_end_value: ', value_after)
                    #print('')

                    #now code to interpolate over the values
                    try:
                        interp_diff_index = interp_end_index - interp_start_index
                    except UnboundLocalError:
#                         print('the first value in list is 0, use the function start_value_cleanup to fix')
                        break
                    #print('interp_diff_index is:', interp_diff_index)

                    new_values = np.linspace(value_before, value_after, interp_diff_index)
                    #print(new_values)

                    interp_index = interp_start_index+1
                    for x in range(interp_diff_index):
                        #print('interp_index is:', interp_index)
                        #print('new_value should be:', new_values[x])
                        coords_list[interp_index] = new_values[x]
                        interp_index +=1
        if index == len(coords_list)-1:
            if value ==0:
                for x in range(30):
                    coords_list[index-x] = coords_list[index-30]
                    #print('')
#     print('function exiting')
    return(coords_list)

def load_input_params(pp_path, file, mouse_session_recording):
    input_params_path = os.path.join(pp_path, file, 'trainingData', f'params_{mouse_session_recording}.json')
    input_config = load_json(input_params_path)
    behav_time_interval_start = input_config['time_span'][0]
    print(f"      A corresponding time span has been found. Time span set to {behav_time_interval_start}")
    return behav_time_interval_start

def load_behav_data(dat_path, behav_time_interval_start):
    print("\nLOADING BEHAV DATA")
    behav_sync = pd.read_csv(os.path.join(dat_path, 'behav_sync', '2_task', 'Behav_Ephys_Camera_Sync.csv'))
    transitions = pd.read_csv(os.path.join(dat_path, 'behav_sync', '2_task', 'Transition_data_sync.csv'))

    behav_mask = (behav_sync.PokeIN_EphysTime > behav_time_interval_start[0]) & (behav_sync.PokeIN_EphysTime < behav_time_interval_start[1])
    poke_in_times = behav_sync[behav_mask].PokeIN_EphysTime - behav_time_interval_start[0]
    ports = behav_sync[behav_mask].Port
    print('done')
    return behav_sync, transitions, poke_in_times, ports,behav_mask

def Load_example_data(pp_path, file, tracking_path, dat_path, mouse_session_recording = '178_1_7'):
    # Load processed spike data
    latent_event_history_df_split, spikes_seq_type_adjusted, neuron_order, ordered_preferred_type, neuron_index, config = load_processed_spike_data(pp_path, file)
    
    # Load DLC tracking data
    print("\nLOADING DLC TRACKING DATA")
    back_head_centre, back_ports = load_tracking_data(tracking_path)
    
    # Load the timespan used for pppseq
    behav_time_interval_start = load_input_params(pp_path, file, mouse_session_recording)
    
    # Load behaviour data
    behav_sync, transitions, poke_in_times, ports,behav_mask = load_behav_data(dat_path, behav_time_interval_start)
    
    
    neuron_response_df = pd.read_csv(pp_path + file + r"\neuron_response.csv")
    bkgd_log_proportions_array = pd.read_csv(pp_path + file + r"\bkgd_log_proportions_array.csv")

    # Return all loaded data for further processing if needed
    return {
        "latent_event_history_df_split": latent_event_history_df_split,
        "spikes_seq_type_adjusted": spikes_seq_type_adjusted,
        "neuron_order": neuron_order,
        "ordered_preferred_type": ordered_preferred_type,
        "neuron_index": neuron_index,
        "config": config,
        "back_head_centre": back_head_centre,
        "back_ports": back_ports,
        "behav_sync": behav_sync,
        "transitions": transitions,
        "poke_in_times": poke_in_times,
        "ports": ports,
        "neuron_response_df" : neuron_response_df,
        "bkgd_log_proportions_array" : bkgd_log_proportions_array,
        "behav_time_interval_start" : behav_time_interval_start,
        "behav_mask" : behav_mask
    }

def plot_zoomed_example_raster(data,ordering,colors,timeframe):

    mask = (data["spikes_seq_type_adjusted"].timestamp>timeframe[0])*(data["spikes_seq_type_adjusted"].timestamp<timeframe[-1])

    ## neuron order:

    #define neuron order
    neuron_index,ordered_preferred_type = sortperm_neurons(data["bkgd_log_proportions_array"],data["config"],data["neuron_response_df"], sequence_ordering=ordering)
    # make a list of idndies for each neurons new position
    neuron_permute_loc = np.zeros(len(neuron_index))
    for i in range(len(neuron_index)):
        neuron_permute_loc[i] = int(list(neuron_index).index(i))

    neuron_permute_loc = np.array(shuffle(list(neuron_permute_loc.astype(int))))
    neuron_order = neuron_permute_loc[data["spikes_seq_type_adjusted"].neuron.values.astype(int)-1]

    # make a list of idndies for each neurons new position
    neuron_permute_loc = np.zeros(len(neuron_index))
    for i in range(len(neuron_index)):
        neuron_permute_loc[i] = int(list(neuron_index).index(i))
    #         neuron_order = neuron_permute_loc[unmasked_spikes_df.neuron-1]
    neuron_order = neuron_permute_loc[ data["spikes_seq_type_adjusted"].neuron.values.astype(int)-1]

    ## plotting:
    nrow = 1
    ncol = 1

    fig, ax = plt.subplots(nrow, ncol,figsize=(4, 5))

    # plot background in grey 
    background_keep_mask =  data["spikes_seq_type_adjusted"][mask].sequence_type_adjusted <= 0
    ax.scatter( data["spikes_seq_type_adjusted"][mask][background_keep_mask].timestamp, neuron_order[mask][background_keep_mask],marker = 'o', s=20, linewidth=0,color = 'grey' ,alpha=0.3)

    # plot spikes without background
    background_remove_mask = data["spikes_seq_type_adjusted"][mask].sequence_type_adjusted >= 0
    c_ = np.array(colors)[data["spikes_seq_type_adjusted"][mask][background_remove_mask].sequence_type_adjusted.values.astype(int)]
    # ## faster:
    ax.scatter( data["spikes_seq_type_adjusted"][mask][background_remove_mask].timestamp, neuron_order[mask][background_remove_mask],marker = 'o', s=20, linewidth=0,color = c_ ,alpha=1)

    return

def sortperm_neurons(bkgd_log_proportions_array,config,neuron_response_df, sequence_ordering=None, th=0.2):
    ## this is number of neurons in total
    N_neurons= bkgd_log_proportions_array.shape[1]
    ## number of sequences from json file 
    n_sequences = config["num_sequence_types"]
    # the 18 neuron params for each neuron from the last iteration
    all_final_globals = neuron_response_df.iloc[-N_neurons:]
    # this cuts it down to just the first 6 params - i think this correspond sto the first param for each seq type? response probABILITY - ie the chance that a neuron spikes in a given latent seq 
    resp_prop = np.exp(all_final_globals.values[:, :n_sequences])#
    # this takes the next 6 params - which i think are the offset values
    offset = all_final_globals.values[-N_neurons:, n_sequences:2*n_sequences]
    ## finds the max response value - ie. which seq it fits to? 
    peak_response = np.amax(resp_prop, axis=1)
    # then threshold the reponse
    has_response = peak_response > np.quantile(peak_response, th)
    # I thin this is the sequence that the neuron has the max response for: ie. we are ordering them by max response 
    preferred_type = np.argmax(resp_prop, axis=1)
    if sequence_ordering is None:
        # order them by max reponse 
        ordered_preferred_type = preferred_type
    else:
        #order them differnetly 
        ordered_preferred_type = np.zeros(N_neurons)#
        # loop through each sequence
        for seq in range(n_sequences):
            # where does  max repsone = user defined seque
            seq_indices = np.where(preferred_type == sequence_ordering[seq])
            # change order to different seq
            ordered_preferred_type[seq_indices] = seq

    # reorder the offset params according to max respsone
    preferred_delay = offset[np.arange(N_neurons), preferred_type]
    Z = np.stack([has_response, ordered_preferred_type+1, preferred_delay], axis=1)
    indexes = np.lexsort((Z[:, 2], Z[:, 1], Z[:, 0]))
    return indexes,ordered_preferred_type

def shuffle(aList):
    random.shuffle(aList)
    return aList
