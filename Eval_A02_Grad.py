###############################################################################
# Activate single threading (as much as possible)
###############################################################################
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import cv2
cv2.setNumThreads(0)

###############################################################################
# Primary imports
###############################################################################
import numpy as np
import pandas as pd
import gc
from enum import Enum
from collections.abc import Callable
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from General_A02 import *
from Test_A02 import *
import A02

###############################################################################
# Structs/classes/enums
###############################################################################

class TimingResult:
    def __init__(self, prefix, kernel_sizes, min_timings, scaled_min_timings, timing_stats):
        self.prefix = prefix
        self.kernel_sizes = kernel_sizes
        self.min_timings = min_timings
        self.scaled_min_timings = scaled_min_timings
        self.timing_stats = timing_stats
        
class EXP_NAMES(Enum):
    ODD_FAST = "Odd: Fast"
    ODD_FOUR ="Odd: Fourier"
    ODD_OPT = "Odd: Optimal"
    GAUSS_FAST = "Gauss: Fast"
    GAUSS_FOUR = "Gauss: Fourier"
    GAUSS_SEP_FAST = "Gauss: Separable (Fast)"
    GAUSS_SEP_FOUR = "Gauss: Separable (Fourier)"
    GAUSS_OPT = "Gauss: Optimal"

###############################################################################
# Evaluation settings
###############################################################################

# These settings should not be modified
TEST_IMAGE_FILEPATH = os.path.join("assign02", "images", "ben.png")
MAX_ODD_SIZE = 43
MAX_GAUSS_SIZE = 65

# These settings can be modified 
MAX_TRIAL_CNT = 30
PRINT_DEBUG = True
DO_GRAPH_SMOOTHING = False
GRAPH_MARKER = None # 'o'
GRAPH_USE_LOG_SCALE = True

TIMING_GRAPH_TITLE = "All Timings"    
TIMING_GRAPH_FILENAME = os.path.join(out_dir, "AllTimingsGraph.png")
    
STATS_GRAPH_TITLE = "All Timings: Overall Sum"
STATS_GRAPH_FILENAME = os.path.join(out_dir, "AllTimingsSumGraph.png")

STATS_GRAPH_CHOSEN_STAT = "SUM"
STATS_GRAPH_Y_AXIS_TITLE = "Sum of Timings (ns)"

CHOSEN_EXPERIMENTS = [
    EXP_NAMES.ODD_FAST,
    EXP_NAMES.ODD_FOUR,
    EXP_NAMES.ODD_OPT,
    EXP_NAMES.GAUSS_FAST,
    EXP_NAMES.GAUSS_FOUR,
    EXP_NAMES.GAUSS_SEP_FAST,
    EXP_NAMES.GAUSS_SEP_FOUR,
    EXP_NAMES.GAUSS_OPT
]

###############################################################################
# Kernel-creation functions
###############################################################################
def create_odd_kernels(start_size:int, max_size:int, inc_size:int) -> dict:    
    odd_kernels = []
    kernel_sizes = []
    for i in range(start_size, max_size+1, inc_size):
        odd_kernels.append(np.reshape(np.arange(i*i), (i,i)).astype("float64"))
        kernel_sizes.append(i)
                
    return odd_kernels, kernel_sizes

def create_gaussian_kernels(start_size:int, max_size:int, inc_size:int) -> dict:
    gauss_kernels = []
    kernel_sizes = []
    for i in range(start_size, max_size+1, inc_size):
        gauss_kernels.append(create_gaussian(i, i))
        kernel_sizes.append(i)
    
    return gauss_kernels, kernel_sizes

###############################################################################
# Printing, graphing, and file I/O functions
###############################################################################
def print_overall_stats(prefix, num_trials, timing_stats):
    print(f"{prefix} OVERALL STATS ({num_trials} trials):")
    for time_key in timing_stats:
        ns_time = timing_stats[time_key]
        ns_time /= TIME_SCALE
        print("*", time_key, ":", ns_time)  
        
def plot_timings(experiments, title, do_smoothing=False, marker='o', use_log_scale=False, save_image_filename=None):
    
    exp_keys = list(experiments.keys())
    color_list = ["blue", "red", "green", "orange", "purple", "magenta", "yellow", "cyan"]
    
    plt.figure(figsize=(10, 7))
    
    x_coords_set = set()
    
    for i in range(len(exp_keys)):   
        one_key = exp_keys[i]    
        one_result = experiments[one_key]
        x_coords = one_result.kernel_sizes
        x_coords_set.update(x_coords)
        one_timing = one_result.min_timings
        if do_smoothing:
            one_timing = savgol_filter(one_timing, window_length=5, polyorder=2)
            
        color_index = i % (len(color_list))
        chosen_color = color_list[color_index]
        
        label_name = one_key.name
        
        plt.plot(x_coords, one_timing, label=label_name, color=chosen_color, marker=marker, linewidth=2)
    
    plt.title(title)
    plt.xlabel('Kernel Size (N x N)')
    plt.ylabel('Minimum Execution Time (ns)')
    
    plt.xticks(list(x_coords_set))
    
    if use_log_scale:
        plt.yscale('log')
    
    plt.grid(True, linestyle='--', alpha=0.7)    
    plt.legend()
    plt.tight_layout()
    
    if save_image_filename is not None:
        plt.savefig(save_image_filename, dpi=300)    
        
    plt.show()
    
def plot_stats(experiments, title, use_log_scale=False, save_image_filename=None,
               chosen_stat = "SUM", y_axis_title="Sum of Timings (ns)"):
    
    exp_keys = list(experiments.keys())
    color_list = ["blue", "red", "green", "orange", "purple", "magenta", "yellow", "cyan"]
    
    plt.figure(figsize=(10, 7))
    experiment_names = []
    experiment_values = []
    experiment_colors = []
    
    for i in range(len(exp_keys)):   
        one_key = exp_keys[i]    
        one_result = experiments[one_key]
        
        experiment_names.append(one_key.value)        
        experiment_values.append(one_result.timing_stats[chosen_stat])
        
        color_index = i % (len(color_list))
        experiment_colors.append(color_list[color_index])           
        
    plt.bar(experiment_names, experiment_values, color=experiment_colors, capsize=5)
    
    plt.title(title)
    plt.xlabel("Experiment")
    plt.ylabel(y_axis_title)
    plt.xticks(rotation="vertical")
        
    if use_log_scale:
        plt.yscale("log")
         
    plt.legend()
    plt.tight_layout()
    
    if save_image_filename is not None:
        plt.savefig(save_image_filename, dpi=300)    
        
    plt.show()
    
###############################################################################
# Evaluation functions
###############################################################################
def evaluate_timings(   image_path:str, prefix: str, 
                        eval_kernels:list, 
                        kernel_sizes:list,
                        conv_func: Callable[..., np.ndarray],
                        max_num_trials:int, 
                        do_print:bool = False) -> np.array:
    
    if do_print:
        print("**", prefix, "*******************************")
        
    # Set up timings array
    min_timings = -1*np.ones((len(eval_kernels),), dtype=np.int64)
        
    # Set up indices for kernels
    kernel_indices = np.arange(len(eval_kernels))
        
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Get "pixel-work" arrays
    pixel_work = np.zeros((len(eval_kernels),), dtype="int")
    for k in range(len(eval_kernels)):
        k_shape = eval_kernels[k].shape
        pixel_work[k] = k_shape[0]*k_shape[1]
            
    # Global warmup
    biggest_kernel = eval_kernels[-1]
    conv_func(image, biggest_kernel, convert_uint8=True)               

    # For each trial...
    for t in range(max_num_trials):
        # Shuffle indices
        np.random.shuffle(kernel_indices)
        
        # For each kernel...
        for k in range(len(eval_kernels)):
            # Get kernel
            kernel_index = kernel_indices[k]
            kernel = eval_kernels[kernel_index]
            
            # Disable garbage collector
            gc.disable()             
            
            # Start timing    
            start_time = start_timing()  
            
            # Do the thing!
            conv_func(image, kernel, convert_uint8=False)  
            
            # Calculate time taken
            trial_time = end_timing(start_time)
            
            # Re-enable garbage collector
            gc.enable() 
            
            # Did we set anything yet?
            if min_timings[kernel_index] < 0:
                min_timings[kernel_index] = trial_time
            else:
                # Only store the smaller one
                min_timings[kernel_index] = np.minimum(trial_time, min_timings[kernel_index])
            
            # Print?            
            if do_print:
                display_time = trial_time / TIME_SCALE
                print(f"{prefix} (trial {t}, {kernel.shape} kernel): {display_time:.8f}")
            
    # Get scaled timing
    scaled_min_timings = min_timings / pixel_work
    
    # Calculate overall stats    
    timing_stats = {}    
    timing_stats["AVE"] = np.mean(min_timings)
    timing_stats["SUM"] = np.sum(min_timings) 
    timing_stats["SCALED_SUM"] = np.sum(scaled_min_timings)
    timing_stats["SCALED_AVE"] = np.mean(scaled_min_timings)
    
    if do_print:
        print_overall_stats(prefix, max_num_trials, timing_stats)
                    
    if do_print:
        print("*********************************************")
          
    return TimingResult(prefix, kernel_sizes, min_timings, scaled_min_timings, timing_stats)

def perform_experiments(all_experiment_params, chosen_experiments, 
                        timing_graph_title, timing_graph_filename,
                        stat_graph_title, stat_graph_filename,
                        stat_graph_chosen_stat, stat_graph_y_axis_title):
    # Prepare output
    experiment_output = {}    
           
    # Set up params we always use
    standard_params = {
        "image_path": TEST_IMAGE_FILEPATH,
        "max_num_trials": MAX_TRIAL_CNT,
        "do_print": PRINT_DEBUG
    }
    
    # For each experiment...
    for key_name in chosen_experiments:
        eval_params = all_experiment_params[key_name].copy()
        eval_params.update(standard_params)
        experiment_output[key_name] = evaluate_timings(**eval_params)
    
    # Plot values    
    plot_timings(experiment_output, 
                 title=timing_graph_title,
                 do_smoothing=DO_GRAPH_SMOOTHING,
                 marker=GRAPH_MARKER,
                 use_log_scale=GRAPH_USE_LOG_SCALE,
                 save_image_filename=timing_graph_filename)    
    
    plot_stats(experiment_output, 
               title=stat_graph_title, 
               use_log_scale=GRAPH_USE_LOG_SCALE, 
               save_image_filename=stat_graph_filename,
               chosen_stat=stat_graph_chosen_stat,
               y_axis_title=stat_graph_y_axis_title)

###############################################################################
# Main
###############################################################################
     
def main():
    # Make output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    # Create kernels
    odd_kernels, odd_sizes = create_odd_kernels(3, MAX_ODD_SIZE, 2)
    gauss_kernels, gauss_sizes = create_gaussian_kernels(3, MAX_GAUSS_SIZE, 2)
    
    # Create both variants of separable
    def do_conv_sep_fourier(image, kernel, alpha=1.0, beta=0.0, convert_uint8=True):
        return A02.do_convolution_separable(image, kernel, alpha, beta, convert_uint8, conv_func=A02.do_convolution_fourier)
        
    def do_conv_sep_fast(image, kernel, alpha=1.0, beta=0.0, convert_uint8=True):
        return A02.do_convolution_separable(image, kernel, alpha, beta, convert_uint8, conv_func=A02.do_convolution_fast)
             
    # Set up each experiment params
    all_experiment_params = {
        EXP_NAMES.ODD_FAST:         {"prefix": EXP_NAMES.ODD_FAST.name, "eval_kernels": odd_kernels, "kernel_sizes": odd_sizes,
                                    "conv_func": A02.do_convolution_fast},
        
        EXP_NAMES.ODD_FOUR:         {"prefix": EXP_NAMES.ODD_FOUR.name, "eval_kernels": odd_kernels, "kernel_sizes": odd_sizes,
                                    "conv_func": A02.do_convolution_fourier},
        
        EXP_NAMES.ODD_OPT:          {"prefix": EXP_NAMES.ODD_OPT.name, "eval_kernels": odd_kernels, "kernel_sizes": odd_sizes,
                                    "conv_func": A02.do_convolution_optimal},
        
        EXP_NAMES.GAUSS_FAST:       {"prefix": EXP_NAMES.GAUSS_FAST.name, "eval_kernels": gauss_kernels, "kernel_sizes": gauss_sizes,
                                    "conv_func": A02.do_convolution_fast},
        
        EXP_NAMES.GAUSS_FOUR:       {"prefix": EXP_NAMES.GAUSS_FOUR.name, "eval_kernels": gauss_kernels, "kernel_sizes": gauss_sizes,
                                    "conv_func": A02.do_convolution_fourier},
        
        EXP_NAMES.GAUSS_SEP_FAST:   {"prefix": EXP_NAMES.GAUSS_SEP_FAST.name, "eval_kernels": gauss_kernels, "kernel_sizes": gauss_sizes,
                                    "conv_func": do_conv_sep_fast},
        
        EXP_NAMES.GAUSS_SEP_FOUR:   {"prefix": EXP_NAMES.GAUSS_SEP_FOUR.name, "eval_kernels": gauss_kernels, "kernel_sizes": gauss_sizes,
                                    "conv_func": do_conv_sep_fourier},
        
        EXP_NAMES.GAUSS_OPT:        {"prefix": EXP_NAMES.GAUSS_OPT.name, "eval_kernels": gauss_kernels, "kernel_sizes": gauss_sizes,
                                    "conv_func": A02.do_convolution_optimal},        
    }
    
    # Actually perform experiments      
    perform_experiments(all_experiment_params, CHOSEN_EXPERIMENTS, 
                        TIMING_GRAPH_TITLE, TIMING_GRAPH_FILENAME,
                        STATS_GRAPH_TITLE, STATS_GRAPH_FILENAME,
                        STATS_GRAPH_CHOSEN_STAT, STATS_GRAPH_Y_AXIS_TITLE)
                
if __name__ == "__main__":
    main()
    