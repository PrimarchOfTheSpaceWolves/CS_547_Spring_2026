import time

TIME_SCALE = 1_000_000
TIME_TYPE = "seconds"

def start_timing():
    return time.perf_counter_ns()

def end_timing(start_time):    
    end_time = time.perf_counter_ns()
    time_diff = end_time - start_time        
    return time_diff

def end_timing_print(prefix, start_time):    
    time_diff = end_timing(start_time)
    time_diff /= TIME_SCALE
    print(f"{prefix} TIME: {time_diff:.4f} {TIME_TYPE}")
    return time_diff
