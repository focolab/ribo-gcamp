import numpy as np
from scipy.signal import find_peaks

def risetimes(timeseries_vector):

    # find peaks
    rise_end_frames = find_peaks(timeseries_vector)

    rise_start_frames=np.zeros(len(peaks))

    times=np.zeros(len(peaks))

    return times,rise_start_frames,rise_end_frames