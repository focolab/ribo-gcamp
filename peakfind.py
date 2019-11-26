import numpy as np

def peakfind(vec, thresh=1, relthreshflag = True, peaksearchingmode=1):
# find the peaks and valleys of a time series using threshold tracking
#
#    [PEAKS,VALLEYS] = peakfind(VEC,THRESH,PEAKSEARCHMODESTART) 
#
#    input:
#    VEC is a vector time series
#    THRESH is an absolute threshold criterion (scalar)
#
#    output:
#    PEAKS and VALLEYS are lists ofindices
#
#    if peaksearchingmode=1 then first search for a peak as we
#    scan from left to right
#    else if peaksearchingmode ~= 1 otherwise first search for a valley
#    
#    by default, search for a peak first (say, if initial signal deflection 
#    is positive)
#
#   hints:
#    len(PEAKS) gives number of peaks
#    len(PEAKS) and len(VALLEYS) differ by at most 1
#    peaks and valleys always alternate    
#
#    saul.kato@ucsf.edu
#

  if relthreshflag:
    thresh = thresh * (np.max(vec) - np.min(vec))


  peaks = []
  valleys = []
  max_tracker_val = -np.inf
  min_tracker_val = np.inf
  max_tracker_index = np.nan
  min_tracker_index = np.nan

  for i in range(len(vec)):
      
    vi=vec[i]
    
    # update max_tracker_val if needed
    if vi > max_tracker_val:
      max_tracker_val=vi
      max_tracker_index=i

    # update min_tracker_val if needed
    if vi < min_tracker_val:
      min_tracker_val=vi
      min_tracker_index=i
    
    if peaksearchingmode==1:
      if vi < (max_tracker_val - thresh):
        peaks.append(max_tracker_index) # add entry to peaks
        min_tracker_index=i  # move up min tracker to current time
        min_tracker_val=vi
        peaksearchingmode=0  # switch to valley searching
    else:
      if vi > (min_tracker_val + thresh):
        valleys.append(min_tracker_index) # add entry to valleys
        max_tracker_index=i  # move up max tracker to current time
        max_tracker_val=vi
        peaksearchingmode=1  # switch to peak searching
    
  return peaks, valleys
