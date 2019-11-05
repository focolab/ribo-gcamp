import os
import numpy as np
import itertools
import re

def rois(path):
    """
    import roi centers from (fiji Unzipped RoiSet folder) one frame/zplane in timeseries
    Create a circle around this pixel with radius = 3 pixels to use as roimask(x,y)
    """
    roifiles = list(filter(lambda f:f.endswith('.roi'), os.listdir(path)))
    roi,roimask,roimask_sorted = [{} for i in range(3)]
    for n in roifiles:
        roi[n]=n.split('.roi')[0]
        roi[n]=re.split('-0*',roi[n])
        roimask[np.int(roi[n][0])-1,n] = list(itertools.product(list(range(np.int(roi[n][2])-3,np.int(roi[n][2])+3)), list(range(np.int(roi[n][1])-3,np.int(roi[n][1])+3))))
        #correct for offset by 1 from numZ: np.int(roi[n][0])-1
    n_sorted = sorted(roimask.keys())
    for k in n_sorted:
        roimask_sorted[k] = roimask[k]
    return roimask_sorted