import os
import plotntraces as pn
import numpy as np
import scipy.signal as sc
from scipy.ndimage.filters import gaussian_filter1d
import pandas as pd
import matplotlib.pyplot as plt
import shutil
from scipy.signal import find_peaks as fp

class risetimes(object):
    def __init__(self,path,numZ,exptime,window):
        tr = pn.mkplots(path,numZ,exptime)
        self.traces = tr.gettraces()[0]
        self.rn = tr.gettraces()[1]
        self.window = window
        self.numZ = numZ
        self.exptime = exptime
        self.path = path
    
    def calcrisetimes(self):
        """
        Within a range of x values, find peaks and interpolate between 2 peaks with a magnitude threshold
        Find 90th and 10th percentile of the resulting magnitudes
        Obtain x coordinates of the original trace for the nearest outward timepoints of the magnitudes that qualify
        Calculate risetime by measuring how many timepoints are between these two peaks
        """
        if os.path.exists(self.path+"RisetimesAndRange_allauto"):
            shutil.rmtree(self.path+"RisetimesAndRange_allauto")
        os.mkdir(self.path+"RisetimesAndRange_allauto")
        
        allpeaks,diff2,rise,max,large,gaps,begins,ends,peakrise,peakrise1,peakfp,tallpeak,tallpeak1,tallpeak2, difftallpeak2,maxpeak, allpeaks,peakrise2,includepeaks,svpeak,svbegins,perc10,perc90,risetime=({} for i in range(24))
        # pidx=[]
        for r in range(len(self.traces)):
            diff2[r] = np.diff(self.traces[r][0],n=2,axis=0)
            allpeaks[r] = fp(self.traces[r][0])
            rise[r] = np.where(np.abs(np.diff(self.traces[r][0])) > np.max(np.abs(np.diff(self.traces[r][0])))/10)[0]
            large[r] = np.where(np.abs(diff2[r]) > np.max(np.abs(diff2[r]))/10)[0]
            gaps[r] = np.logical_and((np.diff(large[r]) > self.window),(np.diff(large[r]) < self.window*200))
            begins[r] = np.insert(large[r][1:][gaps[r]], 0, large[r][0])
            ends[r] = np.append(large[r][:-1][gaps[r]],large[r][-1])
            if len(begins[r]) >= 1:
                for p in range(len(begins[r])):
                    if ((ends[r][p]-begins[r][p]>=self.window) and (ends[r][p]-begins[r][p]<=self.window*200)):
                        max[(r,p)] = np.where(self.traces[r][0] == np.max(self.traces[r][0][begins[r][p]:ends[r][p]]))[0][0]
                        if max[(r,p)] >= begins[r][p]:
                            peakrise[(r,p)] = np.intersect1d(rise[r],range(begins[r][p],max[(r,p)]))
                            peakfp[(r,p)] = np.intersect1d(allpeaks[r][0],range(begins[r][p],max[(r,p)]))
            tallpeak1[r]=[]
            peakrise1[r]=[]
            for k,v in peakfp.items():
                if type(k) in [list,tuple,dict] and r in k:
                    tallpeak1[r].append(v)
            tallpeak2[r] = np.hstack(tallpeak1[r])
            difftallpeak2[r] = np.diff(self.traces[r][0][tallpeak2[r]])
            maxpeak[r] = np.where(difftallpeak2[r]==np.max(difftallpeak2[r]))[0]
            allpeaks[r] = np.where(difftallpeak2[r]>=maxpeak[r]/5)[0]
            allpeaks[r] = allpeaks[r].tolist()
            allpeaks[r].extend(allpeaks[r][i]+1 for i in range(len(allpeaks[r])))
            includepeaks[r] = tallpeak2[r][allpeaks[r]].astype(int)

            for i in range(len(includepeaks[r])):
                plt.scatter(includepeaks[r][i],self.traces[0][0][includepeaks[r][i]])
            plt.plot(self.traces[0][0])
            plt.show()

            

            for k,v in peakrise.items():
                if type(k) in [list,tuple,dict] and r in k:
                    peakrise1[r].append(v)
            peakrise2[r]=np.hstack(peakrise1[r])
                # tallpeak[r].append(np.max(np.diff(self.traces[r][0][tallpeak1[r]])))#[x]] for x in range(len(tallpeak1[r])))))
                # includepeaks[r].append(peakfp[(r,p)][np.where(np.diff(self.traces[r][0][peakfp[(r,p)]])>=np.max(np.diff(self.traces[r][0][peakfp[(r,:)]]))/2)[0]])
                            # if len(peakfp[(r,p)])>=1:
                            #     if (np.any(includepeaks[(r,p)] >= begins[r][p]+self.window)):
                            #         svbegins[(r,p)] = includepeaks[]
                            #         # peakfp[(r,p)][np.where(np.diff(self.traces[r][0][peakfp[(r,p)]])==np.max(np.diff(self.traces[r][0][peakfp[(r,p)]])))[0]]
                            #         svpeak[(r,p)] = peakfp[(r,p)][np.where(np.diff(self.traces[r][0][peakfp[(r,p)]])==np.max(np.diff(self.traces[r][0][peakfp[(r,p)]])))[0]+1]#begins[r][p]
                                    # perc10[(r,p)] = [np.int(np.percentile(range(svbegins[(r,p)],svpeak[(r,p)]),10))]
                                    # perc90[(r,p)] = [np.int(np.percentile(range(svbegins[(r,p)],svpeak[(r,p)]),90))]
                                    # risetime[(r,p)] = [(perc90[(r,p)][0]-perc10[(r,p)][0])*(self.numZ*self.exptime)]
                                    # pidx.append((r,p))
            
            # peaks[r] = fp(self.traces[r])
        return begins, ends, large, gaps, max, diff2, allpeaks, rise, peakrise, peakrise1, peakfp, tallpeak1, tallpeak2, maxpeak, allpeaks, includepeaks#, svpeak, svbegins