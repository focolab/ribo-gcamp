import os
import plotntraces as pn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
from scipy.signal import find_peaks as fp
import tifffile as tf
import timeseries_risetime as ts

class risetimes(object):
    def __init__(self,path,tifpath,roipath,bckpath,numZ,exptime,window,wormid):
        tr = pn.mkplots(path,numZ,exptime)
        zroi = ts.Timeseries(tifpath, roipath, bckpath, numZ, exptime)
        self.zt0=tf.imread(tifpath,key=range(numZ))
        self.zr=zroi.ZorganizedROIs()
        self.traces = tr.gettraces()[0]
        self.rn = tr.gettraces()[1]
        self.window = window
        self.numZ = numZ
        self.exptime = exptime
        self.path = path
        self.wormid=wormid

    def calcrisetimes(self):
        """
        Within a range of x values above a 2nd deriv threshold, find peaks and interpolate between 2 peaks that are above a magnitude threshold
        Find 90th and 10th percentile of the resulting magnitudes
        Obtain x coordinates from the original trace for the nearest timepoints of the magnitudes that qualify
        Calculate risetime by measuring how many timepoints are between these two peaks
        """
        if os.path.exists(self.path+"RisetimesAndRange_allauto"):
            shutil.rmtree(self.path+"RisetimesAndRange_allauto")
        os.mkdir(self.path+"RisetimesAndRange_allauto")
        
        allpeaks,diff2,max,large,gaps,begins,ends,peakfp,tallpeak,difftallpeak,maxpeak,allincpeaks,includepeaks,interp,xval,x,y,peak,baseline,self.perc10,self.perc90,self.risetime=({} for i in range(22))
        self.pidx=[]
        for r in range(len(self.traces)):
            diff2[r] = np.diff(self.traces[r][0],n=2,axis=0)
            allpeaks[r] = fp(self.traces[r][0])
            large[r] = np.where(np.abs(diff2[r]) > np.max(np.abs(diff2[r]))/4)[0]
            gaps[r] = np.logical_and((np.diff(large[r]) > self.window),(np.diff(large[r]) < self.window*200))
            begins[r] = np.insert(large[r][1:][gaps[r]], 0, large[r][0])
            ends[r] = np.append(large[r][:-1][gaps[r]],large[r][-1])
            if len(begins[r]) >= 1:
                for p in range(len(begins[r])):
                    if ((ends[r][p]-begins[r][p]>=self.window) and (ends[r][p]-begins[r][p]<=self.window*200)):
                        max[(r,p)] = np.where(self.traces[r][0] == np.max(self.traces[r][0][begins[r][p]:ends[r][p]]))[0][0]
                        if max[(r,p)] >= begins[r][p]:
                            peakfp[(r,p)] = np.intersect1d(allpeaks[r][0],range(begins[r][p],max[(r,p)]))
            tallpeak[r]=[]
            for k,v in peakfp.items():
                if type(k) in [list,tuple,dict] and r in k:
                    tallpeak[r].append(v)
            tallpeak[r] = np.hstack(tallpeak[r])
            difftallpeak[r] = np.diff(self.traces[r][0][tallpeak[r]])
            maxpeak[r] = np.where(difftallpeak[r]==np.max(difftallpeak[r]))[0]
            allincpeaks[r] = np.where(difftallpeak[r]>=maxpeak[r]/2)[0].tolist()
            allincpeaks[r].extend(allincpeaks[r][i]+1 for i in range(len(allincpeaks[r])))
            includepeaks[r] = tallpeak[r][allincpeaks[r]].astype(int)

            for i in range(len(includepeaks[r])):
                plt.scatter(includepeaks[r][i],self.traces[r][0][includepeaks[r][i]])
            plt.plot(self.traces[r][0])
            plt.savefig(self.path+"RisetimesAndRange_allauto/"+self.wormid+self.rn[r]+"risetimefig.pdf",dpi=1200,format='pdf')
            plt.show()
            plt.clf()

            for i in range(0,len(includepeaks[r])-1,2):
                xval[(r,i)] = np.linspace(includepeaks[r][i],includepeaks[r][i+1],5000)
                y[(r,i)] = self.traces[r][0][includepeaks[r][i]:includepeaks[r][i+1]]
                x[(r,i)] = range(includepeaks[r][i],includepeaks[r][i+1])
                interp[(r,i)] = np.interp(xval[(r,i)],x[(r,i)],y[(r,i)])
                peak[(r,i)] = includepeaks[r][i+1]+np.int(np.min(np.where(interp[(r,i)]==np.max(interp[(r,i)]))[0])*((includepeaks[r][i+1]-includepeaks[r][i])/5000))
                baseline[(r,i)] = includepeaks[r][i]+np.int(np.max(np.where(interp[(r,i)]==np.min(interp[(r,i)]))[0])*((includepeaks[r][i+1]-includepeaks[r][i])/5000))
                self.perc10[(r,i)] = [np.int(np.percentile(range(baseline[(r,i)],peak[(r,i)]),10))]
                self.perc90[(r,i)] = [np.int(np.percentile(range(baseline[(r,i)],peak[(r,i)]),90))]
                self.risetime[(r,i)] = [(self.perc90[(r,i)][0]-self.perc10[(r,i)][0])*(self.numZ*self.exptime)]
                self.pidx.append((r,i))

            calcrtimesv=pd.DataFrame.from_dict([self.wormid,self.rn[r],self.traces[r][0],self.perc10,self.perc90,self.risetime])
            calcrtimesv.to_pickle(self.path+"RisetimesAndRange_allauto/"+self.wormid+self.rn[r]+"TraceAndRtimeStuff.pkl")

        return begins, ends, large, gaps, max, diff2, allpeaks, peakfp, tallpeak, difftallpeak, maxpeak, allincpeaks, includepeaks, xval, x, y, interp, peak,baseline,self.perc10, self.perc90, self.risetime

    def plotrisetimesontrace(self):
        """
        on every rise, note the risetime (90%-10% of baseline value)
        """
        z,n=({} for i in range(2))  
        for p in range(len(self.rn)):
            fig,ax = plt.subplots(2,1,num=p,squeeze=True,figsize=(8,6))
            rem_tr = len(self.rn) - p
            print(rem_tr)
            ax[0].plot(self.traces[p],linewidth=0.3,marker='.',markersize=1)
            ax[0].text(0.95, 0.95, self.rn[p], fontsize=8, verticalalignment='top',horizontalalignment='right')#transform=ax[p].transAxes,
            ax[0].text(0.8, 0.8, self.numZ*self.exptime, fontsize=8, verticalalignment='top',horizontalalignment='right')#transform=ax[p].transAxes,
            # ax[0].set_xlim([0,4000])
            for key in self.pidx:
                if key[0] == p:
                    ax[0].text(np.max(self.perc10[key]),0.95,np.round(self.risetime[key][0],decimals=2), fontsize=8)
                    ax[0].plot(self.perc10[key][0],self.traces[key[0]][0][self.perc10[key][0]],'>')
                    ax[0].plot(self.perc90[key][0],self.traces[key[0]][0][self.perc90[key][0]],'<')               
            xlim=ax[0].get_xlim()
            ax[0].set_xticks(range(np.int(xlim[0]),np.int(xlim[1]),np.int(50/(self.exptime*self.numZ))))
            xtl=range(np.int(xlim[0]*self.exptime*self.numZ),np.int(xlim[1]*self.exptime*self.numZ),50)
            ax[0].set_xticklabels(xtl)#, fontsize=8
            ax[0].axes.tick_params(axis='y')#, labelsize=8
            ax[0].set_xlabel("time (s)")#, fontsize=8
            ax[0].set_ylabel("dF/F",fontsize=8)
            z[p]=self.rn[p].split('r')[0]
            z[p]=np.int(z[p].split('z')[1])
            n[p]=np.int(self.rn[p].split('r')[1])
            ax[1].imshow(self.zt0[z[p]])
            for pix in range(len(self.zr[0][z[p]][n[p]])):
                ax[1].scatter(self.zr[0][z[p]][n[p]][pix][0],self.zr[0][z[p]][n[p]][pix][1],marker='.')     
            fig.savefig(self.path+"RisetimesAndRange_allauto/"+self.wormid+self.rn[p]+"RisetimeAndImgfig.pdf",dpi=1200,format='pdf')
            plt.clf()
            plt.close()
        return rem_tr, xtl