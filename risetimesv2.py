import os
import plotntraces as pn
import numpy as np
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
        Within a range of x values above a 2nd deriv threshold, find peaks and interpolate between 2 peaks that are above a magnitude threshold
        Find 90th and 10th percentile of the resulting magnitudes
        Obtain x coordinates from the original trace for the nearest timepoints of the magnitudes that qualify
        Calculate risetime by measuring how many timepoints are between these two peaks
        """
        if os.path.exists(self.path+"RisetimesAndRange_allauto"):
            shutil.rmtree(self.path+"RisetimesAndRange_allauto")
        os.mkdir(self.path+"RisetimesAndRange_allauto")
        
        allpeaks,diff2,max,large,gaps,begins,ends,peakfp,tallpeak,difftallpeak,maxpeak,allincpeaks,includepeaks,interp,xval,x,y,peak,baseline,svpeak,svbegins,perc10,perc90,risetime=({} for i in range(24))
        pidx=[]
        for r in range(len(self.traces)):
            diff2[r] = np.diff(self.traces[r][0],n=2,axis=0)
            allpeaks[r] = fp(self.traces[r][0])
            large[r] = np.where(np.abs(diff2[r]) > np.max(np.abs(diff2[r]))/10)[0]
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
            allincpeaks[r] = np.where(difftallpeak[r]>=maxpeak[r]/5)[0].tolist()
            allincpeaks[r].extend(allincpeaks[r][i]+1 for i in range(len(allincpeaks[r])))
            includepeaks[r] = tallpeak[r][allincpeaks[r]].astype(int)

            for i in range(len(includepeaks[r])):
                plt.scatter(includepeaks[r][i],self.traces[0][0][includepeaks[r][i]])
            plt.plot(self.traces[0][0])
            plt.show()

            for i in range(0,len(includepeaks[r])-1,2):
                xval[(r,i)] = np.linspace(includepeaks[r][i],includepeaks[r][i+1],5000)
                y[(r,i)] = self.traces[r][0][includepeaks[r][i]:includepeaks[r][i+1]]
                x[(r,i)] = range(includepeaks[r][i],includepeaks[r][i+1])
            interp[(r,i)] = np.interp(xval[(r,i)],x[(r,i)],y[(r,i)])
            peak[(r,i)] = includepeaks[r][i]+np.int(np.min(np.where(interp[(r,i)]==np.max(interp[(r,i)]))[0])*((includepeaks[r][i+1]-includepeaks[r][i])/5000))
            baseline[(r,i)] = includepeaks[r][i]+np.int(np.max(np.where(interp[(r,i)]==np.min(interp[(r,i)]))[0])*((includepeaks[r][i+1]-includepeaks[r][i])/5000))
            perc10[(r,i)] = [np.int(np.percentile(range(baseline[(r,i)],peak[(r,i)]),10))]
            perc90[(r,i)] = [np.int(np.percentile(range(baseline[(r,i)],peak[(r,i)]),90))]
            risetime[(r,i)] = [(perc90[(r,i)][0]-perc10[(r,i)][0])*(self.numZ*self.exptime)]
            pidx.append((r,i))

        return begins, ends, large, gaps, max, diff2, allpeaks, peakfp, tallpeak, difftallpeak, maxpeak, allincpeaks, includepeaks, xval, x, y, interp, peak,baseline,perc10, perc90, risetime

    def plotrisetimesontrace(self):
        """
        on every rise, note the risetime (90%-10% of baseline value)
        """
        z,n,val,addrtime,rtimebegins,rtimeends=({} for i in range(6))
        #print(lastntracesperplot)
        #xlast=np.float(len(self.ntrace[0]))*self.exptime*self.numZ    
        for p in range(len(self.rn)):
            fig,ax = plt.subplots(iter,1,sharex=True,num=p,squeeze=True,figsize=(8,6))
            rem_tr = len(self.rn) - p
            print(rem_tr)
            ax[0].plot(self.traces[p],linewidth=0.3,marker='.',markersize=1)
            ax[0].text(0.95, 0.95, self.rn[p], transform=ax[p].transAxes, fontsize=8, verticalalignment='top',horizontalalignment='right')
            ax[0].text(0.8, 0.8, self.numZ*self.exptime, transform=ax[p].transAxes, fontsize=8, verticalalignment='top',horizontalalignment='right')
            ax[0].set_xlim([0,4000])
            # for key in self.pidx:
            #     if key[0] == p:
            #         ax[0].text(np.max(self.peak[key]),0.95,np.round(self.risetime[key][0],decimals=2), fontsize=8)
            #         ax[0].plot(self.perc10[key][0],self.traces[key[0]][0][self.perc10[key][0]],'>')
            #         ax[0].plot(self.perc90[key][0],self.traces[key[0]][0][self.perc90[key][0]],'<')               
            ax[0].set_ylabel("dF/F",fontsize=8)
            cursor = Cursor(ax[p], useblit=True, color='k', linewidth=1)
            zoom_ok = False
            print('\nZoom or pan to view, \npress spacebar when ready to click:\n')
            while not zoom_ok:
                zoom_ok=plt.waitforbuttonpress(timeout=5)
            print('Click once to select timepoint:')
            val=plt.ginput(n=-1,timeout=0,show_clicks=True,mouse_add=1,mouse_pop=2,mouse_stop=3)
            addrtime,rtimebegins,rtimeends=([] for i in range (3))
            for num in range(0,len(val),2):
                ax[0].plot(val[num][0],val[num][1],'>')
                ax[0].plot(val[num+1][0],val[num+1][1],'<')
                ax[0].text(val[num][0],0.95,np.round(((val[num+1][0]-val[num][0])*(self.numZ*self.exptime)),decimals=2),fontsize=8)
                rtimebegins.append(val[num][0])
                rtimeends.append(val[num+1][0])
                addrtime.append(np.round(((val[num+1][0]-val[num][0])*(self.numZ*self.exptime)),decimals=2))
            xtick=range(0,4000,np.int(20/(self.exptime*self.numZ)))
            xtl=range(0,np.int(np.float(4000)*self.exptime*self.numZ),20)
            ax[0].set_xticks(xtick)
            ax[0].set_xticklabels(xtl, fontsize=8)
            ax[0].axes.tick_params(axis='y', labelsize=8)
            ax[0].set_xlabel("time (s)", fontsize=8)
            z[p]=self.rn[p].split('r')[0]
            z[p]=self.rn[p].split('z')[1]
            n[p]=self.rn[p].split('r')[1]
            ax[1].imshow(self.zt0[p])
            for pix in range(len(self.zr[p][n])):
                ax[1].scatter(self.zr[p][n][pix][0],self.zr[p][n][pix][1],marker='.',markersize=1)            
            ntr = "risetimezoom"+str(p)
            fig.savefig(self.path+"RisetimesAndRange/"+ntr+".pdf",dpi=1200,format='pdf')
            plt.clf()
            addrtimesv=pd.DataFrame.from_dict([rtimebegins,rtimeends,addrtime])
            addrtimesv.to_pickle(self.path+"RisetimesAndRange/addRisetimeAndRange.pkl")
        return rem_tr, xtick, xtl