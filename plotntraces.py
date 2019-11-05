import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class mkplots(object):
    def __init__(self,path,numZ,exptime):
        self.path=path
        self.numZ=numZ
        self.exptime=exptime
            
    def gettraces(self):
        listfiles = list(filter(lambda f:f.endswith('.pkl'), os.listdir(self.path)))
        fnum = {}
        rn=[]
        for s in listfiles:
            s = os.path.join(self.path,s)
            fnum[s]=s.split('.pkl')[0]
            fnum[s]=fnum[s].split('roi ')[1]
            rn.append(fnum[s])
        rn=sorted(rn)
        fn = sorted(fnum.keys(),key=fnum.get)
        traces={}
        ntrace=[]
        for n,r in enumerate(rn):
            traces[r] = pd.read_pickle(fn[n])
            ntrace.append(traces[r])
        self.rn = rn
        self.ntrace = ntrace
        return ntrace, rn

    def ntracesperplot(self):
        iter=len(self.rn)
        lastntracesperplot=len(self.rn) % iter
        #print(lastntracesperplot)
        #xlast=np.float(len(self.ntrace[0]))*self.exptime*self.numZ    
        for p in range(0,len(self.rn),iter):
            fig,ax = plt.subplots(iter,1,sharex=True,num=p,squeeze=True,figsize=(8,6))
            rem_tr = len(self.rn) - (p+1*iter)
            print(iter)
            print(rem_tr)
            for n in range(iter):
                ax[n].plot(self.ntrace[p+n],linewidth=0.3)
                # ax[n].set_xlim([0,800])
                ax[n].text(0.95, 0.95, self.rn[p+n], transform=ax[n].transAxes, fontsize=8, verticalalignment='top',horizontalalignment='right')
                ax[np.int(iter/2)].set_ylabel("dF/F",fontsize=8)
                xtick=range(0,len(self.ntrace[p+n]),np.int(100/(self.exptime*self.numZ)))
                xtl=range(0,(np.int(np.float(len(self.ntrace[p+n]))*self.exptime*self.numZ)),100)
                ax[n].set_xticks(xtick)
                ax[n].set_xticklabels(xtl, fontsize=8)
                ax[n].axes.tick_params(axis='y',labelsize=8)
            fig.suptitle("time (s)",x=0.5,y=0.03,fontsize=8)
            ntr = "allrois"+str(p)
            fig.savefig(self.path+ntr+".pdf",dpi=1200,format='pdf')
            # plt.show()
            plt.clf()
            if rem_tr == lastntracesperplot:
                iter = lastntracesperplot
        return lastntracesperplot, rem_tr, xtick, xtl