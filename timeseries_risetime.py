import os
import sortfnames as sortedfn
from tiffreader import TiffReader as tfread
import numpy as np
import ImportRois as improi
#import imreg_dft as ird
import pandas as pd
import matplotlib.pyplot as plt

class Timeseries(object):
    def __init__(self, path, roipath, bckpath, numZ, exptime):
        # self.fnames = sortedfn.sortfn(path)
        self.roimasks = improi.rois(roipath)
        self.bck = improi.rois(bckpath)
        # self.tiffs = tfread(f=self.fnames)
        self.numZ = numZ #number of z positions; 1 if singleplane
        self.exptime = exptime #in seconds
        self.path = path

    def ZorganizedTimepoints(self):
        """
        request z organized timepoints across OME Tiff files
        """
        zTidx,zTchunk,zT0 = ({} for i in range(3))
        zTidx = [dict(T=list(range(z,self.tiffs.meta['shape'][0],self.numZ))) for z in range(self.numZ)]
        for z in range(self.numZ):
            zTchunk[z] = self.tiffs.getchunk(req=zTidx[z]) #this takes the longest amount of time to run..
            # zT0[z] = self.tiffs.getchunk(req=dict(T=z)) #for verifying correct association of rois with image
        self.zTchunk = zTchunk
        return zTchunk, zT0

    def ZorganizedROIs(self):
        """
        Make a list of x,y coordinates of all rois indexed by z plane for the first volume in time
        """
        roixy,bckxy,roilabels=({} for i in range(3))
        roilabels=list(enumerate(self.roimasks.keys()))
        for z in range(self.numZ):
            roixy[z] = []
            bckxy[z] = []
            for k,v in self.roimasks.items():
                if type(k) in [list,tuple,dict] and z in k:
                    roixy[z].append(v)
            for k,v in self.bck.items():
                if type(k) in [list,tuple,dict] and z in k:
                    bckxy[z].append(v)
        # for k in roixy.keys():
        #     if not roixy[k]:
        #         del roixy[k]
        #         del bckxy[k]
        self.roixy = roixy
        self.bckxy = bckxy
        self.roilabels = roilabels
        rlabels=pd.DataFrame.from_dict(roilabels)
        rlabels.to_pickle(self.path+"rlabels.pkl")
        return roixy, bckxy, roilabels
    
    def TranslationCorrection(self):
        translation={}
        trroixy={}
        for z in range(0,self.numZ-1,3):
            for i in range(len(self.zTchunk[z].data)):
                translation[z,i] = ird.translation(self.zTchunk[z].data[0],self.zTchunk[z].data[i])
                if translation[z,i]['tvec'][1] <= -1:
                    trroixy[z,i] = [[(x+np.int(translation[z,i]['tvec'][1]),y) for (x,y) in self.roixy[z][n]] for n in range(len(self.roixy[z]))]
                    trroixy[z+1,i] = [[(x+np.int(translation[z,i]['tvec'][1]),y) for (x,y) in self.roixy[z+1][n]] for n in range(len(self.roixy[z+1]))]
                    trroixy[z+2,i] = [[(x+np.int(translation[z,i]['tvec'][1]),y) for (x,y) in self.roixy[z+2][n]] for n in range(len(self.roixy[z+2]))]
                elif translation[z,i]['tvec'][1] >= 1:
                    trroixy[z,i] = [[(x-np.int(translation[z,i]['tvec'][1]),y) for (x,y) in self.roixy[z][n]] for n in range(len(self.roixy[z]))]
                    trroixy[z+1,i] = [[(x-np.int(translation[z,i]['tvec'][1]),y) for (x,y) in self.roixy[z+1][n]] for n in range(len(self.roixy[z+1]))]
                    trroixy[z+2,i] = [[(x-np.int(translation[z,i]['tvec'][1]),y) for (x,y) in self.roixy[z+2][n]] for n in range(len(self.roixy[z+2]))]
                elif translation[z,i]['tvec'][0] <= -1:
                    trroixy[z,i] = [[(x,y-np.int(translation[z,i]['tvec'][0])) for (x,y) in self.roixy[z][n]] for n in range(len(self.roixy[z]))]
                    trroixy[z+1,i] = [[(x,y-np.int(translation[z,i]['tvec'][0])) for (x,y) in self.roixy[z+1][n]] for n in range(len(self.roixy[z+1]))]
                    trroixy[z+2,i] = [[(x,y-np.int(translation[z,i]['tvec'][0])) for (x,y) in self.roixy[z+2][n]] for n in range(len(self.roixy[z+2]))]
                elif translation[z,i]['tvec'][0] >= 1:
                    trroixy[z,i] = [[(x,y+np.int(translation[z,i]['tvec'][0])) for (x,y) in self.roixy[z][n]] for n in range(len(self.roixy[z]))]
                    trroixy[z+1,i] = [[(x,y+np.int(translation[z,i]['tvec'][0])) for (x,y) in self.roixy[z+1][n]] for n in range(len(self.roixy[z+1]))]
                    trroixy[z+2,i] = [[(x,y+np.int(translation[z,i]['tvec'][0])) for (x,y) in self.roixy[z+2][n]] for n in range(len(self.roixy[z+2]))]
                else:
                    trroixy[z,i] = self.roixy[z]
                    trroixy[z+1,i] = self.roixy[z+1]
                    trroixy[z+2,i] = self.roixy[z+2]
        
        # for z in range(self.numZ):
        #     for r in range(len(self.roixy[z])):        
        #         trroixy[z,i] = trroixy[z,i][len(self.zTchunk[len(range(self.numZ))-1].data)] #there must be at least 1 roi on the last z plane analyzed
        #         translation[z,i] = translation[z,i][len(self.zTchunk[len(range(self.numZ))-1].data)]
        self.trroixy=trroixy
        self.translation=translation
        # trroixy_df=pd.DataFrame.from_dict(trroixy)
        # translation_df=pd.DataFrame.from_dict(translation)
        # trroixy_df.to_pickle(self.path+"Quant_trcorr/"+"trroixy_df.pkl")
        # translation_df.to_pickle(self.path+"Quant_trcorr/"+"translation_df.pkl")
        return trroixy, translation
    
    def roipixels(self):
        """
        Obtain fluorescence intensity values from corresponding image for each (x,y) coordinate
        """
        roi_pixels,roi_pixels_trcorr,bck_pixels = ({} for i in range(3))
        for z in range(self.numZ):
            for t in range(len(self.zTchunk[z].data)):
                roi_pixels[z,t] = [[self.zTchunk[z].data[t][y,x] for (x,y) in self.roixy[z][r]] for r in range(len(self.roixy[z]))]
                #roi_pixels_trcorr[z,t] = [[self.zTchunk[z].data[t][y,x] for (x,y) in self.trroixy[z,t][r]] for r in range(len(self.roixy[z]))]
                bck_pixels[z,t] = [[self.zTchunk[z].data[t][y,x] for (x,y) in self.bckxy[z][r]] for r in range(len(self.bckxy[z]))]
        self.roi_pixels = roi_pixels
        #self.roi_pixels_trcorr = roi_pixels_trcorr
        self.bck_pixels = bck_pixels
        return roi_pixels, bck_pixels

    def dFF(self):
        """
        Calculate and save dFF for each roi with and without background subtraction
        """
        os.mkdir(self.path+"Quant")
        bckg,roi,roi_nob,roi_nob_mean,roi_mean_nobsub,roi_dFF,roi_dFF_nobsub,roin_dFF,roin_dFF_nobsub,roi_bck,rn_dFF,rn_dFF_nobsub = ({} for i in range(12))
        for z in range(self.numZ):
            for t in range(len(self.zTchunk[z].data)):
                bckg[z,t] = np.mean(self.bck_pixels[z,t])
                for r in range(len(self.roixy[z])):
                    roi[(z,r),t] = np.mean(self.roi_pixels[z,t][r])
                    roi_nob[(z,r),t] = [roi[(z,r),t] - bckg[z,t]]
                    roi_mean_nobsub[(z,r)] = np.mean(self.roi_pixels[z,t][r])
        for z in range(self.numZ):
            for r in range(len(self.roixy[z])):
                roi_nob_mean[(z,r)]=[]
                for t in range(len(self.zTchunk[z].data)):
                    roi_nob_mean[(z,r)].extend(roi_nob[(z,r),t])
        for z in range(self.numZ):
            for r in range(len(self.roixy[z])):
                roin_dFF[(z,r)]=[]
                roin_dFF_nobsub[(z,r)]=[]
                roi_bck[z]=[]
                for t in range(len(self.zTchunk[z].data)):            
                    roi_dFF[(z,r),t] = [np.divide(roi_nob[(z,r),t],np.mean(roi_nob_mean[(z,r)]),where=np.mean(roi_nob_mean[(z,r)])!=0)]
                    roin_dFF[(z,r)].extend(roi_dFF[(z,r),t])
                    roi_dFF_nobsub[(z,r),t] = [np.divide(np.mean(self.roi_pixels[z,t][r]),roi_mean_nobsub[(z,r)],where=roi_mean_nobsub[(z,r)]!=0)]
                    roin_dFF_nobsub[(z,r)].extend(roi_dFF_nobsub[(z,r),t])
                    roi_bck[z].extend([bckg[z,t]])
        for z in range(self.numZ):
            for r in range(len(self.roixy[z])):        
                rn_dFF[(z,r)] = roin_dFF[(z,r)][:len(roi_bck[z])-1] #there must be at least 1 roi on the last z plane analyzed
                rn_dFF_nobsub[(z,r)] = roin_dFF_nobsub[(z,r)][:len(roi_bck[z])-1]
        rn_dFF_df=pd.DataFrame.from_dict(rn_dFF)
        rn_dFF_nobsub_df=pd.DataFrame.from_dict(rn_dFF_nobsub)
        rn_dFF_df.to_pickle(self.path+"Quant/"+"rn_dFF.pkl")
        rn_dFF_nobsub_df.to_pickle(self.path+"Quant/"+"rn_dFF_nobsub.pkl")
        self.roin_dFF = roin_dFF
        self.roin_dFF_nobsub = roin_dFF_nobsub
        return roin_dFF, roin_dFF_nobsub, roi_bck, roi_nob, roi, roi_nob_mean, roi_dFF

    def roiplots(self):
        """
        Save dFF timetrace of each roi as pdf and pkl
        """
        os.mkdir(self.path+"Quant/roiplots")
        # for z in range(self.numZ):
        #     for r in range(len(self.roixy[z])):
        for z,r in self.roin_dFF.keys():
            f=plt.figure(1,figsize=(15,3),facecolor='white')
            ax1=plt.subplot(111)
            plt.plot(self.roin_dFF[z,r],linewidth=0.3)
            fr,labels=plt.xticks()
            plt.xticks(fr,(fr*self.exptime*self.numZ).astype(int))
            plt.xlabel("time (s)")
            plt.ylabel("dF/F")
            roin = "roi z"+str(z)+"r"+str(r)
            plt.title(roin)
            plt.tight_layout()
            f.savefig(self.path+"Quant/roiplots/"+roin+".pdf",dpi=1200,format='pdf')
            plt.clf()
            rn_df=pd.DataFrame.from_dict(self.roin_dFF[z,r])
            rn_df.to_pickle(self.path+"Quant/roiplots/"+roin+".pkl")
    # def bleachcorr(self):
    #     """
    #     fir exponential decay to timeseries
    #     """
    #     #find range of time in which exp decay should be fit
    #     deriv0,d0where,maxt,tau,yf=({} for i in range(5))
    #     for z in range(self.numZ):
    #         for r in range(len(self.roixy[z])-1):
    #             deriv0[(z,r)] = np.diff(self.roin_dFF[(z,r)])==0
    #             d0where[(z,r)] = np.ndarray.tolist(np.where(np.logical_and(np.diff(self.roin_dFF[(z,r)])>=-0.005, np.diff(self.roin_dFF[(z,r)])<=0.005))[0])
    #             for k,g in groupby(enumerate(d0where[(z,r)]), lambda ix: ix[0]-ix[1]):
    #                 if len(list(map(itemgetter(1), g))) > 60: #if tderiv is close to 0 for more than 100 consecutive tpoints
    #                     maxt[(z,r)] = list(map(itemgetter(1), g))[0] #take the first tpoint from this group as end of tseries
    #     #then compute tau the decay constant and make a bleachcorrected trace
    #             tau[(z,r)]=1/np.linspace(1/(2*(maxt[(z,r)]-0)),1/(.1*(maxt[(z,r)]-0)),200)
    #             yf[(z,r)]=min(roin_dFF[(z,r)])