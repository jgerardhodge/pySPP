import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde
from scipy.stats import multivariate_normal
from scipy import ndimage

def coin(p1):
    r=random.random()
    if r<(1-p1):
        return 0
    else:
        return 1

def stomata_compare_KDDs(obsZ, modelZ, xbound, ybound, ori_len, ori_wid, ranks, plotting=False):

    cr=np.interp(np.linspace(0, 1, 256), [0.0, 0.25, 0.5, 0.75, 1.0], [0.45, 0.0, 0.0, 0.9, 1.0])
    cg=np.interp(np.linspace(0, 1, 256), [0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.0, 0.0, 0.75, 1.0])
    cb=np.interp(np.linspace(0, 1, 256), [0.0, 0.25, 0.5, 0.75, 1.0], [0.75, 0.75, 0.0, 0.1, 1.0])

    kde_colchan=np.vstack((cr, cg, cb)).T
    kde_cmap=mcolors.ListedColormap(kde_colchan)

    xmin, xmax = -xbound, xbound
    ymin, ymax = -ybound, ybound

    diffZ=(obsZ/np.max(obsZ))-(modelZ/np.max(modelZ))

    if plotting==True:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))

        ax1.set_xlabel('Distance (um)')
        ax1.set_ylabel('Distance (um)')
        ax1.set_title('Observed Manhattan NN Distances')
        im = ax1.imshow(obsZ/np.max(obsZ), aspect='auto', extent=[xmin, xmax, ymin, ymax], cmap=kde_cmap)
        ax1.fill([-ori_len, -ori_len, ori_len, ori_len], [-ori_wid, ori_wid, ori_wid, -ori_wid], linewidth=2, edgecolor=(0,0,0), facecolor=(1,1,1))

        ax2.set_xlabel('Distance (um)')
        ax2.set_ylabel('Distance (um)')
        ax2.set_title('MOM Manhattan NN Distances')
        im = ax2.imshow(modelZ/np.max(modelZ), aspect='auto', extent=[xmin, xmax, ymin, ymax], cmap=kde_cmap)
        ax2.fill([-ori_len, -ori_len, ori_len, ori_len], [-ori_wid, ori_wid, ori_wid, -ori_wid], linewidth=2, edgecolor=(0,0,0), facecolor=(1,1,1))

        ax3.set_xlabel('Distance (um)')
        ax3.set_ylabel('Distance (um)')
        ax3.set_title('Difference in Observed and Expected NN Distances')
        im = ax3.imshow(diffZ, aspect='auto', extent=[xmin, xmax, ymin, ymax], cmap=kde_cmap)
        ax3.fill([-ori_len, -ori_len, ori_len, ori_len], [-ori_wid, ori_wid, ori_wid, -ori_wid], linewidth=2, edgecolor=(0,0,0), facecolor=(1,1,1))


    OEscore=np.round(np.sum(np.abs((obsZ/np.max(obsZ))-(modelZ/np.max(modelZ)))),3)

    return OEscore, diffZ

def stomata_fateplot(Fatemap):

    Fatemap_plotting=Fatemap[(Fatemap['X_center']>-50) & (Fatemap['X_center']<850) & (Fatemap['Y_center']>-50) & (Fatemap['Y_center']<850)]

    plt.figure(figsize=(8,8))

    plt.xlim([0,800])
    plt.ylim([0,800])

    for i in range(0, len(Fatemap_plotting)):
        x=[Fatemap_plotting.iloc[i,6], Fatemap_plotting.iloc[i,6], Fatemap_plotting.iloc[i,6]+Fatemap_plotting.iloc[i,4], Fatemap_plotting.iloc[i,6]+Fatemap_plotting.iloc[i,4]]
        y=[Fatemap_plotting.iloc[i,7], Fatemap_plotting.iloc[i,7]+Fatemap_plotting.iloc[i,5], Fatemap_plotting.iloc[i,7]+Fatemap_plotting.iloc[i,5], Fatemap_plotting.iloc[i,7]]

        if Fatemap_plotting.iloc[i,3]==0:
            cellcol='gray'
        elif Fatemap_plotting.iloc[i,3]==-1:
            cellcol='darkred'
        elif Fatemap_plotting.iloc[i,3]==1:
            cellcol='darkgreen'
        elif Fatemap_plotting.iloc[i,3]==0.5:
            cellcol='darkgray'
        elif Fatemap_plotting.iloc[i,3]==0.75:
            cellcol=(0.85, 0.64, 0.13)
        plt.fill(x, y, linewidth=0, color=cellcol)

    plt.plot(Fatemap_plotting[Fatemap_plotting['Fate']==1].iloc[:,8], Fatemap_plotting[Fatemap_plotting['Fate']==1].iloc[:,9], linewidth=0, marker='o', color='black')
    plt.xlabel('Distance (um)')
    plt.ylabel('Distance (um)')
    plt.title('Simulated OT Scan')
    plt.show()

def stomata_FSC(obsZ):

    ############################
    # Ferguson Spin Correction
    ############################

    #Split Z frame observations on the x-origin
    lhalf =pd.DataFrame(obsZ).iloc[:,0:100]
    rhalf =pd.DataFrame(obsZ).iloc[:,101:200]

    # Find the index of the maximum value in the subset of the array
    lindex = np.unravel_index(np.argmax(lhalf.values), lhalf.shape)

    # Find the index of the maximum value in the subset of the array
    rindex = np.unravel_index(np.argmax(rhalf.values), rhalf.shape)

    bestm=np.abs((rindex[0]-lindex[0])/((100+rindex[1])-lindex[1]))

    for ang in np.arange(0, 360, 1):
        RobsZ = ndimage.rotate(obsZ, ang)

        #Split Z frame observations on the x-origin
        lhalf =pd.DataFrame(RobsZ).iloc[:,0:100]
        rhalf =pd.DataFrame(RobsZ).iloc[:,101:200]

        # Find the index of the maximum value in the subset of the array
        lindex = np.unravel_index(np.argmax(lhalf.values), lhalf.shape)

        # Find the index of the maximum value in the subset of the array
        rindex = np.unravel_index(np.argmax(rhalf.values), rhalf.shape)

        m=np.abs((rindex[0]-lindex[0])/((100+rindex[1])-lindex[1]))

        if m<bestm:
            bestm=m
            best_ang=ang

    print('Angle Correction = '+str(best_ang)+' degrees; Left-Right Peak slope = '+str(bestm))

    RobsZ = ndimage.rotate(obsZ, best_ang)
    RobsZ.shape

    original_shape = obsZ.shape

    # Create an empty array with the original shape
    Z_resized = np.zeros(original_shape)

    # Calculate the size difference between Z_rotated and the original shape
    size_diff = np.array(RobsZ.shape) - np.array(original_shape)

    # Calculate the indices for the subset of Z_rotated that matches the original shape
    i_start = size_diff[0] // 2
    j_start = size_diff[1] // 2
    i_end = i_start + original_shape[0]
    j_end = j_start + original_shape[1]

    # Copy the subset of Z_rotated that matches the original shape to the resized array
    RobsZ = RobsZ[i_start:i_end, j_start:j_end]
    
    return RobsZ

def stomata_KDDs(NNSeries, xbound, ybound, ori_len=20, ori_wid=10, rankno=5, plotting=False):

    KDDy=NNSeries['dist_xdiff']
    KDDx=NNSeries['dist_ydiff']

    kde = gaussian_kde(np.vstack([KDDx, KDDy]))

    xmin, xmax = -xbound, xbound
    ymin, ymax = -ybound, ybound
    X, Y = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
    positions = np.vstack([X.ravel(), Y.ravel()])

    # Evaluate the kernel density at each grid point
    Z = np.reshape(kde(positions).T, Y.shape)

    if plotting==True:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        ax1.set_xlim([-xbound,xbound])
        ax1.set_ylim([-ybound,ybound])
        ax1.set_xlabel('Distance (um)')
        ax1.set_ylabel('Distance (um)')
        ax1.set_title('NN Distances')

        NNcols=[(1,0,0),(0.9,0.75,0.05), (0.2, 0.8,0.1), (0,0.2,0.8), (0,0.1,0.6)]
        for i in range(rankno, 0, -1):
            ax1.plot(NNSeries[NNSeries['NN_rank']==i]['dist_xdiff'], NNSeries[NNSeries['NN_rank']==i]['dist_ydiff'], '.', color=NNcols[i-1])

        ax1.fill([-ori_len, -ori_len, ori_len, ori_len], [-ori_wid, ori_wid, ori_wid, -ori_wid], linewidth=2, edgecolor=(0,0,0), facecolor=(1,1,1))

        cr=np.interp(np.linspace(0, 1, 256), [0.0, 0.25, 0.5, 0.75, 1.0], [0.45, 0.0, 0.0, 0.9, 1.0])
        cg=np.interp(np.linspace(0, 1, 256), [0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.0, 0.0, 0.75, 1.0])
        cb=np.interp(np.linspace(0, 1, 256), [0.0, 0.25, 0.5, 0.75, 1.0], [0.75, 0.75, 0.0, 0.1, 1.0])

        kde_colchan=np.vstack((cr, cg, cb)).T
        kde_cmap=mcolors.ListedColormap(kde_colchan)

        # Plot the kernel density estimate
        plt.figure(figsize=(8,8))

        ax2.set_xlabel('Distance (um)')
        ax2.set_ylabel('Distance (um)')
        ax2.set_title('NN Distances')
        im = ax2.imshow(Z/np.max(Z), aspect='auto', extent=[xmin, xmax, ymin, ymax], cmap=kde_cmap)
        ax2.fill([-ori_len, -ori_len, ori_len, ori_len], [-ori_wid, ori_wid, ori_wid, -ori_wid], linewidth=2, edgecolor=(0,0,0), facecolor=(1,1,1))

        fig.colorbar(im, ax=ax2)
    
    return Z

def stomata_rankedNN_biodock(biodock_SCs, rankedNNs, distance='M', rankno=5):
    
    Genoname=np.unique(cur_geno_rep_FOV['Genotype'])[0]
    Repname=np.unique(cur_geno_rep_FOV['Replicate'])[0] 
    FOVname=np.unique(cur_geno_rep_FOV['Replicate'])[0]
    
    if distance=='M':
    
        for i in range(0,len(biodock_SCs)):
            dx=np.round(np.abs(biodock_SCs.iloc[:,5]-biodock_SCs.iloc[i,5])); 
            dy=np.round(np.abs(biodock_SCs.iloc[:,6]-biodock_SCs.iloc[i,6])); 

            Dm=dx+dy
            Dmr=np.sort(Dm)

            for j in range(1,rankno+1):
                NNm=biodock_SCs.iloc[np.where(Dm==Dmr[j])[0][0],]
                data={'Genotype': Genoname, 'Replicate': Repname, 'FOV': FOVname, 'Current_SC': i, 'MNN_rank': j, 'Manhat_dist':Dmr[j], 'Origin_X': biodock_SCs.iloc[i,5], 'Origin_Y': biodock_SCs.iloc[i,6], 'MNN_x': NNm[5],  'MNN_y': NNm[6], 'Manhat_dist_xdiff': np.round(NNm[5]-biodock_SCs.iloc[i,5],2), 'Manhat_dist_ydiff': np.round(NNm[6]-biodock_SCs.iloc[i,6],2)}
                rankedNNs.loc[len(rankedNNs)] = data
    elif distance=='E':

        for i in range(0,len(SCs)):
            dx=np.round(np.square(biodock_SCs.iloc[:,5]-biodock_SCs.iloc[i,5])); 
            dy=np.round(np.square(biodock_SCs.iloc[:,6]-biodock_SCs.iloc[i,6])); 

            Dm=np.sqrt(dx+dy)
            Dmr=np.sort(Dm)

            for j in range(1,rankno+1):
                NNm=biodock_SCs.iloc[np.where(Dm==Dmr[j])[0][0],]
                data={'Genotype': Genoname, 'Replicate': Repname, 'FOV': FOVname, 'Current_SC': i, 'MNN_rank': j, 'Manhat_dist':Dmr[j], 'Origin_X': biodock_SCs.iloc[i,5], 'Origin_Y': biodock_SCs.iloc[i,6], 'MNN_x': NNm[5],  'MNN_y': NNm[6], 'Manhat_dist_xdiff': np.round(NNm[5]-biodock_SCs.iloc[i,5],2), 'Manhat_dist_ydiff': np.round(NNm[6]-biodock_SCs.iloc[i,6],2)}
                rankedNNs.loc[len(rankedNNs)] = data
    else:
        print('Distance method supplied not recognized. Present options are \'M\' for Manhattan (Default) and \'E\' for Euclidean.')

    return rankedNNs

def stomata_rankedNN(Fatemap, rankedNNs, Rep='NA', distance='M', rankno=5):

    InFOV=Fatemap[(Fatemap['X_center']>0) & (Fatemap['X_center']<800) & (Fatemap['Y_center']>0) & (Fatemap['Y_center']<800)]

    SCs=InFOV[InFOV['Fate']==1]

    if (SCs.shape[0]>=rankno*2):
        
        NNerrors=''

        if distance=='M':

            for i in range(0,len(SCs)):
                dx=np.round(np.abs(SCs.iloc[:,8]-SCs.iloc[i,8])); 
                dy=np.round(np.abs(SCs.iloc[:,9]-SCs.iloc[i,9])); 

                Dm=dx+dy
                Dmr=np.sort(Dm)

                for j in range(1,rankno+1):
                    NNm=SCs.iloc[np.where(Dm==Dmr[j])[0][0],]
                    data={'Rep': Rep, 'Current_SC': i, 'NN_rank': j, 'dist':Dmr[j], 'Origin_X': SCs.iloc[i,8], 'Origin_Y': SCs.iloc[i,9], 'NN_x': NNm[8],  'NN_y': NNm[9], 'dist_xdiff': np.round(NNm[8]-SCs.iloc[i,8],2), 'dist_ydiff': np.round(NNm[9]-SCs.iloc[i,9],2)}
                    rankedNNs.loc[len(rankedNNs)] = data
        elif distance=='E':

            for i in range(0,len(SCs)):
                dx=np.round(np.square(SCs.iloc[:,8]-SCs.iloc[i,8])); 
                dy=np.round(np.square(SCs.iloc[:,9]-SCs.iloc[i,9])); 

                Dm=np.sqrt(dx+dy)
                Dmr=np.sort(Dm)

                for j in range(1,rankno+1):
                    NNm=SCs.iloc[np.where(Dm==Dmr[j])[0][0],]
                    data={'Rep': Rep, 'Current_SC': i, 'MNN_rank': j, 'Manhat_dist':Dmr[j], 'Origin_X': SCs.iloc[i,8], 'Origin_Y': SCs.iloc[i,9], 'MNN_x': NNm[8],  'MNN_y': NNm[9], 'Manhat_dist_xdiff': np.round(NNm[8]-SCs.iloc[i,8],2), 'Manhat_dist_ydiff': np.round(NNm[9]-SCs.iloc[i,9],2)}
                    rankedNNs.loc[len(rankedNNs)] = data
        else:
            print('Distance method supplied not recognized. Present options are \'M\' for Manhattan (Default) and \'E\' for Euclidean.')
            

        return NNerrors, rankedNNs 
    else:

        NNerrors='NNs_scarce_fatal_run'
        
        return NNerrors, rankedNNs

def stomatagenesis(VeinMu, VeinVar, MesoWgt, MesoVar, SigMaxV, SigMaxM, SigVar, SFMu, SFVar, SFEntropy, VeinFileMu, VeinFileVar, MesoFileMu, MesoFileVar, CPLenMu, CPLenVar, CPLenCov, PLenMu, PLenVar, PLenCov, GCLenMu, GCLenVar, GCLenCov, AsymLenMu, AsymLenVar, AsymLenCov, Yskew=3):
    
    FOV_summary_stats=pd.DataFrame(columns=['Rep', 'Avg. Stomata Len', 'Avg. Stomata Wid', 'Avg. Stomata Area', 'Avg. Pavement Len', 'Avg. Pavement Wid', 'Avg. Pavement Area', 'SD', 'PD', 'SI', 'VD'])

    ManhatNNSeries=pd.DataFrame(columns=['Rep', 'Current_SC', 'MNN_rank', 'Manhat_dist', 'Origin_X', 'Origin_Y', 'MNN_x', 'MNN_y', 'Manhat_dist_xdiff', 'Manhat_dist_ydiff'])

    #Model Size (These should not be modified as they are designed to fill a field of view!)
    TotalVeins=31 #Odd number required with how mesophyll are generated
    Derivatives=50
    
    #Cell type shape parameters including mean lengths, widths, and covariances to define shape and size
    #CP=Costal Pavement, P=Pavement, GC=Guard cell (pair), Asym=Asymmetric Pavement from SMC, Subs=Subsidiary cell

    VeinFileWid=[VeinFileMu,VeinFileVar]
    MesoFileWid=[MesoFileMu,MesoFileVar]

    CPMu=[VeinFileWid[0],CPLenMu]
    CPCov=[[VeinFileWid[1],CPLenCov],[CPLenCov,CPLenVar]]

    PMu=[MesoFileWid[0],PLenMu]
    PCov=[[MesoFileWid[1],PLenCov],[PLenCov,PLenVar]]

    GCMu=[MesoFileWid[0],GCLenMu]
    GCCov=[[MesoFileWid[1],GCLenCov],[GCLenCov,GCLenVar]]

    AsymMu=[MesoFileWid[0],AsymLenMu]
    AsymCov=[[MesoFileWid[1],AsymLenCov],[AsymLenCov,AsymLenVar]]

    VeinFiles=np.round(np.random.normal(VeinMu, VeinVar, TotalVeins))

    #Sanity check mask to ensure veins are at least 1-cell wide (otherwise they wouldn't exist)
    mask = VeinFiles<1
    VeinFiles[mask] = 1

    #Use the Random normal Vein file number information to generate mesophyll between sequential veins
    #(given the allometric mesophyll weight provided above)
    LeafFiles=[]

    LeafFiles.append(['Vein',VeinFiles[0]]);

    for i in range(1,len(VeinFiles)):
        V1=VeinFiles[i-1]; V2=VeinFiles[i]
        Me=np.round(((V1*MesoWgt)+(V2*MesoWgt))/2)
        LeafFiles.append(['Meso',Me]); LeafFiles.append(['Vein', V2])

    #Convert to Pandas for ease of parsing this dataframe by rows and columns later
    LeafFiles=pd.DataFrame(LeafFiles)
    LeafFiles=LeafFiles.rename(columns={0:'Type', 1:'Fileno'})

    #Estimate the cumulative file numbers of the interchanging veins and mesophyll for later histogenic maps
    CumulFiles=np.cumsum(LeafFiles.iloc[:,1]) 
    LeafFiles['CumFileno']=CumulFiles

    #Now we begin to create our histogenic maps y-axis by generating a series of files which correspond
    #to what we randomly generated for alternating vein and mesophyll bands
    fileno=int(LeafFiles.iloc[-1,2])

    Vein_filemap = pd.DataFrame(index=range(fileno), columns=['Identity', 'TissueBand'])
    Vein_filemap

    for band in range(0,len(LeafFiles)):
        CurTis=LeafFiles.iloc[band,]
        Vein_filemap.iloc[range(int(CurTis[2]-CurTis[1]), int(CurTis[2])),0]=CurTis[0]
        Vein_filemap.iloc[range(int(CurTis[2]-CurTis[1]), int(CurTis[2])),1]=band

    #Plotting while not particularly necessary is a handy confirmation the code is porting over correctly

    x=np.arange(0,fileno,1)
    Bands=np.unique(Vein_filemap['TissueBand'])
    SMC_filesignal=pd.DataFrame(index=x, columns=Bands)

    Vcnt=0; Mcnt=0

    #plt.figure()
    for CurBand in Bands:
        #Mean and variance of diffusion signal is inferred from the files of each band of tissue
        CurMu=int(np.mean(np.where(Vein_filemap['TissueBand']==CurBand)))
        CurVar=int(np.sum(Vein_filemap['TissueBand']==CurBand))*SigVar

        BandIdentity=Vein_filemap[Vein_filemap['TissueBand']==CurBand].iloc[0,0]

        if BandIdentity=='Vein':
            SigMax=SigMaxV
        elif BandIdentity=='Meso':
            SigMax=SigMaxM

        #Signal is calculated laterally based on these parameters
        CurSig=SigMax*np.exp(-(np.square(x-CurMu)/(2*np.square(CurVar))))

        SigIdent=Vein_filemap[Vein_filemap['TissueBand']==CurBand].iloc[0,0]

        #Merge the Vein-Mesophyll signals onto the filesignal dataframe
        SMC_filesignal[CurBand]=CurSig

        #Change column names from Band number to the ordered vein or mesophyll band positions
        SigIdent=Vein_filemap[Vein_filemap['TissueBand']==CurBand].iloc[0,0]
        if SigIdent=='Vein':
            Vcnt=Vcnt+1; color='r'
            SigIdent=SigIdent+str(Vcnt)
        else:
            Mcnt=Mcnt+1; color='g'
            SigIdent=SigIdent+str(Mcnt)

        #Rename the file signal columns to their corresponding Vein or Mesophyll band numbers
        SMC_filesignal=SMC_filesignal.rename(columns={CurBand: SigIdent})

        #plt.plot(CurSig, x, color, label=SigIdent)
    #plt.show()

    #Now begin to integrate the signal between the veins (auxin?) and mesophyll (Stomagen?)  

    #Given format of 'SMC_filesignal' this should take every other column corresponding to the Mesophyll bands
    MesoColNames=SMC_filesignal.filter(like='Meso').columns.tolist()
    MesoCols=[SMC_filesignal.columns.get_loc(col) for col in MesoColNames]

    SMC_filtersignal=pd.DataFrame(index=x, columns=SMC_filesignal.columns[MesoCols])

    #plt.figure()
    for CurMesophyll in MesoCols:
        #Extract signals for the current mesophyll band and it's flanking veins
        V1sig=SMC_filesignal.iloc[:,CurMesophyll-1]
        Msig=SMC_filesignal.iloc[:,CurMesophyll]
        V2sig=SMC_filesignal.iloc[:,CurMesophyll+1]

        #Create a joint-signal based on the interaction between the mesophyll and it's flanking veins
        Jsig=(V1sig*Msig)+(V2sig*Msig)
        SMC_filtersignal[SMC_filesignal.columns[CurMesophyll]]=Jsig

        #plt.plot(Jsig, x, 'b', label='Joint')

    #plt.show()

    #Now with the file positions specified and an integrated vein-mesophyll circle corresponding 
    #to these lateral positions we can begin to create a probability field the derivatives will sample...

    MesoBands=Vein_filemap[Vein_filemap['Identity']=='Meso']

    BandNo=np.unique(MesoBands['TissueBand'])

    Probs=np.zeros(len(Vein_filemap))
    Probs

    for CurBand in BandNo:
        MB=MesoBands[MesoBands['TissueBand']==CurBand].index               

        if len(MB)>1:
            rowstart=int(np.where(SMC_filtersignal.index==MB[0])[0])
            rowstop =int(np.where(SMC_filtersignal.index==MB[1])[0])
            curcol=int(np.where(BandNo==CurBand)[0])

            #print(rowstart); print(rowstop); print(curcol)
            Probs[rowstart:(rowstop+1)]=SMC_filtersignal.iloc[rowstart:(rowstop+1),curcol]
        else:
            rowstart=int(np.where(SMC_filtersignal.index==MB[0])[0])
            curcol=int(np.where(BandNo==CurBand)[0])

            #print(rowstart); print(rowstop); print(curcol)
            Probs[rowstart]=SMC_filtersignal.iloc[rowstart,curcol]

    Vein_filemap['SF_Prob']=Probs
    Vein_filemap['SF_Memory']=np.zeros(len(Vein_filemap))
    #Vein_filemap

    Fatemap=pd.DataFrame()

    for Deriv in range(0, Derivatives):

        CurDerivs=[]

        for i in range(0,len(Vein_filemap)):
            #Begin by assessing if non-committed files will be allowed to become stomatal fates
            Pr=Vein_filemap.iloc[i,2]
            State=Vein_filemap.iloc[i,3]; NewState=0;
            if (State==0):
                SFfate=coin(Pr)
                if SFfate==1:
                    NewState=np.round(np.random.normal(SFMu, SFVar, 1))[0]
                    Vein_filemap.iloc[i,3]=NewState

            #Now with memories created, leverage these files to produce a wave of derivatives...
            if Vein_filemap.iloc[i,0]=='Vein':
                CurDerivs.append(-1)
            elif Vein_filemap.iloc[i,3]>0:
                Vein_filemap.iloc[i,3]=Vein_filemap.iloc[i,3]-1
                SMCFate=coin(1-SFEntropy)
                CurDerivs.append(SMCFate)
            else:
                CurDerivs.append(0)

        Fatemap[Deriv]=CurDerivs

    Filewidths=np.zeros(len(Vein_filemap))
    CumFilelengths=Filewidths.copy()

    VFs=np.where(Vein_filemap['Identity']=='Vein')
    MFs=np.where(Vein_filemap['Identity']=='Meso')


    VFwid=np.round(np.random.normal(VeinFileWid[0], VeinFileWid[1], len(VFs[0])),2)
    MFwid=np.round(np.random.normal(MesoFileWid[0], MesoFileWid[1], len(MFs[0])),2)

    Filewidths[VFs]=VFwid; Filewidths[MFs]=MFwid
    Filewidths
    CumFilewidths=np.cumsum(Filewidths)-Filewidths

    Fatemap_scaling=Vein_filemap.iloc[:,0:2]
    Fatemap_scaling['FileWid_um']=Filewidths
    Fatemap_scaling['CumulFileWid_um']=CumFilewidths
    Fatemap_scaling['CumulFileLen_um']=CumFilelengths

    Fileno=len(Fatemap_scaling)

    #Fatemap_scaling

    Fatemap2=pd.DataFrame(columns=['File', 'Deriv', 'Historegion', 'Fate', 'Len', 'Wid', 'CumulLen', 'CumulWid'])

    #plt.figure()

    CumulLen=np.zeros(len(Fatemap))

    for Deriv in range(0, Derivatives):

        #Deriv=0

        CurDerivs=Fatemap.iloc[:,Deriv]

        Celltypes=[-1,0,1]

        index=0

        DerivLens=np.zeros(len(CurDerivs))
        DerivWids=np.zeros(len(CurDerivs))

        for Curtype in Celltypes:

            Historegion=Fatemap_scaling[CurDerivs==Curtype].iloc[:,1]
            CellWid=Fatemap_scaling[CurDerivs==Curtype].iloc[:,2]
            CumulWid=Fatemap_scaling[CurDerivs==Curtype].iloc[:,3]

            if Curtype==-1:

                mean_given_x = CPMu[1] + CPCov[1][index] / CPCov[index][index] * (CellWid - CPMu[index])
                cov_given_x = CPCov[1][1] - CPCov[1][index] / CPCov[index][index] * CPCov[index][1]
                CellLen = np.round(np.random.normal(mean_given_x, np.sqrt(cov_given_x), len(CellWid)),2)

                Filepos=np.where(CurDerivs==Curtype)[0]

                FileDeriv=np.repeat(Deriv,len(CellWid))
                CellFate=np.repeat(Curtype, len(CellWid))

                data={'File': Filepos, 'Deriv':FileDeriv, 'Historegion':Historegion, 'Fate':CellFate, 'Len':CellLen, 'Wid':CellWid, 'CumulLen':CumulLen[Filepos], 'CumulWid':CumulWid}
                CumulLen[Filepos]=CumulLen[Filepos]+CellLen
                CPdata=pd.DataFrame(data)

                #plt.plot(CellLen, CellWid, 'o', color='darkred')

            if Curtype==0:

                mean_given_x = PMu[1] + PCov[1][index] / PCov[index][index] * (CellWid - PMu[index])
                cov_given_x = PCov[1][1] - PCov[1][index] / PCov[index][index] * PCov[index][1]
                CellLen = np.round(np.random.normal(mean_given_x, np.sqrt(cov_given_x), len(CellWid)),2)

                Filepos=np.where(CurDerivs==Curtype)[0]
                FileDeriv=np.repeat(Deriv,len(CellWid))
                CellFate=np.repeat(Curtype, len(CellWid))

                data={'File': Filepos, 'Deriv':FileDeriv, 'Historegion':Historegion, 'Fate':CellFate, 'Len':CellLen, 'Wid':CellWid, 'CumulLen':CumulLen[Filepos], 'CumulWid':CumulWid}
                CumulLen[Filepos]=CumulLen[Filepos]+CellLen

                Pdata=pd.DataFrame(data)

                #plt.plot(CellLen, CellWid, 'o', color='darkgrey')

            if Curtype==1:

                mean_given_x = AsymMu[1] + AsymCov[1][index] / AsymCov[index][index] * (CellWid - AsymMu[index])
                cov_given_x = AsymCov[1][1] - AsymCov[1][index] / AsymCov[index][index] * AsymCov[index][1]
                CellLen1 = np.round(np.random.normal(mean_given_x, np.sqrt(cov_given_x), len(CellWid)),2)

                mean_given_x = GCMu[1] + GCCov[1][index] / GCCov[index][index] * (CellWid - GCMu[index])
                cov_given_x = GCCov[1][1] - GCCov[1][index] / GCCov[index][index] * GCCov[index][1]
                CellLen2 = np.round(np.random.normal(mean_given_x, np.sqrt(cov_given_x), len(CellWid)),2)

                Filepos=np.where(CurDerivs==Curtype)[0]
                FileDeriv=np.repeat(Deriv,len(CellWid))
                CellFate=np.repeat(Curtype, len(CellWid))

                data={'File': Filepos, 'Deriv':FileDeriv, 'Historegion':Historegion, 'Fate':CellFate-0.5, 'Len':CellLen1, 'Wid':CellWid, 'CumulLen':CumulLen[Filepos], 'CumulWid':CumulWid}
                CumulLen[Filepos]=CumulLen[Filepos]+CellLen1
                Asymdata=pd.DataFrame(data)

                data={'File': Filepos, 'Deriv':FileDeriv+0.5, 'Historegion':Historegion, 'Fate':CellFate, 'Len':CellLen2, 'Wid':CellWid, 'CumulLen':CumulLen[Filepos], 'CumulWid':CumulWid}
                CumulLen[Filepos]=CumulLen[Filepos]+CellLen2
                SCdata=pd.DataFrame(data)

                #plt.plot(CellLen1, CellWid, 'o', color='green')
                #plt.plot(CellLen2, CellWid, 'o', color='darkgreen')

        Deriv_scaling=pd.concat([CPdata, Pdata, Asymdata, SCdata])
        Deriv_scaling=Deriv_scaling.sort_values(by=['File', 'Deriv'])

        Fatemap2=pd.concat([Fatemap2, Deriv_scaling])

    ycent=(Fatemap2['CumulWid']+(Fatemap2['Wid']/2))
    xcent=(Fatemap2['CumulLen']+(Fatemap2['Len']/2))

    Fatemap2['X_center']=xcent
    Fatemap2['Y_center']=ycent

    #plt.show()

    InFOV=Fatemap2[(Fatemap2['X_center']>0) & (Fatemap2['X_center']<800) & (Fatemap2['Y_center']>0) & (Fatemap2['Y_center']<800)]

    ###############################################################
    # Local Lateral Inhibition (i.e. the Subsidiary 'sanity check')
    ###############################################################

    Fatemap3=Fatemap2.copy()

    Mesoregion=np.unique(InFOV[InFOV['Fate']==1]['Historegion'])

    for m in Mesoregion:

        #m=Mesoregion[1]

        #plt.figure(figsize=(8,8))
        #plt.xlim([0,800])
        #plt.ylim([0,800])

        Curregion=InFOV[InFOV['Historegion']==m]

        Stomata_index=np.where(Curregion['Fate']==1)[0]

        Stomata_count=sum(Curregion['Fate']==1)
        z_avg=[]

        #First generate the signal field
        for i in Stomata_index:

            mean=[Curregion.iloc[i,8],Curregion.iloc[i,9]]
            cov=[[(Curregion.iloc[i,4]*Curregion.iloc[i,5]*(1/Yskew)), 0], [0, (Curregion.iloc[i,4]*Curregion.iloc[i,5])]]

            #mean=[32,8]
            #cov=[[32,0], [0, 8]]

            x, y = np.meshgrid(np.linspace(0, 800, 400), np.linspace(0, 800, 400))
            pos = np.dstack((x, y))

            rv = multivariate_normal(mean, cov)
            z = rv.pdf(pos)

            if len(z_avg)==0:
                z_avg=z.copy()
            else:
                z_avg=z_avg+z

        z_avg=z_avg/Stomata_count
        z_avg=z_avg/np.max(z_avg)

        GCpos=Curregion[Curregion['Fate']==1]
        GCxb=[GCpos.iloc[:,6], GCpos.iloc[:,6], GCpos.iloc[:,6]+GCpos.iloc[:,4], GCpos.iloc[:,6]+GCpos.iloc[:,4]]
        GCyb=[GCpos.iloc[:,7], GCpos.iloc[:,7]+GCpos.iloc[:,5], GCpos.iloc[:,7]+GCpos.iloc[:,5], GCpos.iloc[:,7]]
        #plt.fill(GCxb, GCyb, linewidth=3, facecolor='none', edgecolor='red')

        #plt.contourf(x, y, z_avg)
        #plt.colorbar()
        #plt.title('Epidermal patterning inhibition - Intercostal Band '+str(1+np.where(Mesoregion==m)[0][0]))
        #plt.xlabel('X')
        #plt.ylabel('Y')
        #plt.show()

        ############################################################################################################################
        #Define Guard Cell Neighbor Pairs here by driving signal laterally out and seeing if this falls in another GC's bounding box
        ############################################################################################################################

        xwins=[]
        ywins=[]
        for i in range(0,len(GCpos)):
            xwins.append((np.round(GCpos.iloc[i,6]), np.round(GCpos.iloc[i,6]+GCpos.iloc[i,4])))
            ywins.append((np.round(GCpos.iloc[i,7]), np.round(GCpos.iloc[i,7]+GCpos.iloc[i,5])))

        conflicts=[]

        for i in range(0,len(GCpos)):

            imatch=np.where((np.round(GCpos.iloc[i,8])>=np.array(xwins)[:,0]) & 
                             (np.round(GCpos.iloc[i,8])<=np.array(xwins)[:,1]) & 
                             (np.round(GCpos.iloc[i,9]+(5+(GCpos.iloc[i,5]/2)))>=np.array(ywins)[:,0]) & 
                             (np.round(GCpos.iloc[i,9]-(5+(GCpos.iloc[i,5]/2)))<=np.array(ywins)[:,1]))
            conflict=np.array(imatch)[np.array(imatch)!=i]

            if len(conflict)>0:
                if i<conflict[0]:
                    conflicts.append([i, conflict[0]])
                else:
                    conflicts.append([conflict[0], i])

        if (len(conflicts)>0):

            GCpairs=np.unique(conflicts, axis=0)
            z_baseline=np.percentile(z_avg, 90)

            #####################################################################
            #Then sample the signal field for each stomata pair to define a 'winner'
            #####################################################################

            for j in range(0,len(GCpairs)):
                GC1x=[int(Curregion.iloc[GCpairs[j][0],6]), int((Curregion.iloc[GCpairs[j][0],6]+Curregion.iloc[GCpairs[j][0],4]))]
                GC1y=[int(Curregion.iloc[GCpairs[j][0],7]), int((Curregion.iloc[GCpairs[j][0],7]+Curregion.iloc[GCpairs[j][0],5]))]

                GC2x=[int(Curregion.iloc[GCpairs[j][1],6]), int((Curregion.iloc[GCpairs[j][1],6]+Curregion.iloc[GCpairs[j][1],4]))]
                GC2y=[int(Curregion.iloc[GCpairs[j][1],7]), int((Curregion.iloc[GCpairs[j][1],7]+Curregion.iloc[GCpairs[j][1],5]))]

                xwin1=np.where((x>=GC1x[0]) & (x < GC1x[1])); ywin1=np.where((y>=GC1y[0]) & (y < GC1y[1]))
                xwin2=np.where((x>=GC2x[0]) & (x < GC2x[1])); ywin2=np.where((y>=GC2y[0]) & (y < GC2y[1]))

                z1=np.ravel(z_avg[ywin1[0][0]:ywin1[0][-1]+1, xwin1[0][0]:xwin1[0][-1]+1])
                z2=np.ravel(z_avg[ywin2[0][0]:ywin2[0][-1]+1, xwin2[0][0]:xwin2[0][-1]+1])

                SCS1=np.round(np.sum(z1[z1>z_baseline])/len(z1),3); SCS2=np.round(np.sum(z2[z2>z_baseline])/len(z2),3)

                if SCS1<SCS2:
                    fp, dp = GCpos.iloc[GCpairs[j][0],0:2]
                else:
                    fp, dp = GCpos.iloc[GCpairs[j][1],0:2]

                Fatemap3.loc[(Fatemap3['File'] == fp) & (Fatemap3['Deriv'] == dp) & (Fatemap3['Fate'] == 1), 'Fate']=0.75

    return Fatemap3

def stomata_summstats(Fatemap, Rep):

    InFOV=Fatemap[(Fatemap['X_center']>0) & (Fatemap['X_center']<800) & (Fatemap['Y_center']>0) & (Fatemap['Y_center']<800)]

    SD=np.sum(InFOV['Fate']==1)
    PD=np.sum(InFOV['Fate']!=1)

    Sl=np.round(np.mean(InFOV[InFOV['Fate']==1]['Len']),2)
    Sw=np.round(np.mean(InFOV[InFOV['Fate']==1]['Wid']),2)
    Pl=np.round(np.mean(InFOV[InFOV['Fate']!=1]['Len']),2)
    Pw=np.round(np.mean(InFOV[InFOV['Fate']!=1]['Wid']),2)


    CD=len(InFOV)

    SI=np.round(SD/(SD+PD),2)

    Transect=InFOV[InFOV['Deriv']==0]

    Transect_veins=Transect['Fate']==-1

    VD=len(np.unique(Transect[Transect_veins]['Historegion']))

    FOV_stats={'Rep': Rep+1, 'Avg. Stomata Len': Sl, 'Avg. Stomata Wid': Sw, 'Avg. Stomata Area': Sl*Sw, 'Avg. Pavement Len': Pl, 'Avg. Pavement Wid': Pw, 'Avg. Pavement Area': Pl*Pw, 'SD': SD, 'PD': PD, 'SI': SI, 'VD': VD}
    
    return FOV_stats
