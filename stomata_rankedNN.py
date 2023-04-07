import pandas as pd
import numpy as np

def stomata_rankedNN(Fatemap, rankedNNs, distance='M', rankno=5):

    InFOV=Fatemap[(Fatemap['X_center']>0) & (Fatemap['X_center']<800) & (Fatemap['Y_center']>0) & (Fatemap['Y_center']<800)]

    SCs=InFOV[InFOV['Fate']==1]

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

    return rankedNNs