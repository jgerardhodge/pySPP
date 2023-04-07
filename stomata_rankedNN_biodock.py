import pandas as pd
import numpy as np

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