import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def stomata_compare_KDDs(obsZ, modelZ, xbound, ybound, ori_len, ori_wid, ranks):

    cr=np.interp(np.linspace(0, 1, 256), [0.0, 0.25, 0.5, 0.75, 1.0], [0.45, 0.0, 0.0, 0.9, 1.0])
    cg=np.interp(np.linspace(0, 1, 256), [0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.0, 0.0, 0.75, 1.0])
    cb=np.interp(np.linspace(0, 1, 256), [0.0, 0.25, 0.5, 0.75, 1.0], [0.75, 0.75, 0.0, 0.1, 1.0])

    kde_colchan=np.vstack((cr, cg, cb)).T
    kde_cmap=mcolors.ListedColormap(kde_colchan)

    xmin, xmax = -xbound, xbound
    ymin, ymax = -ybound, ybound

    diffZ=(obsZ/np.max(obsZ))-(modelZ/np.max(modelZ))
    
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