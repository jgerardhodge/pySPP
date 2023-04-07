import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib.colors as mcolors

def stomata_KDDs(NNSeries, xbound, ybound, ori_len=20, ori_wid=10, rankno=5):

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
    KDDy=NNSeries['dist_xdiff']
    KDDx=NNSeries['dist_ydiff']

    kde = gaussian_kde(np.vstack([KDDx, KDDy]))

    xmin, xmax = -xbound, xbound
    ymin, ymax = -ybound, ybound
    X, Y = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
    positions = np.vstack([X.ravel(), Y.ravel()])

    # Evaluate the kernel density at each grid point
    Z = np.reshape(kde(positions).T, Y.shape)

    # Plot the kernel density estimate
    plt.figure(figsize=(8,8))

    ax2.set_xlabel('Distance (um)')
    ax2.set_ylabel('Distance (um)')
    ax2.set_title('NN Distances')
    im = ax2.imshow(Z/np.max(Z), aspect='auto', extent=[xmin, xmax, ymin, ymax], cmap=kde_cmap)
    ax2.fill([-ori_len, -ori_len, ori_len, ori_len], [-ori_wid, ori_wid, ori_wid, -ori_wid], linewidth=2, edgecolor=(0,0,0), facecolor=(1,1,1))

    fig.colorbar(im, ax=ax2)
    
    return Z
