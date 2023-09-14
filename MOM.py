import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
from scipy.stats import multivariate_normal
from scipy.ndimage import rotate
from scipy import ndimage

def coin(p1):
    r=random.random()
    if r<(1-p1):
        return 0
    else:
        return 1


def stomata_rankedNN(sample_data,  distance='M', rankno=5):

    rankedNNs=pd.DataFrame(columns=['Genotype', 'Fieldplot', 'Replicate', 'FOV', 'Current_SC', 'NN_rank', 'NN_dist', 'Origin_X', 'Origin_Y', 'NN_x', 'NN_y', 'NN_dist_xdiff', 'NN_dist_ydiff'])

    x_series=sample_data['x_center']
    y_series=sample_data['y_center']

    coord_index=x_series.index

    for i in coord_index:

        if distance=='M':
            dx=np.abs(x_series-x_series[i])
            dy=np.abs(y_series-y_series[i])
            D=np.array(dx+dy)
        elif distance=='E':
            dx=np.square(x_series-x_series[i])
            dy=np.square(y_series-y_series[i])
            D=np.array(np.sqrt(dx+dy))
        else:
            print('Distance method supplied not recognized. Present options are \'M\' for Manhattan (Default) and \'E\' for Euclidean.')
            return 

        Dr=D.copy()
        Dr.sort()

        NN_dist=Dr[1:(rankno+1)]

        rank_index=[]

        for cur_rdist in NN_dist:
            rank_index.append(np.where(D==cur_rdist)[0][0])

        cur_NN=sample_data.iloc[rank_index,:]

        cg=np.repeat(geno, rankno)
        cp=np.repeat(plt, rankno)
        cr=np.repeat(rep, rankno)
        cf=np.repeat(fov, rankno)
        ci=np.repeat(np.where(coord_index==i)[0][0]+1, rankno)
        NN_rank=range(1,rankno+1)
        ori_x=np.repeat(x_series[i], rankno)
        ori_y=np.repeat(y_series[i], rankno)
        NN_x=cur_NN['x_center']
        NN_y=cur_NN['y_center']
        NN_xdiff=NN_x-ori_x
        NN_ydiff=NN_y-ori_y

        cur_NN_out=pd.DataFrame({'Genotype': cg, 'Fieldplot': cp, 'Replicate': cr, 'FOV': cf, 'Current_SC': ci, 'NN_rank': NN_rank, 'NN_dist': NN_dist, 'Origin_X': ori_x, 'Origin_Y': ori_y, 'NN_x': NN_x,  'NN_y': NN_y, 'NN_dist_xdiff': NN_xdiff, 'NN_dist_ydiff': NN_ydiff})

        rankedNNs = pd.concat([rankedNNs, cur_NN_out], axis=0, ignore_index=True)

        return rankedNNs


def plot_rankedNN(sample_data, ranked_NNs):
    
    # Create a figure and axis
    fig, ax = plot.subplots(figsize=(8,7))

    edge_colors=['red', 'orange', 'green', 'blue', 'navy']

    #Initially plot the bounding boxes of each stomata (based on length/widths around centroid)
    for box in range(0, len(sample_data)):
        cent_x=sample_data.iloc[box,4]
        ori_len=(sample_data.iloc[box,6]/2)

        cent_y=sample_data.iloc[box,5]
        ori_wid=(sample_data.iloc[box,7]/2)

        ax.fill([cent_x-ori_len, cent_x-ori_len, cent_x+ori_len, cent_x+ori_len], [cent_y-ori_wid, cent_y+ori_wid, cent_y+ori_wid, cent_y-ori_wid], linewidth=2, edgecolor=(0,0,0), facecolor=(1,1,1))

    #Plot arrows indicating rank ordered NN relationships between stomata
    for rank in range(rankno,0,-1):

        NN_edges=rankedNNs.loc[rankedNNs['NN_rank']==rank]

        # Loop through the data and draw arrows
        for i in range(len(NN_edges)):
            ax.arrow(NN_edges.iloc[i,7], NN_edges.iloc[i,8], NN_edges.iloc[i,9]-NN_edges.iloc[i,7], NN_edges.iloc[i,10]-NN_edges.iloc[i,8], head_width=15, head_length=0, fc=edge_colors[rank-1], ec=edge_colors[rank-1])

    # Set axis limits
    ax.set_xlim(0, 512)
    ax.set_ylim(0, 512)

    # Set axis labels
    ax.set_xlabel('Longitudinal Distance')
    ax.set_ylabel('Medial Distance')

    # Show the plot
    plot.show()


def stomata_KDDs(NNSeries, xbound=100, ybound=100, ori_len=20, ori_wid=10, rankno=5, rankmethod='avgrank', plotting=False):

    if not rankmethod=='avgrank' and not rankmethod=='currank':
        raise ValueError('Error with the rank method called! For averaging the 1-N ranks call \"avgrank\" whereas for screening a particular rank relationship use \"currank\".')

    if rankmethod=='avgrank':
        KDDy=NNSeries.loc[NNSeries['NN_rank']<=rankno]['NN_dist_xdiff']
        KDDx=NNSeries.loc[NNSeries['NN_rank']<=rankno]['NN_dist_ydiff']

        kde = gaussian_kde(np.vstack([KDDx, KDDy]))

        xmin, xmax = -xbound, xbound
        ymin, ymax = -ybound, ybound

        xres=xbound*2
        yres=ybound*2

        X, Y = np.mgrid[xmin:xmax:xres*1j, ymin:ymax:yres*1j]
        positions = np.vstack([X.ravel(), Y.ravel()])

        # Evaluate the kernel density at each grid point
        Z = np.reshape(kde(positions).T, Y.shape)

    elif rankmethod=='currank':
        KDDy=NNSeries.loc[NNSeries['NN_rank']==rankno]['NN_dist_xdiff']
        KDDx=NNSeries.loc[NNSeries['NN_rank']==rankno]['NN_dist_ydiff']

        kde = gaussian_kde(np.vstack([KDDx, KDDy]))

        xmin, xmax = -xbound, xbound
        ymin, ymax = -ybound, ybound

        xres=xbound*2
        yres=ybound*2
        
        X, Y = np.mgrid[xmin:xmax:xres*1j, ymin:ymax:yres*1j]
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
        if rankmethod=='avgrank':
            for i in range(rankno, 0, -1):
                ax1.plot(NNSeries[NNSeries['NN_rank']==i]['NN_dist_xdiff'], NNSeries[NNSeries['NN_rank']==i]['NN_dist_ydiff'], '.', color=NNcols[i-1])
        else:
            for i in range(rankno, 0, -1):
                ax1.plot(NNSeries[NNSeries['NN_rank']==rankno]['NN_dist_xdiff'], NNSeries[NNSeries['NN_rank']==rankno]['NN_dist_ydiff'], '.', color=NNcols[rankno-1])

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


def stomata_KDD_hist(NNSeries, Z, xbound, ybound, ori_len=20, ori_wid=10, rankno=5, plotting=False, plotname='Plotname'):

    KDDy=NNSeries['NN_dist_xdiff']
    KDDx=NNSeries['NN_dist_ydiff']

    kde = gaussian_kde(np.vstack([KDDx, KDDy]))

    xmin, xmax = -xbound, xbound
    ymin, ymax = -ybound, ybound

    cr=np.interp(np.linspace(0, 1, 256), [0.0, 0.25, 0.5, 0.75, 1.0], [0.45, 0.0, 0.0, 0.9, 1.0])
    cg=np.interp(np.linspace(0, 1, 256), [0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.0, 0.0, 0.75, 1.0])
    cb=np.interp(np.linspace(0, 1, 256), [0.0, 0.25, 0.5, 0.75, 1.0], [0.75, 0.75, 0.0, 0.1, 1.0])

    kde_colchan=np.vstack((cr, cg, cb)).T
    kde_cmap=mcolors.ListedColormap(kde_colchan)

    hori_p=np.zeros(Z.shape[0])
    hori_x=np.arange(-xbound, xbound, (xbound*2)/Z.shape[0])

    vert_y=np.arange(-ybound, ybound, (ybound*2)/Z.shape[1])
    vert_p=np.zeros(Z.shape[1])

    for i in range(0,Z.shape[0]):
        hori_p=hori_p+Z[i,:]
        vert_p=vert_p+Z[:,i]

    plt.ioff()

    if plotting==True:
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, gridspec_kw={'width_ratios': [8, 4], 'height_ratios': [4, 8]}, figsize=(12, 12))
        ax2.axis('off')

        #gs = fig.add_gridspec(2, 2, height_ratios=[1, 2], width_ratios=[2, 1])

        #ax1 = fig.add_subplot(gs[0, 0])
        #plt.figure(figsize=(8,4))
        ax1.set_xlabel('Distance (um)')
        ax1.set_ylabel('Average Prob.')
        ax1.set_title('Horizontal NN Distances')
        ax1.plot(hori_x, hori_p)

        #ax4 = fig.add_subplot(gs[1, 1])
        #plt.figure(figsize=(4,8))
        ax4.set_xlabel('Average Prob.')
        ax4.set_ylabel('Distance (um)')
        ax4.set_title('Vertical NN Distances')
        ax4.plot(vert_p, vert_y)

        #ax3 = fig.add_subplot(gs[1, 0])
        #plt.figure(figsize=(8,8))
        ax3.set_xlabel('Distance (um)')
        ax3.set_ylabel('Distance (um)')
        ax3.set_title('NN Distances')
        im = ax3.imshow(Z/np.max(Z), aspect='auto', extent=[xmin, xmax, ymin, ymax], cmap=kde_cmap)
        ax3.fill([-ori_len, -ori_len, ori_len, ori_len], [-ori_wid, ori_wid, ori_wid, -ori_wid], linewidth=2, edgecolor=(0,0,0), facecolor=(1,1,1))

    elif plotname!='Plotname':

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, gridspec_kw={'width_ratios': [8, 4], 'height_ratios': [4, 8]}, figsize=(12, 12))
        ax2.axis('off')

        #gs = fig.add_gridspec(2, 2, height_ratios=[1, 2], width_ratios=[2, 1])

        #ax1 = fig.add_subplot(gs[0, 0])
        #plt.figure(figsize=(8,4))
        ax1.set_xlabel('Distance (um)')
        ax1.set_ylabel('Average Prob.')
        ax1.set_title('Horizontal NN Distances')
        ax1.plot(hori_x, hori_p)

        #ax4 = fig.add_subplot(gs[1, 1])
        #plt.figure(figsize=(4,8))
        ax4.set_xlabel('Average Prob.')
        ax4.set_ylabel('Distance (um)')
        ax4.set_title('Vertical NN Distances')
        ax4.plot(vert_p, vert_y)

        #ax3 = fig.add_subplot(gs[1, 0])
        #plt.figure(figsize=(8,8))
        ax3.set_xlabel('Distance (um)')
        ax3.set_ylabel('Distance (um)')
        ax3.set_title('NN Distances')
        im = ax3.imshow(Z/np.max(Z), aspect='auto', extent=[xmin, xmax, ymin, ymax], cmap=kde_cmap)
        ax3.fill([-ori_len, -ori_len, ori_len, ori_len], [-ori_wid, ori_wid, ori_wid, -ori_wid], linewidth=2, edgecolor=(0,0,0), facecolor=(1,1,1))

        plt.savefig(plotname+'.pdf', format='pdf', bbox_inches='tight')
        plt.close()

    return hori_p, vert_p


def stomata_KDD_deriv_anno(vp, Z, xbound, ybound, ori_len=20, ori_wid=10, plotting=False, plotname='Plotname'):

    cr=np.interp(np.linspace(0, 1, 256), [0.0, 0.25, 0.5, 0.75, 1.0], [0.45, 0.0, 0.0, 0.9, 1.0])
    cg=np.interp(np.linspace(0, 1, 256), [0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.0, 0.0, 0.75, 1.0])
    cb=np.interp(np.linspace(0, 1, 256), [0.0, 0.25, 0.5, 0.75, 1.0], [0.75, 0.75, 0.0, 0.1, 1.0])

    kde_colchan=np.vstack((cr, cg, cb)).T
    kde_cmap=mcolors.ListedColormap(kde_colchan)

    #############################################################
    # Calculate derivatives from the flattened KDD
    #############################################################

    for i in range(0,len(vp)):
        dy_dx = np.gradient(vp, np.arange(0,len(vp)))

        # Calculate the second derivative
        d2y_dx2 = np.gradient(dy_dx, np.arange(0,len(vp)))

        # Calculate the third derivative
        d3y_dx3 = np.gradient(d2y_dx2, np.arange(0,len(vp)))
        

    origin=np.where(vp==np.max(vp))[0][0]

    dd_Lscree=np.where(d2y_dx2==np.max(d2y_dx2[np.arange(0,origin)]))[0][0]
    dd_Rscree=np.where(d2y_dx2==np.max(d2y_dx2[np.arange(origin,len(vp))]))[0][0]

    Lsierras=vp[np.arange(0, dd_Lscree)]
    Rsierras=vp[np.arange(dd_Rscree, len(vp))]

    dd_Lsierras=d2y_dx2[np.arange(0, dd_Lscree)]
    dd_Rsierras=d2y_dx2[np.arange(dd_Rscree,len(vp))]

    Lpeak=np.where(vp==Lsierras[np.where(dd_Lsierras==np.min(dd_Lsierras))[0][0]])[0][0]
    Rpeak=np.where(vp==Rsierras[np.where(dd_Rsierras==np.min(dd_Rsierras))[0][0]])[0][0]

    Lvalley=vp[np.arange(Lpeak, dd_Lscree)]
    Rvalley=vp[np.arange(dd_Rscree, Rpeak)]

    #Run a simple heuristic along the length of the left or right scree slopes reaching out to the left or right
    #peaks, when the descent shifts between steps from positive (descending) to negative (climbing) this will indicate
    #a saddle in the distribution where the trench is located.

    Lvalley_descent=[]
    Rvalley_descent=[]

    for step in np.arange(len(Lvalley)-1,0,-1):
        Lvalley_descent.append(Lvalley[step]-Lvalley[step-1])

    for step in np.arange(1, len(Rvalley), 1):
        Rvalley_descent.append(Rvalley[step-1]-Rvalley[step])

    #Find the saddle position in the heuristic series
    Lvalley_pos=np.where(Lvalley_descent==np.min(Lvalley_descent))[0]
    Rvalley_pos=np.where(Rvalley_descent==np.min(Rvalley_descent))[0]

    #Identify the position in the overall distribution the saddle corresponds to
    Ltrench=np.arange(dd_Lscree, Lpeak, -1)[Lvalley_pos][0]
    Rtrench=np.arange(dd_Rscree, Rpeak, 1)[Rvalley_pos][0]

    #Take scree slope positions between off file peaks and their trenches
    Lpeak_scree=np.arange(Lpeak,Ltrench)
    Rpeak_scree=np.arange(Rtrench,Rpeak)

    #Take the probabilities between the off file peaks and their trenches along the scree positions
    Lpeak_scree_p=vp[Lpeak_scree]
    Rpeak_scree_p=vp[Rpeak_scree]

    #Normalize the probabilities so that the peak is now 1 and the trench is now zero
    Normalized_Lscree=(Lpeak_scree_p-np.min(Lpeak_scree_p))/(np.max(Lpeak_scree_p)-np.min(Lpeak_scree_p))
    Normalized_Rscree=(Rpeak_scree_p-np.min(Rpeak_scree_p))/(np.max(Rpeak_scree_p)-np.min(Rpeak_scree_p))

    #Subtract this normalized probability series by 0.5 to identify the halfway point between the peak and the trench
    L_halfindex=np.where(np.diff(np.sign(Normalized_Lscree-0.5)))[0][0]
    R_halfindex=np.where(np.diff(np.sign(Normalized_Rscree-0.5)))[0][0]

    #Identify where in the off file scree slopes these halfway probabilities are passed as a estimate how far trenches span
    Ltrench_span=np.abs(Lpeak_scree[L_halfindex]-Ltrench)
    Rtrench_span=np.abs(Rpeak_scree[R_halfindex]-Rtrench)

    #Calculate probabilities of trenches, peaks, and the origin
    LTprob=vp[Ltrench]
    LPprob=vp[Lpeak]

    RTprob=vp[Rtrench]
    RPprob=vp[Rpeak]

    oriprob=vp[origin]

    ori_norm=np.arange(-ybound, ybound)[origin]
    LP_norm=np.arange(-ybound, ybound)[Lpeak]
    RP_norm=np.arange(-ybound, ybound)[Rpeak]
    # LT_norm=np.mean([ori_norm, LP_norm]) #
    LT_norm=np.arange(-ybound, ybound)[Ltrench]
    # RT_norm=np.mean([ori_norm, RP_norm]) #
    RT_norm=np.arange(-ybound, ybound)[Rtrench]

    #Now leverage these rescaled values for distance metrics
    origin_trench_dist=np.mean([np.abs(origin-Ltrench), np.abs(origin-Rtrench)])
    origin_peak_dist=np.mean([np.abs(origin-Lpeak), np.abs(origin-Rpeak)])
    trenchprob_FC=np.round(np.mean([(LTprob-oriprob)/oriprob,(RTprob-oriprob)/oriprob]),3)
    Peaksprob_FC=np.round(np.mean([(LPprob-oriprob)/oriprob,(RPprob-oriprob)/oriprob]),3)   

    ################################################################
    #Generate a plot of the vertical KDDs and their 2nd Derivatives
    ################################################################

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    ################################################################
    #The initial Vertical KDD distribution
    ################################################################

    #Plot the vertical KDD distribution
    ax1.plot(vp,np.arange(-ybound,ybound))

    #Plot the origin
    ax1.axhline(ori_norm, color='blue', linestyle='-')

    #Plot the off file peaks 
    ax1.axhline(LT_norm, color='red', linestyle='--')
    ax1.axhline(LP_norm, color='red', linestyle='-')

    #Plot the probability trench straddling the origin and off file peaks
    ax1.axhline(RT_norm, color='red', linestyle='--')
    ax1.axhline(RP_norm, color='red', linestyle='-')

    ax1.set_xlabel('Probabilities')
    ax1.set_ylabel('Distance (um)')
    ax1.set_title('Vertically Collated KDD probabilities')

    ################################################################
    # Second Derivative of the Vertical KDD distribution
    ################################################################

    #Plot the 2nd derivatives
    ax2.plot(d2y_dx2,np.arange(-ybound,ybound))

    #Plot the origin
    ax2.axhline(ori_norm, color='blue', linestyle='-')

    #Plot the off file peaks 
    ax2.axhline(LT_norm, color='red', linestyle='--')
    ax2.axhline(LP_norm, color='red', linestyle='-')

    #Plot the probability trench straddling the origin and off file peaks
    ax2.axhline(RT_norm, color='red', linestyle='--')
    ax2.axhline(RP_norm, color='red', linestyle='-')

    ax2.set_xlabel('Derivatives')
    ax2.set_ylabel('Distance (um)')
    ax2.set_title('Second derivative of Vertically Collated KDD probabilities')

    ########################################################################
    # Superimpose the regions identified onto the KDD for visual assessment
    ########################################################################    

    ax3.set_xlabel('Distance (um)')
    ax3.set_ylabel('Distance (um)')
    ax3.set_title('NN Distances')

    im = plt.imshow(Z/np.max(Z), aspect='auto', extent=[-xbound, xbound, -ybound, ybound], cmap=kde_cmap)

    #Plot the origin
    ax3.axhline(ori_norm, color='blue', linestyle='-')

    #Plot the off file peaks 
    ax3.axhline(LT_norm, color='red', linestyle='--')
    ax3.axhline(LP_norm, color='red', linestyle='-')

    #Plot the probability trench straddling the origin and off file peaks
    ax3.axhline(RT_norm, color='red', linestyle='--')
    ax3.axhline(RP_norm, color='red', linestyle='-')

    ax3.fill([-ori_len, -ori_len, ori_len, ori_len], [-ori_wid, ori_wid, ori_wid, -ori_wid], linewidth=2, edgecolor=(0,0,0), facecolor=(1,1,1))
    fig.colorbar(im, ax=ax3)

    if plotting!=True:
        plt.close()

    elif plotname!='Plotname':
        plt.savefig(plotname+'.pdf', format='pdf', bbox_inches='tight')
        plt.close()

    else:
        plt.show()

    return origin_trench_dist, origin_peak_dist, trenchprob_FC, Peaksprob_FC


def stomata_compare_KDDs(obsZ, modelZ, xbound, ybound, ori_len, ori_wid, ranks, plotting=False, filename=None):

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

        if filename is not None:
            plt.savefig(filename)
            plt.close(fig)

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

def stomata_FSC(obsZ, ang=False):

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

    startindex=[lindex, rindex]

    bestm=np.abs((rindex[0]-lindex[0])/((100+rindex[1])-lindex[1]))

    if ang==False:
        for ang in np.arange(0, 360, 1):
            RobsZ = ndimage.rotate(obsZ, ang)

            #Split Z frame observations on the x-origin
            newlhalf =pd.DataFrame(RobsZ).iloc[:,0:100]
            newrhalf =pd.DataFrame(RobsZ).iloc[:,101:200]

            # Find the index of the maximum value in the subset of the array
            newlindex = np.unravel_index(np.argmax(newlhalf.values), newlhalf.shape)

            # Find the index of the maximum value in the subset of the array
            newrindex = np.unravel_index(np.argmax(newrhalf.values), newrhalf.shape)

            m=np.abs((newrindex[0]-newlindex[0])/((100+newrindex[1])-newlindex[1]))

            if m<bestm:
                bestm=m
                best_ang=ang
                newindex=[newlindex, newrindex]

    else:
        best_ang=ang
        RobsZ = ndimage.rotate(obsZ, ang)

        #Split Z frame observations on the x-origin
        newlhalf =pd.DataFrame(RobsZ).iloc[:,0:100]
        newrhalf =pd.DataFrame(RobsZ).iloc[:,101:200]

        # Find the index of the maximum value in the subset of the array
        newlindex = np.unravel_index(np.argmax(newlhalf.values), newlhalf.shape)

        # Find the index of the maximum value in the subset of the array
        newrindex = np.unravel_index(np.argmax(newrhalf.values), newrhalf.shape)

        bestm=np.abs((rindex[0]-lindex[0])/((100+rindex[1])-lindex[1]))
        newindex=[newlindex, newrindex]

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
    
    return RobsZ, startindex, newindex
    


def stomata_KDD_rotate_symmetry(Z, xbound=100, ybound=100, interval=5, thresh=0.5, fastrun=True, plotting=False, plotname='Plotname'):

    cr=np.interp(np.linspace(0, 1, 256), [0.0, 0.25, 0.5, 0.75, 1.0], [0.35, 0.0, 0.0, 0.9, 1.0])
    cg=np.interp(np.linspace(0, 1, 256), [0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.0, 0.0, 0.75, 1.0])
    cb=np.interp(np.linspace(0, 1, 256), [0.0, 0.25, 0.5, 0.75, 1.0], [0.70, 0.75, 0.0, 0.1, 0.8])

    kde_colchan=np.vstack((cr, cg, cb)).T
    kde_cmap=mcolors.ListedColormap(kde_colchan)

    symtab_colnames=('Rotation_angle', '5um', '10um', '15um', '20um', '25um', 
                 '30um', '35um', '40um', '45um', '50um', '55um', '60um', '65um', '70um', '75um', '80um', '85um', '90um', '95um')

    #Define the rotation angles in degrees which will be performed iteratively on the original KDD
    angles=np.arange(0,-180,-5)

    #Define the step interval to extract probabilities from following rotation
    steps=np.arange(interval,xbound,interval)

    #Define the extent of the x and y bounds
    extent = [-xbound, xbound, -ybound, ybound]

    #Define a variable to hold the symmetric probability differences across the steps for the given
    #rotation angle
    sym_prob_diffs=[]

    for deg in angles:

        #Rotate the 2D array, Z, which holds the KDD distribution by 'deg' degrees
        frotZ = rotate(Z, angle=deg, reshape=False)

        #If fastrun set to false perform the same rotation in the reverse direction by 'deg' degrees
        if fastrun==False:
            rrotZ = rotate(Z, angle=-deg, reshape=False)

        #Define a variable to hold the symmetric probability differences across the steps for the given
        #rotation angle
        spd=[]

        #Iterate across the 5-100um steps at a 5um window
        for step in steps:

            kernel_x=step
            kernel_y=0

            #Log transform the probabilities to ensure the values approaching zero near the edges of the KDD
            #don't saturate the signal for the symmetric probability differences (SPD)

            #Retrieve kernel probabilities
            fl_pr=np.abs(frotZ[kernel_y+100, kernel_x+100])
            fr_pr=np.abs(frotZ[100-kernel_y, 100-kernel_x])

            # Transform
            fl_pr = np.log(fl_pr)
            fr_pr = np.log(fr_pr)

            #Symmetric difference calc in reverse direction
            forward_diff=np.abs((fl_pr-fr_pr)/(fl_pr+fr_pr))

            #If fast run set to false symmetric probability difference becomes an average of the forward and reverse
            #rotations
            if fastrun==False:

                #Retrieve kernel probabilities
                rl_pr=np.abs(rrotZ[kernel_y+100, kernel_x+100])
                rr_pr=np.abs(rrotZ[100-kernel_y, 100-kernel_x])

                # Transform
                rl_pr = np.log(rl_pr)
                rr_pr = np.log(rr_pr)

                #Symmetric difference calc in reverse direction
                reverse_diff=np.abs((rl_pr-rr_pr)/(rl_pr+rr_pr))

                #SPD becomes an average of the forward and reverse absolute differences
                spd.append((forward_diff+reverse_diff)/2)

            else:
                #SPD becomes the absolute difference of the forward rotation direction
                spd.append(forward_diff)   

        sym_prob_diffs.append(spd)

    #With symmetric differences calculated across all combinatorial rotations and window steps collate results into dataframe
    SymDiff_array=pd.DataFrame(sym_prob_diffs)

    SymDiff_array.insert(0, 'Rotation_angle', angles)
    
    SymDiff_array.columns=symtab_colnames

    #If an output figure is desired to visually assess and cross reference run plotting functionality

    if plotting==True:
        rot45 = rotate(Z, angle=45, reshape=False)
        rot90 = rotate(Z, angle=90, reshape=False)

        fig, axes = plt.subplots(2, 2, figsize=(10,10))
        ax1, ax2, ax3, ax4 = axes.ravel()

        legend_index=[]
        for angle in range(0, len(angles)):
            outlier=np.max(np.abs(sym_prob_diffs[angle]))>thresh
            if outlier==True:
                ax1.plot(steps, sym_prob_diffs[angle])
                legend_index.append(angle)

        ax1.set_ylim(-0.5,1)
        ax1.set_xlabel('Distance from Origin')
        ax1.set_ylabel('Symmetric Difference in log(Pr)')
        ax1.set_title('Symmetric Differences with Rotation Angle')
        ax1.plot(steps, np.repeat(0, len(steps)), color='black', linestyle=':')
        ax1.legend(labels=angles[legend_index], fontsize=6, ncol=4, loc='lower right')

        ax2.imshow(Z, extent=extent, cmap=kde_cmap)
        ax2.scatter(steps, np.repeat(0, len(steps)), marker='x', color='green')
        ax2.scatter(-steps, np.repeat(0, len(steps)), marker='x', color='red')
        ax2.set_xlim(-xbound,xbound)
        ax2.set_ylim(-ybound,ybound)
        ax2.set_xlabel('Distance')
        ax2.set_ylabel('Distance')
        ax2.set_title('Rotation angle: 0 degrees')

        ax3.imshow(rot45, extent=extent, cmap=kde_cmap)
        ax3.scatter(steps, np.repeat(0, len(steps)), marker='x', color='green')
        ax3.scatter(-steps, np.repeat(0, len(steps)), marker='x', color='red')
        ax3.set_xlim(-xbound,xbound)
        ax3.set_ylim(-ybound,ybound)
        ax3.set_xlabel('Distance')
        ax3.set_ylabel('Distance')
        ax3.set_title('Rotation angle: 45 degrees')

        ax4.imshow(rot90, extent=extent, cmap=kde_cmap)
        ax4.scatter(steps, np.repeat(0, len(steps)), marker='x', color='green')
        ax4.scatter(-steps, np.repeat(0, len(steps)), marker='x', color='red')
        ax4.set_xlim(-xbound,xbound)
        ax4.set_ylim(-ybound,ybound)
        ax4.set_xlabel('Distance')
        ax4.set_ylabel('Distance')
        ax4.set_title('Rotation angle: 90 degrees')

        if plotname!='Plotname':
            plt.savefig(plotname)
            plt.close(fig)

    return SymDiff_array

def stomata_KDD_rotate_symmetry_stats(Z, xbound=100, ybound=100, angle=0, plotting=False, plotname='Plotname'):

    cr=np.interp(np.linspace(0, 1, 256), [0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.30, 0.1, 0.35, 0.95])
    cg=np.interp(np.linspace(0, 1, 256), [0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.0, 0.0, 0.3, 0.90])
    cb=np.interp(np.linspace(0, 1, 256), [0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.70, 0.7, 0.25, 0.10])

    kde_colchan=np.vstack((cr, cg, cb)).T
    kde_cmap=mcolors.ListedColormap(kde_colchan)

    #Rotate the 2D array, Z, which holds the KDD distribution by 'deg' degrees
    rotZ = rotate(Z, angle=angle, reshape=False)

    left_half = np.flip(rotZ[:, :100])
    right_half = rotZ[:, 100:]

    consensus = ((left_half+right_half)-np.abs(right_half-left_half))**2
    conpeak = np.max(consensus)
    #consensus = consensus/conpeak
    
    
    squared_diff = (right_half - left_half)**2
    dissonance = np.sqrt(squared_diff)
    dispeak=np.max(dissonance)
    #dissonance = dissonance/dispeak
    
    mean_diff = np.mean(squared_diff)
    mirror_RMSD = np.sqrt(mean_diff)
    RMSD=np.sum(mirror_RMSD)
    CV=(np.std(squared_diff)/mean_diff)

    if plotting==True:

        fig, axes = plt.subplots(1, 5, figsize=(22,4))
        ax1, ax2, ax3, ax4, ax5 = axes.ravel()
        
        norm = Normalize(vmin=0, vmax=0.3)
        
        im = plt.imshow(Z/np.max(Z), aspect='auto', extent=[-xbound, xbound, -ybound, ybound], cmap='hot')
        fig.colorbar(im, ax=ax1, norm=norm)
        ax1.imshow(rotZ, aspect='auto', cmap='hot')
        ax1.set_title('Normalized KDD')
        
        ax2.imshow(left_half, aspect='auto', cmap='hot')
        ax2.set_title('KDD Left Face (Reversed)')
        ax2.grid(True, color='white')
        
        ax3.imshow(right_half, aspect='auto', cmap='hot')
        ax3.set_title('KDD Right Face')
        ax3.grid(True, color='white')
        
        im4 = plt.imshow(consensus, aspect='auto', extent=[-xbound, xbound, -ybound, ybound], cmap='bone')
        fig.colorbar(im4, ax=ax4, norm=norm)
        ax4.imshow(consensus, aspect='auto', cmap='bone')
        ax4.set_title('Norm. Symmetry Signal (max='+str(np.round(conpeak,2))+')')
        ax4.grid(True, color='white')
        
        
        copper_cmap = plt.get_cmap('copper')
        copper_cmap = plt.get_cmap('copper')
        cmap_start=copper_cmap(0)
        cmap_stop=copper_cmap(dispeak/0.4)
        
        cr=np.interp(np.linspace(0, 1, 256), [0.0, 1.0], [cmap_start[0], cmap_stop[0]])
        cg=np.interp(np.linspace(0, 1, 256), [0.0, 1.0], [cmap_start[1], cmap_stop[1]])
        cb=np.interp(np.linspace(0, 1, 256), [0.0, 1.0], [cmap_start[2], cmap_stop[2]])

        asym_colchan=np.vstack((cr, cg, cb)).T
        asym_cmap=mcolors.ListedColormap(asym_colchan)

        im5 = plt.imshow(dissonance, aspect='auto', extent=[-xbound, xbound, -ybound, ybound], cmap='copper')
        im5.norm.autoscale([0, 1])
        colorbar = fig.colorbar(im5, ax=ax5) #,  extend='neither', ticks=np.linspace(0, dispeak, 6), boundaries=np.linspace(0, dispeak, 128))
        tick_positions = np.linspace(0, 1, 6)
        tick_labels = np.round(np.linspace(0, 0.4, 6),2)  # Format the labels as desired

        # Set tick positions and labels on the color bar
        colorbar.set_ticks(tick_positions)
        colorbar.set_ticklabels(tick_labels)
        
        ax5.imshow(dissonance, aspect='auto', cmap=asym_cmap)
        ax5.set_title('Norm. Asymmetry Signal (max='+str(np.round(dispeak,3))+')') #Formerly known as the Root Squared Difference
        ax5.grid(True, color='white')

        if plotname!='Plotname':
            plt.savefig(plotname)
            plt.close(fig)

    return RMSD, CV


def stomata_KDD_rorshach(Z, plotting=False):
    ###############################
    # Prospective Rorshach Function
    ###############################

    avgZ=Z.copy()

    X=avgZ.shape[0]-1
    Y=avgZ.shape[1]-1

    for cx in np.arange(0,X):
        for cy in np.arange(0,Y):

            Q1=avgZ[cy, cx]
            Q2=avgZ[cy, X-cx]
            Q3=avgZ[Y-cy, cx]
            Q4=avgZ[Y-cy, X-cx]
            avgZ[cy, cx]=np.mean([Q1, Q2, Q3, Q4])

    
    if plotting==True:
        plt.figure(figsize=(8,8))
        plt.imshow(avgZ/np.max(avgZ))
        
    return avgZ


def stomata_KDD_peak_spacing(Zif, Zsp, xbound, ybound, masking_val=0.5, trench_thresh=25, buffer=5, plotting=False, plotname='Plotname'):

    # Create a series of empty vectors to house output values
    prob_len=[]
    prob_wid=[]
    prob_area=[]
    cent_x=[]
    cent_y=[]

    ####################################################
    #First we will mask and segment the in-file moduli #
    ####################################################
    
    Z_mask=Zif.copy()

    #Use 2nd derivative annotated trench to either mask out side or in-file moduli to better expedite peak masking
    trench_thresh=int(trench_thresh)
    rows=np.arange(0,Z_mask.shape[0])
    midrow=Z_mask.shape[0]/2

    if_mask=(rows < (midrow - trench_thresh)) | (rows > (midrow + trench_thresh))
    Z_mask[if_mask]=0
    Z_mask[Z_mask>masking_val]=1
    Z_mask[Z_mask<masking_val]=0


    #Iterate through the binary mask and label each conterminous mask with an identifier, then generate a bounding box to measure it
    labeled_array_if, num_labels = ndimage.label(Z_mask == 1)
    bounding_boxes = ndimage.find_objects(labeled_array_if)

    mask_sizes = np.bincount(labeled_array_if.ravel())[1:]  # Calculate sizes
    sorted_labels = np.argsort(mask_sizes)[::-1]
    primary_labels = sorted_labels[:2]

    for label in primary_labels:  # Exclude background label 0

        patch_slice = bounding_boxes[label - 1]  # Get the slice of the patch
        patch = Z_mask[patch_slice]  # Extract the patch from the array
        patch_shape = patch.shape  # Get the shape of the patch

        # Calculate centroid position
        indices = np.indices(patch.shape)
        cent_y.append(np.round((indices[0] * patch).sum() / patch.sum() + patch_slice[0].start)-ybound)
        cent_x.append(np.round((indices[1] * patch).sum() / patch.sum() + patch_slice[1].start)-xbound)

        # Calculate the file length (i.e. x-axis)
        prob_len.append(patch_slice[1].stop - patch_slice[1].start)

        # Calculate the file width/height (i.e. y-axis)
        prob_wid.append(patch_slice[0].stop - patch_slice[0].start)

    if_thresh=np.mean([prob_wid[0],prob_wid[1]])/2
    
    ########################################################
    #Next we will mask and segment side peak moduli moduli #
    ########################################################
    
    Z_mask2=Zsp.copy()

    #Use 2nd derivative annotated trench to either mask out side or in-file moduli to better expedite peak masking
    trench_thresh=int(if_thresh)
    rows=np.arange(0,Z_mask2.shape[0])
    midrow=Z_mask2.shape[0]/2

    sp_mask=(rows > (midrow - (if_thresh+buffer))) & (rows < (midrow + (if_thresh+buffer)))
    Z_mask2[sp_mask]=0
    Z_mask2[Z_mask2>masking_val]=1
    Z_mask2[Z_mask2<masking_val]=0


    #Iterate through the binary mask and label each conterminous mask with an identifier, then generate a bounding box to measure it
    labeled_array_sp, num_labels = ndimage.label(Z_mask2 == 1)
    bounding_boxes = ndimage.find_objects(labeled_array_sp)

    mask_sizes = np.bincount(labeled_array_sp.ravel())[1:]  # Calculate sizes
    sorted_labels = np.argsort(mask_sizes)[::-1]
    secondary_labels = sorted_labels[:2]

    for label in secondary_labels:  # Exclude background label 0
        patch_slice = bounding_boxes[label]  # Get the slice of the patch
        patch = Z_mask2[patch_slice]  # Extract the patch from the array
        patch_shape = patch.shape  # Get the shape of the patch

        # Calculate centroid position
        indices = np.indices(patch.shape)
        cent_y.append(np.round((indices[0] * patch).sum() / patch.sum() + patch_slice[0].start)-ybound)
        cent_x.append(np.round((indices[1] * patch).sum() / patch.sum() + patch_slice[1].start)-xbound)

        # Calculate the file length (i.e. x-axis)
        prob_len.append(patch_slice[1].stop - patch_slice[1].start)

        # Calculate the file width/height  (i.e. y-axis)
        prob_wid.append(patch_slice[0].stop - patch_slice[0].start)

    #####################################################
    # With masking complete, plot and calculate metrics #
    #####################################################
    
    if plotting==True:

        extent=[-xbound,xbound,-ybound,ybound]

        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(6,6))
        ax1.imshow(Zif, extent=extent, cmap='inferno')
        ax1.set_title('In-file Peak KDDs')
        ax2.imshow(Zsp, extent=extent, cmap='inferno')
        ax2.set_title('In-file Peak Masks')
        ax3.imshow(labeled_array_if, extent=extent, cmap='inferno')
        ax3.scatter(cent_x[0:2], cent_y[0:2], color='green')
        ax3.set_title('Side Peak KDDs')
        ax4.imshow(labeled_array_sp, extent=extent, cmap='inferno')
        ax4.set_title('Side Peak Masks')
        ax4.scatter(cent_x[2:4], cent_y[2:4], color='green')
        
        if plotname!='Plotname':
            plt.savefig(plotname)
            plt.close(fig)
        else:
            plt.show()

    infile_len=(prob_len[0]+prob_len[1])/2
    infile_wid=(prob_wid[0]+prob_wid[1])/2
    infile_area=infile_len*infile_wid
    infile_dist=np.round(np.mean(np.sqrt((cent_x[0]**2)+(cent_y[0]**2))+np.sqrt((cent_x[1]**2)+(cent_y[1]**2))),2)

    sidepeak_len=(prob_len[2]+prob_len[3])/2
    sidepeak_wid=(prob_wid[2]+prob_wid[3])/2
    sidepeak_area=sidepeak_len*sidepeak_wid
    sidepeak_dist=np.round(np.mean(np.sqrt((cent_x[2]**2)+(cent_y[2]**2))+np.sqrt((cent_x[3]**2)+(cent_y[3]**2))),2)


    return infile_len, infile_wid, infile_area, infile_dist, sidepeak_len, sidepeak_wid, sidepeak_area, sidepeak_dist 
    


def stomata_KDD_rorshach_old(Z, plotting=False):
    ###############################
    # Prospective Rorshach Function
    ###############################

    avgZ=Z.copy()

    Xmid=avgZ.shape[0]/2
    Ymid=avgZ.shape[1]/2

    Xq1=np.arange(Xmid+1, (Xmid*2)+1, 1)-1
    Yq1=np.arange(Ymid+1, (Ymid*2)+1, 1)-1

    Xq2=np.arange(Xmid, Xmid-Xmid, -1)-1
    Yq2=np.arange(Ymid+1, (Ymid*2)+1, 1)-1

    Xq3=np.arange(Xmid, Xmid-Xmid, -1)-1
    Yq3=np.arange(Ymid, Ymid-Ymid, -1)-1

    Xq4=np.arange(Xmid+1, (Xmid*2)+1, 1)-1
    Yq4=np.arange(Ymid, Ymid-Ymid, -1)-1

    Quadrants=pd.DataFrame({'Q1x': Xq1, 'Q1y': Yq1, 'Q2x': Xq2, 'Q2y': Yq2, 'Q3x': Xq3, 'Q3y': Yq3, 'Q4x': Xq4, 'Q4y': Yq4})
    Quadrants

    for i in range(0, len(Quadrants)):

        Q1_prob=avgZ[int(Xq1[i]),int(Yq1[i])]
        Q2_prob=avgZ[int(Xq2[i]),int(Yq2[i])]
        Q3_prob=avgZ[int(Xq3[i]),int(Yq3[i])]
        Q4_prob=avgZ[int(Xq4[i]),int(Yq4[i])]
        
        quadavg=(Q1_prob+Q2_prob+Q3_prob+Q4_prob)/4
        quadavg

        avgZ[int(Xq1[i]),int(Yq1[i])]=quadavg
        avgZ[int(Xq2[i]),int(Yq2[i])]=quadavg
        avgZ[int(Xq3[i]),int(Yq3[i])]=quadavg
        avgZ[int(Xq4[i]),int(Yq4[i])]=quadavg

    if plotting==True:
        plt.figure(figsize=(8,8))
        plt.imshow(avgZ/np.max(avgZ))


    return avgZ



####
# Experimental beyond this point!
####



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

    VeinFiles=np.round(np.random.normal(VeinMu, np.sqrt(VeinVar), TotalVeins))

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
                    NewState=np.round(np.random.normal(SFMu, np.sqrt(SFVar), 1))[0]
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

    #Critical sanity check to prevent generation of negative widths when initializing scaling of vein and mesophyll cells
    VFwid_coarse=np.round(np.random.normal(VeinFileWid[0], np.sqrt(VeinFileWid[1]), 1000))
    VFwid=np.random.choice(VFwid_coarse, len(VFs[0]))

    MFwid_coarse=np.round(np.random.normal(MesoFileWid[0], np.sqrt(MesoFileWid[1]), 1000))
    MFwid=np.random.choice(MFwid_coarse, len(MFs[0]))
    
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
                GC1x=[int(GCpos.iloc[GCpairs[j][0],6]/2), int((GCpos.iloc[GCpairs[j][0],6]+GCpos.iloc[GCpairs[j][0],4])/2)]
                GC1y=[int(GCpos.iloc[GCpairs[j][0],7]/2), int((GCpos.iloc[GCpairs[j][0],7]+GCpos.iloc[GCpairs[j][0],5])/2)]

                GC2x=[int(GCpos.iloc[GCpairs[j][1],6]/2), int((GCpos.iloc[GCpairs[j][1],6]+GCpos.iloc[GCpairs[j][1],4])/2)]
                GC2y=[int(GCpos.iloc[GCpairs[j][1],7]/2), int((GCpos.iloc[GCpairs[j][1],7]+GCpos.iloc[GCpairs[j][1],5])/2)]

                xwin1=np.where((x>=GC1x[0]) & (x < GC1x[1])); ywin1=np.where((y>=GC1y[0]) & (y < GC1y[1]))
                xwin2=np.where((x>=GC2x[0]) & (x < GC2x[1])); ywin2=np.where((y>=GC2y[0]) & (y < GC2y[1]))

                try:
                    z1=np.ravel(z_avg[ywin1[0][0]:ywin1[0][-1]+1, xwin1[0][0]:xwin1[0][-1]+1])
                    z2=np.ravel(z_avg[ywin2[0][0]:ywin2[0][-1]+1, xwin2[0][0]:xwin2[0][-1]+1])

                    SCS1=np.round(np.sum(z1[z1>z_baseline])/len(z1),3); SCS2=np.round(np.sum(z2[z2>z_baseline])/len(z2),3)

                    if SCS1<SCS2:
                        fp, dp = GCpos.iloc[GCpairs[j][0],0:2]
                    else:
                        fp, dp = GCpos.iloc[GCpairs[j][1],0:2]

                    Fatemap3.loc[(Fatemap3['File'] == fp) & (Fatemap3['Deriv'] == dp) & (Fatemap3['Fate'] == 1), 'Fate']=0.75
                except:
                    print('Subsidiary sanity check error')
                    print('xwins: ', xwin1, xwin2)
                    print('ywins: ', ywin1, ywin2)

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
