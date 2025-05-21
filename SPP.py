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


def stomata_rankedNN(sample_data,  distance='M', rankno=5):

    """
    
    ----------
    Parameters
    ----------
    
    sample_data : (REQUIRED INPUT) A pandas dataframe which follows the standard formatting of this library for initial inputs with the 
                   the following columns expected:
                       'Genotype' -           Genotype for the stomatal coordinates 
                       'Fieldplot' -          Fieldplot or Environmental group of the current Genotype
                       'Replicate' -          Biological Replicate of the current Genotype-Fieldplot
                       'FOV' -                Technical Replicate (i.e., image) of the current Genotype-Fieldplot-Replicate
                       'x_center' -           The centroid x-coordinate location for the stomata
                       'y_center' -           The centroid y-coordinate location for the stomata
                       'length' -             The mask/bounding box length of the stomata
                       'width' -              The mask/bounding box width of the stomata
                       'stoma_area' -         The mask/bounding box area of the stomata
                       'length_width_ratio' - The mask/bounding box length/width ratio of the stomata
    distance :    (Defaults to 'M') Takes a single character being either 'M' or 'E' that defines the distance method used to 
                   assess rank-ordered neighbor relationships between stomatal coordinates, 'M' or Manhattan distance typically 
                   works best on grasses due to their file system, 'E' or Euclidean is also available as an option however.
    rankno :      (Defaults to 5) An integer representing the number of rank-order nearest neighbors to identify for each origin 
                   coordinate.
    
    ----------
    Returns
    ----------
    rankedNNs :   (pandas.core.frame.DataFrame) The rank-order nearest neighbors for each individual stomata contained with the image(s). 
    
    
    ----------
    Notes
    ----------
    A primary function of the SPP library used to assess all-by-all rank-ordered nearest neighbor relationships necessary for 
    estimating the Stomatal Patterning Phenotype (SPP).
    
    """
    
    geno=np.unique(sample_data['Genotype'])[0]
    fld=np.unique(sample_data['Fieldplot'])[0]
    rep=np.unique(sample_data['Replicate'])[0]
    fov=np.unique(sample_data['FOV'])[0]

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
        cp=np.repeat(fld, rankno)
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

        NN_out=pd.DataFrame({'Genotype': cg, 'Fieldplot': cp, 'Replicate': cr, 'FOV': cf, 'Current_SC': ci, 'NN_rank': NN_rank, 'NN_dist': NN_dist, 'Origin_X': ori_x, 'Origin_Y': ori_y, 'NN_x': NN_x,  'NN_y': NN_y, 'NN_dist_xdiff': NN_xdiff, 'NN_dist_ydiff': NN_ydiff})

        rankedNNs = pd.concat([rankedNNs, NN_out], axis=0, ignore_index=True)

    return rankedNNs


def stomata_nullNN(distance='M', rankno=5, ori_len=20, ori_wid=10, xlim=100, ylim=100, n=40, plotno=1, repno=1, techno=1):

    """
    
    ----------
    Parameters
    ----------

    
    distance :   (Defaults to 'M') Takes a single character being either 'M' or 'E' that defines the distance method used to 
                  assess rank-ordered neighbor relationships between stomatal coordinates, 'M' or Manhattan distance typically 
                  works best on grasses due to their file system, 'E' or Euclidean is also available as an option however.
    rankno :     (Defaults to 5) An integer representing the rank number to used for generating KDDs, either the average among ranks up 
                  to this number or as this current specific number by itself. The context of how rankno is used is governed by the 
                  subsequent argument, rankmethod.
    ori_len :    (Defaults to 20) An integer representing the average length of the origin objects (i.e., stomatal complex lengths) 
                  being assessed, used for plotting.
    ori_wid :    (Defaults to 10) An integer representing the average width of the origin objects (i.e., stomatal complex widths) 
                  being assessed, used for plotting.
    xbound :     (Defaults to 100) An integer representing the bounds of the x-axis relative to the origin (typically best to slightly 
                  overshoot the broadest rank-order distance.
    ybound :     (Defaults to 100) An integer representing the bounds of the y-axis relative to the origin (typically best to slightly 
                  overshoot the broadest rank-order distance.
    n :          (Defaults to 40) An integer representing the average density of coordinates to randomly generate per simulated 'image',
                  equivalent to average stomatal density. 
    plotno :     (Defaults to 1) An integer representing the average number of fieldplot replicates to simulate.
    repno :      (Defaults to 1) An integer representing the average number of biological replicates to simulate.
    techno :     (Defaults to 1) An integer representing the average number of technical replicates to simulate.


    ----------
    Returns
    ----------
    nullNNs :     (pandas.core.frame.DataFrame) An artifical set of rank-order nearest neighbors for for 'n' * 'plotno' * 'repno' * 'techno' 
                   numbers of starting coordinates. 
    
    
    ----------
    Notes
    ----------
    
    A supplementary function of the SPP library used to generate an artifical dataset assuming that either the Manhattan or Euclidean distance method
    is the only source of variation being seen within this dataset to act as a null model for comparison.

    
    """
    
    geno='Simulation'
    
    #Simulate the same number of images used to generate the current genotypes ranked NNs given a set image resolution (xlim, ylim), coordinate density (n), field plot replicates (plotno), 
    #biological replicates (repno), and finally technical 'FOV' replicates (techno)
    
    flds=np.arange(0,plotno)
    reps=np.arange(0,repno)
    fovs=np.arange(0,techno)

    #Empty arrays to house simulated coordinate positions
    geno_sim=[]
    plot_sim=[]
    rep_sim=[]
    fov_sim=[]
    index_sim=[]
    xcenter_sim=[]
    ycenter_sim=[]


    for fld in flds:
        for rep in reps:
            for fov in fovs:

                g_sim=np.repeat('Simulation', n)
                p_sim=np.repeat(fld, n)
                r_sim=np.repeat(rep+1, n)
                f_sim=np.repeat(fov+1, n)
                i_sim=np.arange(0,n)
                x_sim=np.random.uniform(0, 512 + 1, size=n)
                y_sim=np.random.uniform(0, 512 + 1, size=n)

                geno_sim.append(g_sim)
                plot_sim.append(p_sim)
                rep_sim.append(r_sim)
                fov_sim.append(f_sim)
                index_sim.append(i_sim)
                xcenter_sim.append(x_sim)
                ycenter_sim.append(y_sim)


    geno_sim=np.array(geno_sim).flatten()
    plot_sim=np.array(plot_sim).flatten()
    rep_sim=np.array(rep_sim).flatten()
    fov_sim=np.array(fov_sim).flatten()
    index_sim=np.array(index_sim).flatten()
    xcenter_sim=np.array(xcenter_sim).flatten()
    ycenter_sim=np.array(ycenter_sim).flatten()

    Sim_coords=pd.DataFrame({'Genotype':geno_sim, 'Fieldplot': plot_sim, 'Replicate': rep_sim, 'FOV': fov_sim, 'Index': index_sim, 'x_center': xcenter_sim, 'y_center': ycenter_sim})

    print(str(len(Sim_coords))+' artificial coordinates generated for null model... (NOTE: Given sufficient replication the next phase can take some time!)')

    nullNNs=pd.DataFrame(columns=['Genotype', 'Fieldplot', 'Replicate', 'FOV', 'Current_SC', 'NN_rank', 'NN_dist', 'Origin_X', 'Origin_Y', 'NN_x', 'NN_y', 'NN_dist_xdiff', 'NN_dist_ydiff'])
    
    for fld in flds:
        for rep in reps:
            for fov in fovs:

                Sim_img=Sim_coords.loc[(Sim_coords['Replicate']==rep) & (Sim_coords['FOV']==fov)]
                x_series=Sim_img['x_center']
                y_series=Sim_img['y_center']

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

                    #Remove self NN's with distance of zero as well as all random points less than the average cell size away
                    Dr=Dr[Dr>((ori_len+ori_wid)/2)]

                    NN_dist=Dr[0:(rankno)]

                    rank_index=[]
                    
                    for cur_rdist in NN_dist:
                        rank_index.append(np.where(D==cur_rdist)[0][0])

                    cur_NN=Sim_img.iloc[rank_index,:]

                    cg=np.repeat(geno, rankno)
                    cp=np.repeat(fld, rankno)
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

                    NN_out=pd.DataFrame({'Genotype': cg, 'Fieldplot': cp, 'Replicate': cr, 'FOV': cf, 'Current_SC': ci, 'NN_rank': NN_rank, 'NN_dist': NN_dist, 'Origin_X': ori_x, 'Origin_Y': ori_y, 'NN_x': NN_x,  'NN_y': NN_y, 'NN_dist_xdiff': NN_xdiff, 'NN_dist_ydiff': NN_ydiff})

                    nullNNs = pd.concat([nullNNs, NN_out], axis=0, ignore_index=True)

    return nullNNs

def plot_rankedNN(sample_data, rankedNNs, rank=1, xlimit=512, ylimit=512, bounds=None, plotting=True, plotname='Plotname'):
    
    """
    
    ----------
    Parameters
    ----------
    
    sample_data : (REQUIRED INPUT) A pandas dataframe which follows the standard formatting of this library for initial inputs with the 
                   the following columns expected:
                       'Genotype' -           Genotype for the stomatal coordinates 
                       'Fieldplot' -          Fieldplot or Environmental group of the current Genotype
                       'Replicate' -          Biological Replicate of the current Genotype-Fieldplot
                       'FOV' -                Technical Replicate (i.e., image) of the current Genotype-Fieldplot-Replicate
                       'x_center' -           The centroid x-coordinate location for the stomata
                       'y_center' -           The centroid y-coordinate location for the stomata
                       'length' -             The mask/bounding box length of the stomata
                       'width' -              The mask/bounding box width of the stomata
                       'stoma_area' -         The mask/bounding box area of the stomata
                       'length_width_ratio' - The mask/bounding box length/width ratio of the stomata
    rankedNNs :   (REQUIRED INPUT) A pandas dataframe with rank-order nearest neighbor dataframe generated from stomata_rankedNN(...).
    rank :        (Defaults to 1) A integer which represents the current rank-order nearest neighbor relationship to plot as vectors between 
                   the coordinates.
    xlimit :      (Defaults to 512) A integer which defines the bounds of the x-axis to use for plotting (best if comparable to the original
                   images dimensions.
    ylimit :      (Defaults to 512) A integer which defines the bounds of the y-axis to use for plotting (best if comparable to the original
                   images dimensions.
    bounds :      (Defaults to None) A integer which specifies the bounding edge filter to apply along the x and y axes, which is illustrated
                   as a dotted line box within the plot.
    plotting :    (Defaults to False) Boolean, specifies whether to output a plotting result of the current KDD, when using to generate batch 
                   data particularly for large populations it is advised to keep this set to False.
    plotname :    (Defaults to 'Plotname') Character string where if the default 'plotname' is not retained, the plot will be saved as a PDF
                   using this string argument as the filename.
    
    ----------
    Returns
    ----------
    
    *This function is only used for graphical outputs, no data objects are returned to the program environment.
    
    ----------
    Notes
    ----------
    A supplementary function of the SPP library used to visualize the rank-ordered nearest neighbor relationships of the dataset at a specific
    rank-order.  These are illustrated as vectors radiating from each origin to their current ranks nearest neighbor.
    
    """
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(6,5))

    edge_colors=['red', 'orange', 'green', 'blue', 'navy']

    #Initially plot the bounding boxes of each stomata (based on length/widths around centroid)
    for box in range(0, len(sample_data)):
        cent_x=sample_data.iloc[box,4]
        ori_len=(sample_data.iloc[box,6]/2)

        cent_y=sample_data.iloc[box,5]
        ori_wid=(sample_data.iloc[box,7]/2)

        ax.fill([cent_x-ori_len, cent_x-ori_len, cent_x+ori_len, cent_x+ori_len], [cent_y-ori_wid, cent_y+ori_wid, cent_y+ori_wid, cent_y-ori_wid], linewidth=2, edgecolor=(0,0,0), facecolor=(1,1,1))

    #Plot arrows indicating rank ordered NN relationships between stomata

    NN_edges=rankedNNs.loc[rankedNNs['NN_rank']==rank]

    # Loop through the data and draw arrows
    for i in range(len(NN_edges)):
        ax.arrow(NN_edges.iloc[i,7], NN_edges.iloc[i,8], NN_edges.iloc[i,9]-NN_edges.iloc[i,7], NN_edges.iloc[i,10]-NN_edges.iloc[i,8], linewidth=1.5, head_width=5, head_length=5, fc=edge_colors[rank-1], ec=edge_colors[rank-1])

    #Draw FOV North-West Rule Bounds if needed
    if bounds!=None:
        ax.fill([bounds, bounds, xlimit-bounds, xlimit-bounds], [bounds, ylimit-bounds, ylimit-bounds, bounds], linewidth=4, edgecolor='darkgray', facecolor=(0,0,0,0), linestyle='dotted')


    # Set axis limits
    ax.set_xlim(0, xlimit)
    ax.set_ylim(0, ylimit)
    
    # Set axis labels
    ax.set_title('Rank '+str(rank)+' NN Associations', fontsize="22")
    ax.set_xlabel('Longitudinal Distance', fontsize="16")
    ax.set_ylabel('Lateral Distance', fontsize="16")
    ax.tick_params(axis='both', which='major', labelsize=14)
    #Invert y-axis to mirror normal orientation for images
    plt.gca().invert_yaxis()
    
    # Show the plot
    
    if (plotting==True) & (plotname=='Plotname'):
        plt.show()
    elif (plotting==True) & (plotname!='Plotname'):
        plt.savefig(plotname+'.pdf', format='pdf', bbox_inches='tight')
        plt.close()


def stomata_KDDs(NNSeries, xbound=100, ybound=100, ori_len=20, ori_wid=10, rankno=5, rankmethod='avgrank', plotting=False, plotname='Plotname'):

    """
    
    ----------
    Parameters
    ----------

    
    NNSeries :   (REQUIRED INPUT) A pandas dataframe with rank-order nearest neighbor dataframe generated from stomata_rankedNN(...).
    xbound :     (Defaults to 100) An integer representing the bounds of the x-axis relative to the origin (typically best to slightly 
                  overshoot the broadest rank-order distance.
    ybound :     (Defaults to 100) An integer representing the bounds of the y-axis relative to the origin (typically best to slightly 
                  overshoot the broadest rank-order distance.
    ori_len :    (Defaults to 20) An integer representing the average length of the origin objects (i.e., stomatal complex lengths) 
                  being assessed, used for plotting.
    ori_wid :    (Defaults to 10) An integer representing the average width of the origin objects (i.e., stomatal complex widths) 
                  being assessed, used for plotting.
    rankno :     (Defaults to 5) An integer representing the rank number to used for generating KDDs, either the average among ranks up 
                  to this number or as this current specific number by itself. The context of how rankno is used is governed by the 
                  subsequent argument, rankmethod.
    rankmethod : (Defaults to 'avgrank') Character string specifying the behavior of how the 'rankno' integer will be used. If assigned to 
                  'avgrank' a KDD will be generated of a average from the 1st - rankno sequence of rank-orders.  If assigned to 'currank' 
                  a KDD will be generated corresponding to the current rank-order number presented in rankno.
    plotting :   (Defaults to False) Boolean, specifies whether to output a plotting result of the current KDD, when using to generate batch 
                  data particularly for large populations it is advised to keep this set to False.
    plotname :   (Defaults to 'Plotname') Character string where if the default 'plotname' is not retained, the plot will be saved as a PDF
                  using this string argument as the filename.


    ----------
    Returns
    ----------
    
    Z :           (numpy.ndarray) A 2-dimensional kernel density distribution matrix representing the spatial probability that corresponds to 
                   the rank-order nearest neighbor distances from the origin.
    
    
    ----------
    Notes
    ----------
    A primary function of the SPP library used to convert the rank-ordered nearest neighbors into a 2D probability distribution leveraging a 
    kernel density distribution (KDD) method.  This heatmap represents the Stomatal Patterning Phenotype (SPP) that is intrinsic to this 
    library.
    
    """
        
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
    
    NNcols=[(0.9, 0.8, 0.1), (0.65, 0.45, 0.25), (0, 0, 0), (0.1, 0, 0.9), (0.4, 0, 0.7)]

    cr=np.interp(np.linspace(0, 1, 256), [0.0, 0.25, 0.5, 0.75, 1.0], [0.45, 0.0, 0.0, 0.9, 1.0])
    cg=np.interp(np.linspace(0, 1, 256), [0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.0, 0.0, 0.75, 1.0])
    cb=np.interp(np.linspace(0, 1, 256), [0.0, 0.25, 0.5, 0.75, 1.0], [0.75, 0.75, 0.0, 0.1, 1.0])

    kde_colchan=np.vstack((cr, cg, cb)).T
    kde_cmap=mcolors.ListedColormap(kde_colchan)

    # Plot the kernel density estimate
    #plt.figure(figsize=(8,8))
    
    if (plotting==True):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

        ax1.set_xlim([-xbound,xbound])
        ax1.set_ylim([-ybound,ybound])
        ax1.set_xlabel('Distance (um)', fontsize=16)
        ax1.set_ylabel('Distance (um)', fontsize=16)
        ax1.set_title('NN Distances', fontsize=22)
        ax1.tick_params(axis='both', which='major', labelsize=14)

        if rankmethod=='avgrank':
            for i in range(rankno, 0, -1):
                ax1.plot(NNSeries[NNSeries['NN_rank']==i]['NN_dist_xdiff'], NNSeries[NNSeries['NN_rank']==i]['NN_dist_ydiff'], '.', color=NNcols[i-1])
        else:
            for i in range(rankno, 0, -1):
                ax1.plot(NNSeries[NNSeries['NN_rank']==rankno]['NN_dist_xdiff'], NNSeries[NNSeries['NN_rank']==rankno]['NN_dist_ydiff'], '.', color=NNcols[rankno-1])

        ax1.fill([-ori_len, -ori_len, ori_len, ori_len], [-ori_wid, ori_wid, ori_wid, -ori_wid], linewidth=2, edgecolor=(0,0,0), facecolor=(1,1,1))

        ax2.set_xlabel('Distance (um)', fontsize=16)
        ax2.set_title('NN Distances', fontsize=22)
        ax2.tick_params(axis='both', which='major', labelsize=14)
        im = ax2.imshow(Z/np.max(Z), aspect='auto', extent=[xmin, xmax, ymin, ymax], cmap='inferno')
        ax2.fill([-ori_len, -ori_len, ori_len, ori_len], [-ori_wid, ori_wid, ori_wid, -ori_wid], linewidth=2, edgecolor=(0,0,0), facecolor=(1,1,1))

        fig.colorbar(im, ax=ax2)

    if (plotname=='Plotname'):
        plt.show()
    elif (plotname!='Plotname'):
        fig.savefig(plotname+'.pdf', format='pdf', bbox_inches='tight')
        plt.close(fig)
        
    return Z

def stomata_KDD_hist(NNSeries, Z, xbound, ybound, ori_len=20, ori_wid=10, rankno=5, plotting=False, plotname='Plotname', horimax=None, vertmax=None):

    
    """
    
    ----------
    Parameters
    ----------

    NNSeries :   (REQUIRED INPUT) A pandas dataframe with rank-order nearest neighbor dataframe generated from stomata_rankedNN(...).
    Z :          (REQUIRED INPUT) A 2-dimensional numpy array representing the kernel density distribution of the rank-order nearest 
                  neighbors generated from the stomata_KDDs(...) function.
    xbound :     (Defaults to 100) An integer representing the bounds of the x-axis relative to the origin (typically best to slightly 
                  overshoot the broadest rank-order distance.
    ybound :     (Defaults to 100) An integer representing the bounds of the y-axis relative to the origin (typically best to slightly 
                  overshoot the broadest rank-order distance.
    ori_len :    (Defaults to 20) An integer representing the average length of the origin objects (i.e., stomatal complex lengths) 
                  being assessed, used for plotting.
    ori_wid :    (Defaults to 10) An integer representing the average width of the origin objects (i.e., stomatal complex widths) 
                  being assessed, used for plotting.. 
    rankno -     (Defaults to 5) An integer representing the rank number to used for generating KDDs, specifically the average among ranks up 
                  to this number, this function lacks a 'currank' rankmethod behavior akin to the stomata_KDDs(...) function.  
    plotting :   (Defaults to False) Boolean, specifies whether to output a plotting result of the current KDD, when using to generate batch 
                  data particularly for large populations it is advised to keep this set to False.
    plotname :   (Defaults to 'Plotname') Character string where if the default 'plotname' is not retained, the plot will be saved as a PDF
                  using this string argument as the filename.
    horimax :    (Defaults to None) Integer representing the ceiling when plotting the horizontal density distribution, this can be useful when 
                  comparing multiple flattened histogram plots as the magnitude can be normalized to a global ceiling.
    vertmax :    (Defaults to None) Integer representing the ceiling when plotting the vertical density distribution, this can be useful when 
                  comparing multiple flattened histogram plots as the magnitude can be normalized to a global ceiling.
    
    
    ----------
    Returns
    ----------
    
    hp :          (numpy.ndarray) A horizontally flattened 1-dimensional probability density distribution corresponding to the 2D KDD object, 'Z' 
                   which was used as an input.
    vp :          (numpy.ndarray) A vertically flattened 1-dimensional probability density distribution corresponding to the 2D KDD object, 'Z' 
                   which was used as an input.

    
    ----------
    Notes
    ----------
    
    A primary function of the SPP library used to extract flattened 1-dimensional horizontal (hp) and vertical (vp) distributions corresponding to
    the Z spatial probabilities representing the Stomatal Patterning Phenotype (SPP).  This data reduction can help with spatial annotations and 
    downstream trait quantification, particularly from the 'vp' distribution in grass systems.
    
    """
        
    
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

    if (plotting==True):
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, gridspec_kw={'width_ratios': [8, 4], 'height_ratios': [4, 8]}, figsize=(12, 12))
        ax2.axis('off')

        #gs = fig.add_gridspec(2, 2, height_ratios=[1, 2], width_ratios=[2, 1])

        #ax1 = fig.add_subplot(gs[0, 0])
        #plt.figure(figsize=(8,4))
        ax1.set_ylabel('Average Prob.', fontsize=18)
        ax1.set_title('Horizontal NN Distances', fontsize=20)
        ax1.tick_params(axis='both', which='major', labelsize=14)
        if (horimax!=None):
            ax1.set_ylim([np.min(hori_p),horimax])
        ax1.plot(hori_x, hori_p)

        #ax4 = fig.add_subplot(gs[1, 1])
        #plt.figure(figsize=(4,8))
        ax4.set_xlabel('Average Prob.', fontsize=18)
        ax4.set_title('Vertical NN Distances', fontsize=22)
        ax4.tick_params(axis='both', which='major', labelsize=14)
        if (vertmax!=None):
            ax4.set_xlim([np.min(vert_p),vertmax])
        ax4.plot(vert_p, vert_y)

        #ax3 = fig.add_subplot(gs[1, 0])
        #plt.figure(figsize=(8,8))
        ax3.set_xlabel('Distance (um)', fontsize=18)
        ax3.set_ylabel('Distance (um)', fontsize=18)
        ax3.set_title('SPP Phenotype', fontsize=22)
        ax3.tick_params(axis='both', which='major', labelsize=14)
        im = ax3.imshow(Z/np.max(Z), aspect='auto', extent=[xmin, xmax, ymin, ymax], cmap='inferno')
        ax3.fill([-ori_len, -ori_len, ori_len, ori_len], [-ori_wid, ori_wid, ori_wid, -ori_wid], linewidth=2, edgecolor=(0,0,0), facecolor=(1,1,1))
        

        if (plotname!='Plotname'):
            plt.savefig(plotname+'.pdf', format='pdf', bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    return hori_p, vert_p


def stomata_KDD_deriv_anno(vp, Z, xbound, ybound, ori_len=20, ori_wid=10, plotting=False, plotname='Plotname'):

    """
    ----------
    Parameters
    ----------
    
    vp :       (REQUIRED INPUT) A 1-dimensional numpy array representing a vertically flattened density distribution generated from 
                stomata_KDD_hist(...) function.
    Z :        (REQUIRED INPUT) The Z probability surface KDD generated from stomata_KDDs(...), used for plotting.
    xbound :   (Defaults to 100) An integer representing the bounds of the x-axis relative to the origin (typically best to slightly 
                overshoot the broadest rank-order distance.
    ybound :   (Defaults to 100) An integer representing the bounds of the y-axis relative to the origin (typically best to slightly 
                overshoot the broadest rank-order distance.
    ori_len :  (Defaults to 20) An integer representing the average length of the origin objects (i.e., stomatal complex lengths) 
                being assessed, used for plotting.
    ori_wid :  (Defaults to 10) An integer representing the average width of the origin objects (i.e., stomatal complex widths) 
                being assessed, used for plotting.. 
    plotting : (Defaults to False) Boolean, specifies whether to output a plotting result of the current KDD, when using to generate batch 
                data particularly for large populations it is advised to keep this set to False.
    plotname : (Defaults to 'Plotname') Character string where if the default 'plotname' is not retained, the plot will be saved as a PDF
                using this string argument as the filename.
    
    ----------
    Returns
    ----------

    origin_trench_dist : (numpy.float64) A floating point number representing the distance between the probability trenches and the 
                          origin's in-file peak.
    origin_peak_dist :   (numpy.float64)  A floating point number representing the distance between the side-file peaks and the origin's
                          in-file peak.
    trenchprob_FC :      (numpy.float64) A floating point number representing the fold change difference between the proabilities of the 
                          probability trenches compared to the primary in-file peak.
    Peaksprob_FC :       (numpy.float64) A floating point number representing the fold change difference between the proabilities of the 
                          side-file peaks compared to the primary in-file peak.
    
    ----------
    Notes
    ----------
    
    A primary function of the SPP library used to generate positional annotations for the probability trench and side-file peaks of a 
    vertically flattened 1-dimensional distribution which are subsequently used to generate distance and probability fold change trait
    measurements. 
    
    
    """
    
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

    im = plt.imshow(Z/np.max(Z), aspect='auto', extent=[-xbound, xbound, -ybound, ybound], cmap='inferno')

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


def stomata_compare_KDDs(obsZ, modelZ, xbound, ybound, ori_len, ori_wid, rankno, plotting=False, plotname='Plotname'):

    """
    ----------
    Parameters
    ----------

    obsZ :       (REQUIRED INPUT) A 2-dimensional numpy array representing the observed kernel density distribution of the rank-order 
                  nearest neighbors generated from the stomata_KDDs(...) function.
    modelZ :     (REQUIRED INPUT) A 2-dimensional numpy array representing a modeled (or alternative observed) kernel density distribution 
                  of the rank-order nearest neighbors generated from the stomata_KDDs(...) function.
    xbound :     (Defaults to 100) An integer representing the bounds of the x-axis relative to the origin (typically best to slightly 
                  overshoot the broadest rank-order distance.
    ybound :     (Defaults to 100) An integer representing the bounds of the y-axis relative to the origin (typically best to slightly 
                  overshoot the broadest rank-order distance.
    ori_len :    (Defaults to 20) An integer representing the average length of the origin objects (i.e., stomatal complex lengths) 
                  being assessed, used for plotting.
    ori_wid :    (Defaults to 10) An integer representing the average width of the origin objects (i.e., stomatal complex widths) 
                  being assessed, used for plotting.. 
    rankno :     (Defaults to 5) An integer representing the rank number to used for generating KDDs, specifically the average among ranks up 
                  to this number, this function lacks a 'currank' rankmethod behavior akin to the stomata_KDDs(...) function.  
    plotting :   (Defaults to False) Boolean, specifies whether to output a plotting result of the current KDD, when using to generate batch 
                  data particularly for large populations it is advised to keep this set to False.
    plotname :   (Defaults to 'Plotname') Character string where if the default 'plotname' is not retained, the plot will be saved as a PDF
                  using this string argument as the filename. 
    
    ----------
    Returns
    ----------
    
    OEscore :    (numpy.float64) A floating point number representing a overall score of the absolute difference between all X-Y kernel grid 
                  positions between the obsZ and modelZ numpy array inputs.
    diffZ :      (numpy.ndarray) A 2-dimensional numpy array representing the differences at each X-Y kernel grid position between the obsZ 
                  and modelZ input arrays.
    
    ----------
    Notes
    ----------
    
    A supplementary function of the SPP library used to enable comparison between two SPP 2-dimensional arrays either as a observed vs. null 
    model comparison or through comparing two genotypes in order to recognize where these two spatial probabilities vary and in what direction, 
    given that positive values represent a 'obsZ' bias whereas negative values represent a 'modelZ' bias.
    
    """
    
    cr=np.interp(np.linspace(0, 1, 256), [0.0, 0.25, 0.5, 0.75, 1.0], [0.45, 0.0, 0.0, 0.9, 1.0])
    cg=np.interp(np.linspace(0, 1, 256), [0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.0, 0.0, 0.75, 1.0])
    cb=np.interp(np.linspace(0, 1, 256), [0.0, 0.25, 0.5, 0.75, 1.0], [0.75, 0.75, 0.0, 0.1, 1.0])

    kde_colchan=np.vstack((cr, cg, cb)).T
    kde_cmap=mcolors.ListedColormap(kde_colchan)

    xmin, xmax = -xbound, xbound
    ymin, ymax = -ybound, ybound

    diffZ=(obsZ-modelZ)

    floor=np.min([np.min(obsZ), np.min(modelZ)])
    ceiling=np.max([np.max(obsZ), np.max(modelZ)])

    if plotting==True:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))

        ax1.set_xlabel('Distance (um)', fontsize=16)
        ax1.set_ylabel('Distance (um)', fontsize=16)
        ax1.set_title('Observed SPP', fontsize=22)
        ax1.tick_params(axis='both', which='major', labelsize=14)
        im1 = ax1.imshow(obsZ, aspect='auto', extent=[xmin, xmax, ymin, ymax], cmap='inferno', vmin=floor, vmax=ceiling)
        ax1.fill([-ori_len, -ori_len, ori_len, ori_len], [-ori_wid, ori_wid, ori_wid, -ori_wid], linewidth=2, edgecolor=(0,0,0), facecolor=(1,1,1))
        
        ax2.set_xlabel('Distance (um)', fontsize=16)
        ax2.set_title('Expected Model', fontsize=22)
        ax2.tick_params(axis='both', which='major', labelsize=14)
        im2 = ax2.imshow(modelZ, aspect='auto', extent=[xmin, xmax, ymin, ymax], cmap='inferno', vmin=floor, vmax=ceiling)
        ax2.fill([-ori_len, -ori_len, ori_len, ori_len], [-ori_wid, ori_wid, ori_wid, -ori_wid], linewidth=2, edgecolor=(0,0,0), facecolor=(1,1,1))
        
        ax3.set_xlabel('Distance (um)', fontsize=16)
        ax3.set_title('Obs. - Exp. Difference', fontsize=22)
        ax3.tick_params(axis='both', which='major', labelsize=14)
        im3 = ax3.imshow(diffZ, aspect='auto', extent=[xmin, xmax, ymin, ymax], cmap='inferno')
        ax3.fill([-ori_len, -ori_len, ori_len, ori_len], [-ori_wid, ori_wid, ori_wid, -ori_wid], linewidth=2, edgecolor=(0,0,0), facecolor=(1,1,1))
        plt.colorbar(im3, ax=ax3)

        if plotname!='Plotname':
            fig.savefig(plotname+'.pdf', format='pdf', bbox_inches='tight')
            plt.close(fig)

    OEscore=np.round(np.sum(np.abs((obsZ/np.max(obsZ))-(modelZ/np.max(modelZ)))),3)

    return OEscore, diffZ

def stomata_KDD_rorshach(Z, plotting=False):
    
    """
    
    ----------
    Parameters
    ----------
    
    Z :          (REQUIRED INPUT) A 2D numpy array representing the Z probability surface KDD generated from stomata_KDDs(...).
    plotting :   (Defaults to False) Boolean, specifies whether to output a plotting result of the current KDD, when using to generate batch 
                  data particularly for large populations it is advised to keep this set to False.
    
    ----------
    Returns
    ----------
    Zrorshach :   (numpy.ndarray) A rorshach-folding normalized version of the original Z numpy array input.
    
    
    ----------
    Notes
    ----------
    A supplementary function of the SPP library used to normalize uneven Z probability surfaces by folding the 4 quadrants relative to the origin 
    over one another and averaging (akin to a rorshach-blot, hence the name) then using this averaged value to generate a new normalized Z array.
   
     
    """

    xlim=int(Z.shape[0])
    ylim=int(Z.shape[1])
    xcent=int((xlim/2))
    ycent=int((ylim/2))

    #print(xlim, ylim)
    #print(xcent, ycent)

    Zrorshach=Z.copy()

    Zq1=Z[0:(ycent),xcent:(xlim+1)]
    Zq2=Z[0:(ycent),0:(xcent)]
    Zq3=Z[ycent:(ylim+1),0:(xcent)]
    Zq4=Z[ycent:(ylim+1),xcent:(xlim+1)]

    #print(Zq1.shape,Zq2.shape,Zq3.shape,Zq4.shape)

    #Flip inverted quadrants
    Zq2_flip=np.flip(Zq2, axis=1)
    Zq3_flip=np.flip(np.flip(Zq3, axis=0), axis=1)
    Zq4_flip=np.flip(Zq4, axis=0)

    Zqr1=(Zq1+Zq2_flip+Zq3_flip+Zq4_flip)/4
    Zqr2=np.flip(Zqr1, axis=1)
    Zqr3=np.flip(np.flip(Zqr1, axis=0), axis=1)
    Zqr4=np.flip(Zqr1, axis=0)

    Zrorshach[0:(ycent),xcent:(xlim+1)]=Zqr1
    Zrorshach[0:(ycent),0:(xcent)]=Zqr2
    Zrorshach[ycent:(ylim+1),0:(xcent)]=Zqr3
    Zrorshach[ycent:(ylim+1),xcent:(xlim+1)]=Zqr4

    if plotting==True:
        plt.figure(figsize=(8,8))
        plt.imshow(Zrorshach, cmap='inferno')
        plt.colorbar()
        
    return Zrorshach


def stomata_KDD_peak_spacing(Zif, Zsp, xbound, ybound, masking_val=0.5, trench_thresh=25, buffer=5, plotting=False, plotname='Plotname'):

    """
    ----------
    Parameters
    ----------
    
    Zif :           (REQUIRED INPUT) The Z probability surface focused on the in-file peaks KDD generated from stomata_KDDs(...).
    Zsp :           (REQUIRED INPUT) The Z probability surface focused on the side-file peaks KDD generated from stomata_KDDs(...).
    xbound :        (Defaults to 100) An integer representing the bounds of the x-axis relative to the origin (typically best to slightly 
                     overshoot the broadest rank-order distance.
    ybound :        (Defaults to 100) An integer representing the bounds of the y-axis relative to the origin (typically best to slightly 
                     overshoot the broadest rank-order distance.
    masking_val :   (Defaults to 0.5) floating point number ranging from 0.0 - 1.0, T
    trench_thresh : (Defaults to 10) Integer
    buffer -        (Defaults to 5) Integer
    ori_len :       (Defaults to 20) An integer representing the average length of the origin objects (i.e., stomatal complex lengths) 
                     being assessed, used for plotting.
    ori_wid :       (Defaults to 10) An integer representing the average width of the origin objects (i.e., stomatal complex widths) 
                     being assessed, used for plotting.. 
    plotting :      (Defaults to False) Boolean, specifies whether to output a plotting result of the current KDD, when using to generate batch 
                     data particularly for large populations it is advised to keep this set to False.
    plotname :      (Defaults to 'Plotname') Character string where if the default 'plotname' is not retained, the plot will be saved as a PDF
                     using this string argument as the filename.
    
    ----------
    Returns
    ----------

    if_len :        (float) A floating point number representing the in-file mask x-axis bounding box length, used as a measure of variance for 
                     the in-file peak.
    if_wid :        (float) A floating point number representing the in-file mask y-axis bounding box width, used as a measure of variance for 
                     the in-file peak.
    if_area :       (float) A floating point number representing the in-file mask bounding box area, used as a measure of variance for the in-file 
                     peak.
    if_dist :       (float) A floating point number representing the centroid distance between the in-file masks and the origin.
    sp_len :        (float) A floating point number representing the side-file mask x-axis bounding box length, used as a measure of variance for 
                     the side-file peak.
    sp_wid :        (float) A floating point number representing the side-file mask y-axis bounding box width, used as a measure of variance for 
                     the side-file peak.
    sp_area :       (float) A floating point number representing the side-file mask bounding box area, used as a measure of variance for the 
                     side-file peak.
    sp_dist :       (float) A floating point number representing the centroid distance between the side-file masks and the origin.
    
    ----------
    Notes
    ----------
    
    A primary function of the SPP library used to generate two sets of binary masks for the in-file and side-file peaks respectively which are then 
    used to generate a series of variance measures based on the bounding box length, width, and areas for these peaks as well as their corresponding
    centroid distance from the origin.
    
    
    """
        
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

    # if_mask=(rows < (midrow - trench_thresh)) | (rows > (midrow + trench_thresh))
    # Z_mask[if_mask]=0
    # Z_mask[Z_mask>masking_val]=1
    # Z_mask[Z_mask<masking_val]=0

    if_y_mask=(rows < (midrow - trench_thresh)) | (rows > (midrow + trench_thresh))
    if_x_mask=(rows < (midrow + buffer)) & (rows > (midrow - buffer))
    Z_mask[if_y_mask,:]=0
    Z_mask[:,if_x_mask]=0
    Z_mask[Z_mask>masking_val]=1
    Z_mask[Z_mask<masking_val]=0

    #Iterate through the binary mask and label each conterminous mask with an identifier, then generate a bounding box to measure it
    labeled_array_if, num_labels = ndimage.label(Z_mask == 1)
    bounding_boxes = ndimage.find_objects(labeled_array_if)

    mask_sizes = np.bincount(labeled_array_if.ravel())  # Calculate sizes
    sorted_labels = np.argsort(mask_sizes)[::-1]        # Identify the labels corresponding to each mask
    primary_labels = sorted_labels[1:3]                 # Sort the labels to extract the 2nd and 3rd (i.e. two primary/non-background) masks

    # Filter out the labels not in secondary_labels
    fragment_if = sorted_labels[np.isin(sorted_labels, primary_labels, invert=True)]

    # Set positions to zero in Z_mask if their mask is deemed a fragment
    for fraglabel in fragment_if:
        labeled_array_if[labeled_array_if == fraglabel] = 0

    
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

    mask_sizes = np.bincount(labeled_array_sp.ravel())  # Calculate sizes
    sorted_labels = np.argsort(mask_sizes)[::-1]        # Identify the labels corresponding to each mask
    secondary_labels = sorted_labels[1:3]               # Sort the labels to extract the 2nd and 3rd (i.e. two primary/non-background) masks

    # Filter out the labels not in secondary_labels
    fragment_sf = sorted_labels[np.isin(sorted_labels, secondary_labels, invert=True)]
    
    # Set positions to zero in Z_mask2 if their mask is deemed a fragment
    for fraglabel in fragment_sf:
        labeled_array_sp[labeled_array_sp == fraglabel] = 0
        
        
    for label in secondary_labels:  # Exclude background label 0
        patch_slice = bounding_boxes[label - 1]  # Get the slice of the patch
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

        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(14,14))
        ax1.imshow(Zif, extent=extent, cmap='inferno')
        ax1.set_title('In-file Peak KDDs')
        ax2.imshow(Zsp, extent=extent, cmap='inferno')
        ax2.set_title('Side Peak KDDs')
        ax3.imshow(labeled_array_if, extent=extent, cmap='inferno')
        ax3.scatter(cent_x[0:2], cent_y[0:2], color='green')
        ax3.set_title('In-file Peak Masks')
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
    