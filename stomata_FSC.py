import pandas as pd
import numpy as np
from scipy import ndimage

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

    for ang in np.arange(0, 360, 0.1):
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

    print(best_ang, bestm)

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