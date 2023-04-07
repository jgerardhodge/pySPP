import pandas as pd
import numpy as np

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