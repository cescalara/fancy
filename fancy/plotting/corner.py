import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from .HPD_regions import plot_HPD_levels
from .colours import purple

__all__  = ['Corner']


class Corner():
    """
    Simple corner plotting.
    """

    def __init__(self, data_frame, truth = None, color = purple, levels = [0.99, 0.9, 0.6, 0.3], fontsize = 22):
        """
        Simple corner plotting.
        
        :param data_frame: pandas DataFrame object
        :param truth: list of true values
        :param color: colour of plots
        """

        # Fontsize
        plt.rcParams['font.size'] = fontsize
        plt.rcParams['axes.labelsize'] = fontsize
        plt.rcParams['xtick.labelsize'] = fontsize - 2
        plt.rcParams['ytick.labelsize'] = fontsize - 2
        
        pairgrid = sns.PairGrid(data_frame, diag_sharey = False, despine = False)

        # Lower triangle only
        for i, j in zip(*np.triu_indices_from(pairgrid.axes, 1)):
            pairgrid.axes[i, j].set_visible(False)
    
        # KDE plots
        pairgrid = pairgrid.map_diag(sns.kdeplot, color = color, shade = True, lw = 2, zorder = 2)
        pairgrid = pairgrid.map_offdiag(plot_HPD_levels, color = color, shade = True, levels = levels)

        # Truths
        if truth:
        
            N = len(truth)
            for i, t in enumerate(truth):
                for j in range(N):
                    pairgrid.axes[j, i].axvline(t, lw = 2, color = 'k', zorder = 5, alpha = 0.7)
                    for k in range(N):
                        if i != k:
                            pairgrid.axes[i, k].axhline(t, lw = 2, color = 'k', zorder = 5, alpha = 0.7)
                            pairgrid.axes[i, k].spines['right'].set_visible(True)
                            pairgrid.axes[i, k].spines['top'].set_visible(True) 
                        else:
                            sns.despine(ax = pairgrid.axes[i, k])
        else:

            N = np.shape(data_frame)[1]
            for i in range(N):
                for j in range(N):
                    for k in range(N):
                        if i != k:
                            pairgrid.axes[i, k].spines['right'].set_visible(True)
                            pairgrid.axes[i, k].spines['top'].set_visible(True) 
                        else:                   
                            sns.despine(ax = pairgrid.axes[i, k])
     
        # Tidy axes
        sns.despine(ax = pairgrid.axes[0, 0], left = True)
        pairgrid.axes[0, 0].get_yaxis().set_visible(False)
        plt.subplots_adjust(hspace = 0.05, wspace = 0.055)


    
    def save(self, filename):
        """
        Save to file.
        """

        plt.savefig(filename, dpi = 500, bbox_inches = 'tight')


