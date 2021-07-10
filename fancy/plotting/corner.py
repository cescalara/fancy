import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from .colours import purple

__all__ = ['Corner']


class Corner():
    """
    Simple corner plotting.
    """
    def __init__(self,
                 data_frame,
                 truth=None,
                 color=purple,
                 contour_color=purple,
                 levels=[0.97, 0.9, 0.6, 0.3, 0.],
                 fontsize=22):
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

        # corner=True allows for lower region plots only,
        # no need to do anything else fancy as before
        pairgrid = sns.PairGrid(data_frame,
                                diag_sharey=False,
                                despine=False,
                                corner=True)

        # KDE plots
        # KW: - contour levels face the same probem as it passes through HPD_contours
        #     so we simply use the levels as given in the argument
        #     but flip it due to how levels are registered
        #     - Also contour colors are now set separate to the colors for
        #     diag plots due to color differences

        levels = [1. - l for l in levels]

        pairgrid.map_diag(sns.kdeplot, color=color, shade=True, lw=2, zorder=2)
        pairgrid.map_offdiag(sns.kdeplot,
                             color=contour_color,
                             shade=True,
                             levels=levels)

        # Truths
        # KW: plots the lines within the plots that show the true value
        if truth:
            N = len(truth)
            for i, t in enumerate(truth):
                for j in range(N):
                    if pairgrid.axes[j, i]:
                        pairgrid.axes[j, i].axvline(t,
                                                    lw=2,
                                                    color='k',
                                                    zorder=5,
                                                    alpha=0.7)
                        for k in range(N):
                            if pairgrid.axes[i, k]:
                                if i != k:
                                    pairgrid.axes[i, k].axhline(t,
                                                                lw=2,
                                                                color='k',
                                                                zorder=5,
                                                                alpha=0.7)
                                    pairgrid.axes[
                                        i, k].spines['right'].set_visible(True)
                                    pairgrid.axes[
                                        i, k].spines['top'].set_visible(True)
                                else:
                                    sns.despine(ax=pairgrid.axes[i, k])
        else:
            N = np.shape(data_frame)[1]
            for i in range(N):
                for j in range(N):
                    for k in range(N):
                        # KW: if axes is NoneType, since corner=True makes unused axes
                        # into NoneTypes
                        if pairgrid.axes[i, k]:
                            if i != k:
                                pairgrid.axes[
                                    i, k].spines['right'].set_visible(True)
                                pairgrid.axes[i, k].spines['top'].set_visible(
                                    True)
                            else:
                                sns.despine(ax=pairgrid.axes[i, k])

        # Tidy axes
        sns.despine(ax=pairgrid.axes[0, 0], left=True)
        pairgrid.axes[0, 0].get_yaxis().set_visible(False)
        # plt.subplots_adjust(hspace=0.05, wspace=0.055)

        # KW: this does fine since it considers badly placed axis labels
        pairgrid.tight_layout()

    def save(self, filename):
        """
        Save to file.
        """

        plt.savefig(filename, dpi=500, bbox_inches='tight')
