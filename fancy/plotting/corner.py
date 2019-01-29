from matplotlib import pyplot as plt
import seaborn as sns

from .colours import purple

__all__  = ['Corner']


class Corner():
    """
    Simple corner plotting.
    """

    def __init__(self, data_frame, truth, color = purple):
        """
        Simple corner plotting.
        
        :param data_frame: pandas DataFrame object
        :param truth: list of true values
        :param color: colour of plots
        """
        
        pairgrid = sns.PairGrid(df, diag_sharey = False, despine = False)

        # Lower triangle only
        for i, j in zip(*np.triu_indices_from(pairgrid.axes, 1)):
            pairgrid.axes[i, j].set_visible(False)
    
        # KDE plots
        pairgrid = pairgrid.map_diag(sns.kdeplot, color = purple, shade = True, lw = 2, zorder = 2)
        pairgrid = g.map_offdiag(plot_HPD_levels, color = purple, shade = True, lw = 3)

        # truths
        N = len(truth)
        for i, t in enumerate(truth):
            for j in range(N):
                pairgrid.axes[j, i].axvline(t, lw = 2, color = 'k', zorder = 5, alpha = 0.7)
            for k in range(N):
                if i != k:
                    pairgrid.axes[i, k].axhline(t, lw = 2, color = 'k', zorder = 5, alpha = 0.7)
                else:
                    sns.despine(ax = pairgrid.axes[i, k])

        # Tidy axes
        sns.despine(ax = pairgrid.axes[0, 0], left = True)
        pairgrid.axes[0, 0].get_yaxis().set_visible(False)
        plt.subplots_adjust(hspace=0.05, wspace=0.055)

        return pairgrid

    
    def save(self, filename):
        """
        Save to file.
        """

        plt.savefig(filename, dpi = 500, bbox_inches = 'tight')


