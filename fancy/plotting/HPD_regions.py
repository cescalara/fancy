import numpy as np
from scipy.optimize import bisect
from scipy import interpolate
import seaborn as sns


def grid_HPD_contours(data, levels):
    """
    Get the contours for a 2D grid of data at the
    specified levels as a fraction the integral over the whole grid.
    """

    N = int(1e3)
    thresholds = np.linspace(0, data.max(), N)
    integral = ((data >= thresholds[:, None, None]) * data).sum(axis=(1, 2))
    norm = integral[0]

    # Interpolate required threshold from integral
    function = interpolate.interp1d(integral, thresholds)
    contours = function(np.array(levels) * norm)

    return contours


def HPD_contours(xdata, ydata, levels, bins=50):
    """
    Estimate the HPD contours from sample chains (xdata, ydata).
    Output are levels which can be input into seaborn KDE plots.

    KW: Obtained from https://stackoverflow.com/questions/35225307/set-confidence-levels-in-seaborn-kdeplot 
    KW: This somehow returns a weird series of levels that start with 1e-5 to 1e-3, and she uses n_levels instead of levels (which strangely passes to plt.contourf as levels), additionally her 6 contour levels only show 4 in the resulting plot. 
    All in all, after some playing around we can ignore this code and simply use her arbitrarily set levels in the core code.
    """

    # 2D normed histogram
    H, xedges, yedges = np.histogram2d(xdata, ydata, bins=bins, normed=True)
    norm = H.sum()

    # Set target levels as percentage of norm
    target = [norm*l for l in levels]

    def objective(limit, target):
        w = np.where(H > limit)
        count = H[w]
        return count.sum() - target

    # Find levels by summing up histogram to target
    output_levels = []
    for t in target:
        output_levels.append(bisect(objective, H.min(), H.max(), args=(t,)))

    # Define bound level for shading in seaborn KDE.
    output_levels.append(H.max())
    return output_levels


def plot_HPD_levels(xdata, ydata, levels=[0.99, 0.9, 0.6, 0.3],
                    bins=50, **kwargs):
    """
    Plot a KDE plot for the data with specified levels.
    """

    new_levels = HPD_contours(xdata, ydata, levels, bins)
    sns.kdeplot(xdata, ydata, n_levels=new_levels,
                **kwargs)


def plot_HPD_levels_multi(xdata, ydata, levels=[0.99, 0.9, 0.6, 0.3],
                          bins=50, **kwargs):
    """
    Plot a KDE plot for the data with specified levels.
    Handles multiple colors for same dataframe.
    """

    my_color = next(plot_HPD_levels.color_cycle)
    new_levels = HPD_contours(xdata, ydata, levels, bins)
    sns.kdeplot(xdata, ydata, n_levels=new_levels, normed=True,
                color=my_color, shade=True, alpha=0.5)
