import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from .colours import purple

__all__ = ["Corner"]


class Corner:
    """
    Simple corner plotting.
    """

    def __init__(
        self,
        data_frame,
        truth=None,
        color=purple,
        contour_color=purple,
        levels=[0.97, 0.9, 0.6, 0.3, 0.0],
        fontsize=16,
        end_label=None,
    ):
        """
        Simple corner plotting.

        :param data_frame: pandas DataFrame object
        :param truth: list of true values
        :param color: colour of plots
        """

        # Fontsize
        plt.rcParams["font.size"] = fontsize
        plt.rcParams["axes.labelsize"] = fontsize - 2
        plt.rcParams["xtick.labelsize"] = fontsize - 5
        plt.rcParams["ytick.labelsize"] = fontsize - 5

        # corner=True allows for lower region plots only,
        # no need to do anything else fancy as before
        pairgrid = sns.PairGrid(
            data_frame, diag_sharey=False, despine=False, corner=True
        )

        # find index to modify xlims etc for
        if r'$B \: l_C^{1/2}$ / $\mathrm{nG} \: \mathrm{Mpc}^{1/2}$' in data_frame.columns:  # B-field
            b_idx = np.argwhere(data_frame.columns == r'$B \: l_C^{1/2}$ / $\mathrm{nG} \: \mathrm{Mpc}^{1/2}$')[0][0]
        else:
            b_idx = None

        if r"$\kappa$" in data_frame.columns:  # kappa
            kappa_idx = np.argwhere(data_frame.columns == r"$\kappa$")[0][0]
        else:
            kappa_idx = None

        f_idx = np.argwhere(data_frame.columns == r"$f$")[0][0]  # f

        if r"$L$ / $10^{{44}}$ $\mathrm{erg}$ $\mathrm{s}^{{-1}}$" in data_frame.columns:
            L_idx = np.argwhere(
                data_frame.columns == r"$L$ / $10^{{44}}$ $\mathrm{erg}$ $\mathrm{s}^{{-1}}$"
            )[0][
                0
            ]  # L
        else:
            L_idx = None

        if end_label == "tight_B" or end_label == "limitL":
            B_lim = [9.5, 10]
            B_ticks = [9.5, 9.75, 10]
        else:
            B_lim = [9.5, 10]
            B_ticks = [9.5, 9.75, 10]

        # KDE plots
        # KW: - contour levels face the same probem as it passes through HPD_contours
        #     so we simply use the levels as given in the argument
        #     but flip it due to how levels are registered
        #     - Also contour colors are now set separate to the colors for
        #     diag plots due to color differences

        levels = [1.0 - l for l in levels]

        pairgrid.map_diag(sns.kdeplot, color=color, shade=True, lw=2, zorder=2)
        pairgrid.map_offdiag(
            sns.kdeplot, color=contour_color, shade=True, levels=levels
        )

        # Truths
        # KW: plots the lines within the plots that show the true value
        if truth:
            N = len(truth)
            for i, t in enumerate(truth):
                for j in range(N):
                    if pairgrid.axes[j, i]:
                        pairgrid.axes[j, i].axvline(
                            t, lw=2, color="k", zorder=5, alpha=0.7
                        )
                        for k in range(N):
                            if pairgrid.axes[i, k]:
                                if i != k:
                                    pairgrid.axes[i, k].axhline(
                                        t, lw=2, color="k", zorder=5, alpha=0.7
                                    )
                                    pairgrid.axes[i, k].spines["right"].set_visible(
                                        True
                                    )
                                    pairgrid.axes[i, k].spines["top"].set_visible(True)
                                else:
                                    sns.despine(ax=pairgrid.axes[i, k])
                                    if b_idx is not None:
                                        if i == b_idx:
                                            pairgrid.axes[i, k].set_xlim(B_lim)
                                            pairgrid.axes[i, k].set_xticks(B_ticks)

                                        if k == b_idx:
                                            pairgrid.axes[i, k].set_ylim(B_lim)
                                            pairgrid.axes[i, k].set_yticks(B_ticks)

                                    if kappa_idx is not None:
                                        if i == kappa_idx:
                                            # pairgrid.axes[i, k].set_xlim(0, 1000)
                                            pairgrid.axes[i, k].set_xscale("log")
                                            # pairgrid.axes[i, k].set_xticks([0, 10, 100, 1000])
                                            # pairgrid.axes[i, k].get_xaxis().get_major_formatter().labelOnlyBase = False
                                            # pairgrid.axes[i, k].get_xaxis().set_major_formatter(ticker.ScalarFormatter())
                                            # pairgrid.axes[i, k].set_xticklabels([0, 10, 100, 1000])

                                        if k == kappa_idx:
                                            # pairgrid.axes[i, k].set_ylim(0, 1000)
                                            pairgrid.axes[i, k].set_yscale("log")
                                            # pairgrid.axes[i, k].set_yticks([0, 10, 100, 1000])

                                    if i == f_idx:
                                        pairgrid.axes[i, k].set_xlim(0, 1)
                                        pairgrid.axes[i, k].set_xticks([0, 0.5, 1])

                                    if k == f_idx:
                                        pairgrid.axes[i, k].set_ylim(0, 1)
                                        pairgrid.axes[i, k].set_yticks([0, 0.5, 1])

                                    # if i == L_idx:
                                    #     pairgrid.axes[i, k].set_xlim(0, 20)
                                    #     pairgrid.axes[i, k].set_xticks(
                                    #         [0, 10, 20])

                                    # if k == L_idx:
                                    #     pairgrid.axes[i, k].set_ylim(0, 20)
                                    #     pairgrid.axes[i, k].set_yticks(
                                    #         [0, 10, 20])
        else:
            N = np.shape(data_frame)[1]
            for i in range(N):
                for j in range(N):
                    for k in range(N):
                        # KW: checks if axes is NoneType, since corner=True makes unused axes
                        # into NoneTypes
                        if pairgrid.axes[i, k]:
                            if i != k:
                                pairgrid.axes[i, k].spines["right"].set_visible(True)
                                pairgrid.axes[i, k].spines["top"].set_visible(True)
                            else:
                                sns.despine(ax=pairgrid.axes[i, k])
                                if b_idx is not None:
                                    if i == b_idx:
                                        pairgrid.axes[i, k].set_xlim(B_lim)
                                        pairgrid.axes[i, k].set_xticks(B_ticks)

                                    if k == b_idx:
                                        pairgrid.axes[i, k].set_ylim(B_lim)
                                        pairgrid.axes[i, k].set_yticks(B_ticks)

                                if kappa_idx is not None:
                                    if i == kappa_idx:
                                        # pairgrid.axes[i, k].set_xlim(0, 1000)
                                        pairgrid.axes[i, k].set_xscale("log")
                                        # pairgrid.axes[i, k].set_xticks([0, 10, 100, 1000])
                                        # pairgrid.axes[i, k].get_xaxis().get_major_formatter().labelOnlyBase = False
                                        # pairgrid.axes[i, k].get_xaxis().set_major_formatter(ticker.ScalarFormatter())
                                        # pairgrid.axes[i, k].set_xticklabels([0, 10, 100, 1000])

                                    if k == kappa_idx:
                                        # pairgrid.axes[i, k].set_ylim(0, 1000)
                                        pairgrid.axes[i, k].set_yscale("log")
                                        # pairgrid.axes[i, k].set_yticks([0, 10, 100, 1000])

                                if i == f_idx:
                                    pairgrid.axes[i, k].set_xlim(0, 1)
                                    pairgrid.axes[i, k].set_xticks([0, 0.5, 1])

                                if k == f_idx:
                                    pairgrid.axes[i, k].set_ylim(0, 1)
                                    pairgrid.axes[i, k].set_yticks([0, 0.5, 1])

                                if i == L_idx:
                                    pairgrid.axes[i, k].set_xlim(0, 75)
                                    pairgrid.axes[i, k].set_xticks([0, 25, 75])

                                if k == L_idx:
                                    pairgrid.axes[i, k].set_ylim(0, 75)
                                    pairgrid.axes[i, k].set_yticks([0, 25, 75])

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

        plt.savefig(filename, dpi=500, bbox_inches="tight")
