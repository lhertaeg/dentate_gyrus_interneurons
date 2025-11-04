#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 13:42:08 2025

@author: loreen.hertaeg
"""

# %% load libraries

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import os
import pickle

# %% universal parameters

color_GCs = '#646464'
color_PVIIs = '#C80000'
color_PVIOs = '#001BE0'

# %% functions

def load_or_generate(load_path, generate_fn, rerun=False):
    """
    Load data from pickle if it exists, otherwise call generate_fn() to create it.

    Parameters
    ----------
    load_path : str
        Path to the pickle file.
    generate_fn : callable
        Function that returns the data.
    rerun : bool
        If True, ignore the pickle and regenerate.

    Returns
    -------
    data : any
        Loaded or generated data.
    """
    if not os.path.exists(load_path) or rerun:
        data = generate_fn()
    else:
        with open(load_path, "rb") as f:
            data = pickle.load(f)
    
    return data



def plot_population_rates(rates, figsize=(15,4), mode="mean", max_ticks=3, populations=None, 
                          T_assembly=None, n_assemblies_total=None, fs=5, lw=1, axes=None):
    """
    Plot firing rates over time for each population with clean formatting.

    Args:
        rates: dict with keys like "GCs", "PVIOs", "PVIIs",
               each a tensor/array of shape [T, batch, neurons].
        figsize: tuple, figure size.
        mode: "mean" -> average across batch & neurons,
              "all"  -> plot each neuron separately (batch 0).
        max_ticks: maximum number of ticks per axis.
    """

    def to_numpy(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else x

    if populations is None:
        pops = [pop for pop in ["GCs", "PVIOs", "PVIIs"] if pop in rates]
    else:
        pops = [pop for pop in populations if pop in rates]
    colors = {"GCs": color_GCs, "PVIOs": color_PVIOs, "PVIIs": color_PVIIs}
    alphas = {"GCs": 0.2, "PVIOs": 1, "PVIIs": 0.2}

    # ========== MODE: MEAN ==========
    if mode == "mean":
        if axes is None:
            fig, axes = plt.subplots(1, 1, figsize=figsize)
        
        
        if T_assembly is not None and n_assemblies_total is not None:
            for i in range(1, n_assemblies_total + 1):
                x = i * T_assembly
                axes.axvline(x=x, color='gray', linestyle='--', lw=lw)

        for pop in pops:
            data = to_numpy(rates[pop])  # [T, batch, neurons]
            y = data.mean(axis=(1, 2))
            axes.plot(y, label=pop, color=colors[pop], lw=lw)

        axes.set_xlabel("Time steps", fontsize=fs)
        axes.set_ylabel("Firing rate (a.u.)", fontsize=fs)
        axes.set_title("Population firing rates", pad=10, fontsize=fs)
        axes.legend(frameon=False, fontsize=fs, handlelength=1)

        # Keep only bottom x-axis and left y-axis
        
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)

        # Limit tick count
        axes.xaxis.set_major_locator(plt.MaxNLocator(max_ticks))
        axes.yaxis.set_major_locator(plt.MaxNLocator(max_ticks))
        
        axes.set_xlim(left=0)
        axes.set_ylim(bottom=0)
        
        axes.tick_params(size=2.0) 
        axes.tick_params(axis='both', labelsize=fs)
        sns.despine(ax=axes)


    # ========== MODE: ALL ==========
    elif mode == "all":
        
        if axes is None:
            fig, axes = plt.subplots(1, len(pops), figsize=figsize, sharex=True)
            if len(pops) == 1:
                axes = [axes]  # make iterable if only one subplot

        for ax, pop in zip(axes, pops):
            
            if T_assembly is not None and n_assemblies_total is not None:
                for i in range(1, n_assemblies_total + 1):
                    x = i * T_assembly
                    ax.axvline(x=x, color='gray', linestyle='--', lw=1)
            
            data = to_numpy(rates[pop])  # [T, batch, neurons]
            for i in range(data.shape[2]):
                ax.plot(data[:,0,i], color=colors[pop], alpha=alphas[pop], lw=0.8)
            #ax.plot(data[:,0,:].mean(axis=1), color=colors[pop], lw=2.5, ls='-.') # mean
            ax.set_title(f"{pop} neurons", pad=10, fontsize=fs)

            # Style: only bottom + left axes visible
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)

            ax.xaxis.set_major_locator(plt.MaxNLocator(max_ticks))
            ax.yaxis.set_major_locator(plt.MaxNLocator(max_ticks))
            ax.set_xlabel("Time steps", fontsize=fs)
            
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            
            ax.tick_params(size=2.0) 
            ax.tick_params(axis='both', labelsize=fs)
            sns.despine(ax=ax)

        axes[0].set_ylabel("Firing rate (a.u.)", fontsize=fs)
        plt.tight_layout()

    else:
        raise ValueError("mode must be 'mean' or 'all'")


def plot_input_barcode_lines(matrix, threshold, title="Input barcode", figsize=(8, 1.5), linewidth=2.0, 
                             linecolor="black", background="white", show_xaxis=False, pad=0.2):
    """
    Plot a barcode using vertical lines (vector graphics).
    Each neuron with mean input > threshold is drawn as a vertical line.

    Args:
        matrix: np.ndarray or torch tensor, shape [T, neurons] or [T, pre, post]
        threshold: scalar or percentile threshold for mean input
        figsize: (width, height)
        linewidth: thickness of stripes (in points)
        linecolor: color of active stripes
        background: background color
        show_xaxis: whether to display x-axis ticks/label
        pad: padding for tight_layout
    Returns:
        mean_input: numpy array of mean input per neuron
    """
    # convert torch tensors if needed
    if hasattr(matrix, "detach"):
        matrix = matrix.detach().cpu().numpy()

    # collapse 3D -> 2D if necessary
    if matrix.ndim == 3:
        T, pre, post = matrix.shape
        matrix = matrix.reshape(T, pre * post)
    elif matrix.ndim != 2:
        raise ValueError("matrix must be 2D [T, neurons] or 3D [T, pre, post]")

    mean_input = np.mean(matrix, axis=0)
    active = np.where(mean_input > threshold)[0]

    # --- plotting ---
    fig, ax = plt.subplots(figsize=figsize, facecolor=background)
    ax.set_facecolor(background)

    # draw one vertical line per active neuron
    ax.vlines(active, ymin=0, ymax=1, color=linecolor, linewidth=linewidth)

    # limits and cleanup
    ax.set_xlim(0, len(mean_input))
    ax.set_ylim(0, 1)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_yticks([])

    if show_xaxis:
        ax.set_xlabel("Neuron index", labelpad=10)
        ax.xaxis.set_ticks_position("bottom")
    else:
        ax.set_xticks([])
        ax.set_xlabel("")

    ax.set_title(title, pad=8)
    plt.tight_layout(pad=pad)
    plt.show()

    return #mean_input


def plot_suppression_index(rates, assembly_code, assembly_1=1, assembly_2=2,
                           time_window=None, figsize=(8,4), title=" ", 
                           xlabel="Time steps", ylabel="Suppression Index (SI)", fs=5, lw=1, ax=None):
    """
    Plot the suppression index (SI) over time for two assemblies.

    Args:
        rates: dict with keys like "E", each [T, batch, neurons]
        assembly_code: array of neuron labels (same length as number of neurons)
        assembly_1: label of first assembly
        assembly_2: label of second assembly
        time_window: tuple (start, end) to plot only part of the SI
        figsize: figure size
        title: plot title
        xlabel: x-axis label
        ylabel: y-axis label
    """
    # extract mean rates for the two assemblies
    mean_1 = rates["GCs"][:, 0, assembly_code == assembly_1].mean(axis=1)
    mean_2 = rates["GCs"][:, 0, assembly_code == assembly_2].mean(axis=1)

    # compute suppression index
    SI = (mean_1 - mean_2) / (mean_1 + mean_2 + 1e-8)

    # restrict to time window if given
    if time_window is not None:
        start, end = time_window
        SI = SI[start:end]
        x_vals = np.arange(end - start)
    else:
        x_vals = np.arange(len(SI))

    # plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        
    ax.plot(x_vals, SI, color=color_GCs, lw=lw)
    ax.set_ylim([-1, 1.1])
    ax.set_xlim([0, x_vals[-1]])  # start x at zero
    ax.set_xlabel(xlabel, fontsize=fs)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fs)
    ax.set_title(title, fontsize=fs)

    # remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # only show ticks on bottom and left
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))

    # horizontal reference line
    ax.axhline(0, color='k', lw=1, ls='--', alpha=0.7)

    ax.tick_params(size=2.0) 
    ax.tick_params(axis='both', labelsize=fs)
    sns.despine(ax=ax)


def plot_scatter_weights(weights, assembly_mask, aggregate='Avg', fs=5, s=2, axes=None):
    """
    Two-panel scatter plot with mean overlay:
    Panel 1: E->PV weights (weights['PV_FB_E'])
    Panel 2: PV->E weights (weights['E_PV_FB'])
    
    Assumes last time step only, shape [time, post, pre].
    """
    assembly_mask = np.asarray(assembly_mask, dtype=bool)

    # --- Panel 1: E -> PV ---
    w_PV_E = weights["PV_FB_E"][-1, :, :]  # shape [n_E, n_PV]
    if aggregate == "Avg":
        w_x1 = w_PV_E[~assembly_mask, :].mean(axis=0)  # non-assembly E → PV
        w_y1 = w_PV_E[assembly_mask, :].mean(axis=0)    # assembly E → PV
    elif aggregate == "Sum":
        w_x1 = w_PV_E[~assembly_mask, :].sum(axis=0)
        w_y1 = w_PV_E[assembly_mask, :].sum(axis=0)

    # --- Panel 2: PV -> E ---
    w_E_PV = weights["E_PV_FB"][-1, :, :]  # shape [n_PV, n_E]
    if aggregate == "Avg":
        w_x2 = w_E_PV[:, ~assembly_mask].mean(axis=1)  # PV → non-assembly E
        w_y2 = w_E_PV[:, assembly_mask].mean(axis=1)   # PV → assembly E
    elif aggregate == "Sum":
        w_x2 = w_E_PV[:, ~assembly_mask].sum(axis=1)  # PV → non-assembly E
        w_y2 = w_E_PV[:, assembly_mask].sum(axis=1)   # PV → assembly E

    # --- Plot ---
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(10,5))

    # Panel 1: E->PV
    ax = axes[0]
    ax.scatter(w_x1, w_y1, color=color_GCs, alpha=0.5, clip_on=False, s=s)
    # ax.scatter(w_x1.mean(), w_y1.mean(), color=color_GCs, s=100, marker='X')  # mean
    ax.plot([0,0], [0,0], 'k--', lw=1)  # placeholder for diagonal line
    ax.plot([0,1], [0,1], 'k--', lw=1)  # diagonal (optional)
    # x and y range independently
    ax.set_xlim(0.9 * w_x1.min(), 1.1 * w_x1.max())
    ax.set_ylim(0.9 * w_y1.min(), 1.1 * w_y1.max())
    ax.set_xlabel(rf"{aggregate} weight from GC$_\mathrm{{non-A}}$", fontsize=fs)
    ax.set_ylabel(rf"{aggregate} weight from GC$_\mathrm{{A}}$", fontsize=fs)
    ax.set_title(r"GC → PV$_\mathrm{II}$ (Hebbian)", fontsize=fs)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=2))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=2))
    
    ax.tick_params(size=2.0) 
    ax.tick_params(axis='both', labelsize=fs)
    sns.despine(ax=ax)

    # Panel 2: PV->E
    ax = axes[1]
    ax.scatter(w_x2, w_y2, color=color_PVIIs, alpha=0.5, clip_on=False, s=s)
    # ax.scatter(w_x2.mean(), w_y2.mean(), color=color_PVIIs, s=100, marker='X')  # mean
    ax.plot([0,1], [0,1], 'k--', lw=1)  # diagonal
    # x and y range independently
    ax.set_xlim(0.9 * w_x2.min(), 1.1 * w_x2.max())
    ax.set_ylim(0.9 * w_y2.min(), 1.1 * w_y2.max())
    ax.set_xlabel(rf"{aggregate} weight to GC$_\mathrm{{non-A}}$", fontsize=fs)
    ax.set_ylabel(rf"{aggregate} weight to GC$_\mathrm{{A}}$", fontsize=fs)
    ax.set_title(r"PV$_\mathrm{II}$ → GC (Anti-Hebb)", fontsize=fs)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=2))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=2))
    
    ax.tick_params(size=2.0) 
    ax.tick_params(axis='both', labelsize=fs)
    sns.despine(ax=ax)

    plt.show()


# def plot_weight_scatter(weights, assembly_mask, weight_pairs=None,
#                         figsize=(10,4), colors=("#F3A712", "#461353"),
#                         alpha=0.6, s=5):
    
#     assembly_mask = np.asarray(assembly_mask, dtype=bool)

#     if weight_pairs is None:
#         weight_pairs = list(weights.keys())[:2]

#     n_panels = len(weight_pairs)
#     fig, axes = plt.subplots(1, n_panels, figsize=figsize)
#     if n_panels == 1:
#         axes = [axes]

#     for ax, key in zip(axes, weight_pairs):
#         if key not in weights:
#             continue

#         w = weights[key]
#         if hasattr(w, "detach"):
#             w = w.detach().cpu().numpy()

#         w0 = w[0,:,:]   # [presyn, postsyn]
#         wT = w[-1,:,:]

#         n_pre, n_post = w0.shape

#         # E→PV_FB: color by postsynaptic E neurons
#         if key.startswith("E_"):
#             w0_active = w0[:, assembly_mask].flatten()
#             wT_active = wT[:, assembly_mask].flatten()
#             w0_inactive = w0[:, ~assembly_mask].flatten()
#             wT_inactive = wT[:, ~assembly_mask].flatten()

#         # PV_FB→E: color by presynaptic E neurons
#         elif key.endswith("_E"):
#             w0_active = w0[assembly_mask, :].flatten()
#             wT_active = wT[assembly_mask, :].flatten()
#             w0_inactive = w0[~assembly_mask, :].flatten()
#             wT_inactive = wT[~assembly_mask, :].flatten()
            
#         else:
#             w0_active = np.array([])
#             wT_active = np.array([])
#             w0_inactive = w0.flatten()
#             wT_inactive = wT.flatten()

#         # scatter plot
#         ax.scatter(w0_inactive, wT_inactive, color=colors[0], alpha=alpha, s=s, label="E inact")
#         if len(w0_active) > 0:
#             ax.scatter(w0_active, wT_active, color=colors[1], alpha=alpha, s=s, label="E act")

#         # x-axis covers initial weights
#         border = 0.1
#         x_min = min(np.min(w0_active), np.min(w0_inactive))
#         x_max = max(np.max(w0_active), np.max(w0_inactive))
#         x_range = x_max - x_min
#         ax.set_xlim(right=x_max + border*x_range)
        
#         # y-axis covers final weights
#         y_min = min(np.min(wT_active), np.min(wT_inactive))
#         y_max = max(np.max(wT_active), np.max(wT_inactive))
#         y_range = y_max - y_min
#         ax.set_ylim(top=y_max + border*y_range)
        
#         # diagonal only spans the overlap of x and y axes
#         diag_min = max(x_min, y_min)
#         diag_max = min(x_max, y_max)
#         ax.plot([diag_min, diag_max], [diag_min, diag_max], 'k--', lw=1)

#         ax.set_xlabel("Initial weight")
#         ax.set_ylabel("Final weight")
#         ax.set_title(f"{key} weights")
#         if n_pre<n_post:
#             ax.legend(frameon=False, labelspacing=0.2, handlelength=1, handletextpad=0.2, fontsize=10)
#         ax.spines["top"].set_visible(False)
#         ax.spines["right"].set_visible(False)
#         # ax.set_xlim(right=0)
#         # ax.set_ylim(bottom=0)

#     plt.tight_layout()
#     plt.show()


# def plot_weight_evolution(weights, populations=None, figsize=(10,6), mode="avg-post", alpha=0.3):
#     """
#     Plot evolution of synaptic weights over time.

#     Args:
#         weights: dict with keys like "E_PV_FB", "PV_FB_E", each of shape [T, presyn, postsyn]
#         populations: list of populations to plot (default all in weights)
#         figsize: figure size
#         mode: "mean" -> plot mean across postsynaptic neurons,
#               "all"  -> plot each connection separately (alpha used for individual lines)
#         alpha: transparency for individual connections in "all" mode
#     """
#     if populations is None:
#         populations = list(weights.keys())

#     colors = {"E_PV_FB": "r", "PV_FB_E": "b"}  # you can add more if needed

#     plt.figure(figsize=figsize)

#     for pop in populations:
#         if pop not in weights:
#             continue
#         w = weights[pop]  # [T, presyn, postsyn]
#         if hasattr(w, "detach"):
#             w = w.detach().cpu().numpy()

#         if mode == "avg-post":
#             mean_w = w.mean(axis=2)  # mean across postsynaptic neurons
#             plt.plot(mean_w, color=colors.get(pop, None), alpha=1.0)
#         elif mode == "avg-pre":
#             mean_w = w.mean(axis=1)  # mean across presynaptic neurons
#             plt.plot(mean_w, color=colors.get(pop, None), alpha=1.0)
#         else:
#             raise ValueError("mode must be 'mean' or 'all'")

#     plt.plot(np.nan, np.nan, color="r", label="PV_FB --> E")
#     plt.plot(np.nan, np.nan, color="b", label="E --> PV_FB")
#     plt.legend()
    
#     plt.xlabel("Time step")
#     plt.ylabel("Weight")
#     plt.title("Synaptic weight evolution over time")
#     plt.tight_layout()
#     plt.show()



# def plot_matrix_over_time(matrix, title="", xlabel="Time step", ylabel="Neuron index",
#                           cmap="viridis", aspect="auto", origin="lower", figsize=(10,6), colorbar_label=None):
#     """
#     Plot a matrix that evolves over time as a heatmap.
    
#     Args:
#         matrix: torch.Tensor or np.ndarray of shape [T, neurons] or [T, presyn, postsyn].
#                 If 3D, will assume [T, presyn, postsyn] and collapse presyn axis into neurons.
#         title: plot title
#         xlabel: label for x-axis
#         ylabel: label for y-axis
#         cmap: colormap
#         aspect: aspect ratio (default "auto")
#         origin: "lower" puts neuron 0 at bottom
#         figsize: figure size
#         colorbar_label: optional label for colorbar
#     """
    
#     if hasattr(matrix, "detach"):  # torch tensor
#         matrix = matrix.detach().cpu().numpy()
    
#     if matrix.ndim == 3:
#         # Flatten presynaptic axis if needed: [T, presyn*postsyn]
#         T, pre, post = matrix.shape
#         matrix = matrix.reshape(T, pre*post).T
#     elif matrix.ndim == 2:
#         matrix = matrix.T
#     else:
#         raise ValueError("Matrix must be 2D [T, neurons] or 3D [T, presyn, postsyn].")
    
#     plt.figure(figsize=figsize)
#     plt.imshow(matrix, aspect=aspect, cmap=cmap, origin=origin)
#     cbar = plt.colorbar()
#     if colorbar_label:
#         cbar.set_label(colorbar_label)
    
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.title(title)
#     plt.tight_layout()
#     plt.show()


