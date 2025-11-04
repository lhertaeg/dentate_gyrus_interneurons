#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 11:15:06 2025

@author: loreen.hertaeg
"""

# %% import

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os.path

import src.default as default
import run_one_pattern, run_alternating_pattern, run_coactivated_pattern

from src.analysis import load_or_generate, plot_scatter_weights, plot_population_rates, plot_suppression_index

# %% Universal parameters

fs = 6
inch = 2.54

# %% Define files and paths

figure_name = 'Fig_Supp_X.png'
figPath = '../results/final/'

if not os.path.exists(figPath):
    os.mkdir(figPath)


# %% Define figure structure

figsize=(18/inch,3/inch)
fig = plt.figure(figsize=figsize)

G = gridspec.GridSpec(1, 3, figure=fig, hspace=0.6)

ax_G1 = fig.add_subplot(G[0,0])
ax_G2 = fig.add_subplot(G[0,1])
ax_G3 = fig.add_subplot(G[0,2])


# %% run alternating pattern

# load existing data or run simulation
load_path = os.path.join("..", "results", "data_alternating_pattern.pkl")
data = load_or_generate(load_path, run_alternating_pattern.generate_and_save_data, rerun=False)

# get results
rates = data["results"]["rates"]
EC_input2E = data["EC_input2E"]

# define assemblies
assembly_neurons_1 = np.where(EC_input2E[:default.T_assembly, 0, :].mean(axis=0) > 2)[0]
assembly_neurons_2 = np.where(EC_input2E[default.T_assembly:2*default.T_assembly, 0, :].mean(axis=0) > 2)[0]

# Code neurons: 0 = not in any assembly, 1 = assembly 1, 2 = assembly 2
assembly_code = np.zeros(default.n_E, dtype=int)
assembly_code[assembly_neurons_1] = 1
assembly_code[assembly_neurons_2] = 2

plot_population_rates(rates, mode="all", T_assembly=default.T_assembly, n_assemblies_total=2 * 5, axes=[ax_G1, ax_G2, ax_G3]) 


# %% save figure

plt.savefig(figPath + figure_name, bbox_inches='tight', transparent=True, dpi=600)


