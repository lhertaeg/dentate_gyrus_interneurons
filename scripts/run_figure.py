#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 14:01:02 2025

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
import run_one_pattern, run_alternating_pattern, run_coactivated_pattern, run_SI_dependence

from src.analysis import load_or_generate, plot_scatter_weights, plot_population_rates, plot_suppression_index
from src.analysis import plot_SI_heatmap

# %% Universal parameters

fs = 6
inch = 2.54

# %% Define files and paths

figure_name = 'Fig_X.png'
figPath = '../results/final/'

if not os.path.exists(figPath):
    os.mkdir(figPath)


# %% Define figure structure

figsize=(18/inch,14/inch)
fig = plt.figure(figsize=figsize)

G = gridspec.GridSpec(3, 1, figure=fig, hspace=1)
R1 = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=G[0,0], wspace=1, width_ratios=[0.7,0.7,0,1,1]) # middle one is to increase white space
R2 = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=G[1,0], wspace=0.3, width_ratios=[1,1,0.1,1,1]) # middle one is to increase white space
R3 = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=G[2,0], wspace=0.4, width_ratios=[1,1,1,0.05,1]) 

ax_A = fig.add_subplot(R1[0,0])
ax_A.axis('off')
ax_A.text(-0.3, 1.25, 'A', transform=ax_A.transAxes, fontsize=fs+1)

ax_B = fig.add_subplot(R1[0,3:])
ax_B.axis('off')
ax_B.text(-0.25, 1.25, 'B', transform=ax_B.transAxes, fontsize=fs+1)

ax_B1 = fig.add_subplot(R1[0,3])
ax_B2 = fig.add_subplot(R1[0,4])

ax_C = fig.add_subplot(R2[0,:])
ax_C.axis('off')
ax_C.text(-0.05, 1.25, 'C', transform=ax_C.transAxes, fontsize=fs+1)

ax_C1 = fig.add_subplot(R2[0,0])
ax_C2 = fig.add_subplot(R2[0,1])
ax_C3 = fig.add_subplot(R2[0,3])

ax_D = fig.add_subplot(R3[0,:])
ax_D.axis('off')
ax_D.text(-0.05, 1.25, 'D', transform=ax_D.transAxes, fontsize=fs+1)

ax_D2 = fig.add_subplot(R3[0,1])
ax_D3 = fig.add_subplot(R3[0,2])
ax_D4 = fig.add_subplot(R3[0,4])


# %% run one pattern

# load existing data or run simulation
load_path = os.path.join("..", "results", "data_one_pattern.pkl")
data = load_or_generate(load_path, run_one_pattern.generate_and_save_data, rerun=False)

# get results 
rates = data["results"]["rates"]
weights = data["results"]["weights"]
EC_input2E = data["EC_input2E"]

# plot results
threshold = (default.mean_exc + default.assembly_input)/2
plot_scatter_weights(weights, assembly_mask=EC_input2E[:, 0, :].mean(axis=0) > threshold, aggregate='Avg', 
                     fs=fs, axes=[ax_B1, ax_B2])

plot_population_rates(rates, mode="all", populations=["GCs", "PVIIs"], fs=fs, axes=[ax_C1, ax_C2])
plot_population_rates(rates, mode="mean", figsize=(5,4), fs=fs, axes=ax_C3)


# %% run coactivation of familiar and novel pattern

# load existing data or run simulation
load_path = os.path.join("..", "results", "data_coactivated_pattern.pkl")
data = load_or_generate(load_path, run_coactivated_pattern.generate_and_save_data, rerun=False)

# get results
rates = data["results"]["rates"]
EC_input2E_1 = data["EC_input2E_1"]
EC_input2E_2 = data["EC_input2E_2"]

# define assemblies
threshold = (default.mean_exc + default.assembly_input)//2
assembly_neurons_1 = np.where(EC_input2E_1[:, 0, :].mean(axis=0) > threshold)[0]
assembly_neurons_2 = np.where(EC_input2E_2[:, 0, :].mean(axis=0) > threshold)[0]

# Code neurons: 0 = not in any assembly, 1 = assembly 1, 2 = assembly 2
assembly_code = np.zeros(default.n_E, dtype=int)
assembly_code[assembly_neurons_1] = 1
assembly_code[assembly_neurons_2] = 2

# plot supression index
plot_suppression_index(rates, assembly_code, assembly_1=1, assembly_2=2, time_window=(0,100),
                       title="Coactivation of familiar \n& novel assembly", ax=ax_D2) 


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


plot_suppression_index(rates, assembly_code, assembly_1=1, assembly_2=2, figsize=(5,3),
                       title="Alternating between \n2 assemblies", ylabel=None, ax=ax_D3) 


# %% run heatmap for supression index

# load existing data or run simulation
load_path = os.path.join("..", "results", "data_SI_dependency.pkl")
data = load_or_generate(load_path, run_SI_dependence.generate_and_save_data, rerun=False)

# get results
SIs = data["SIs"] 
max_ws = data["max_ws"]
strengths_PVIIs_to_GCs = data["strengths_PVIIs_to_GCs"]

plot_SI_heatmap(SIs, annot=False, vmin=0, vmax=1, x_values=max_ws, y_values=strengths_PVIIs_to_GCs,
                    xlabel="Maximal weight (during learning)", ylabel="Total initial PVII → GC", ax=ax_D4)


# %% save figure

plt.savefig(figPath + figure_name, bbox_inches='tight', transparent=True, dpi=600)
