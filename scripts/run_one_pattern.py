#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 13:54:25 2025

@author: loreen.hertaeg
"""

# %% import libraries

import numpy as np
import pickle
import os

from src.inputs import generate_training_input
from src.model import DentateGyrus
import src.default as default

from src.analysis import load_or_generate, plot_population_rates, plot_input_barcode_lines, plot_weight_histograms


# %% stimulate network with one input pattern

def generate_and_save_data():
    
    np.random.seed(default.seed)
    
    # number of distinct assemblies shown in a total
    n_assemblies_distinct = 1
    n_assemblies_total = 4
    
    # generate the input
    EC_input2E, EC_input2PVFF = generate_training_input(default.T_assembly, 1, default.n_E, mean_exc=default.mean_exc, 
                                                        assembly_input=default.assembly_input, 
                                                        sparsity=default.sparsity, std_exc=default.std_exc, 
                                                        n_assemblies_distinct=n_assemblies_distinct, 
                                                        n_assemblies_total=n_assemblies_total,
                                                        coactivation=1, seed=None)
    
    # define model
    model = DentateGyrus(default.time_const, default.total_weights, n_E=default.n_E)
    
    # run simulation
    T_total = default.T_assembly * n_assemblies_total

    results = model.simulate(EC_input2E=EC_input2E, EC_input2PVFF=EC_input2PVFF, T=T_total, dt=default.dt, 
                             eta=default.eta, w_max=default.w_max, plasticity=True)

    # save and return data    
    data = {"results": results, "EC_input2E": EC_input2E, "EC_input2PVFF": EC_input2PVFF, "model": model}
    
    save_path = os.path.join("..", "results", "data_one_pattern.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(data, f)
        
        print(f"Data saved to {save_path}")

    return data


def plot_data(data):
    
    rates = data["results"]["rates"]
    weights = data["results"]["weights"]
    EC_input2E = data["EC_input2E"]
    model = data["model"]
    
    threshold = (default.mean_exc + default.assembly_input)/2
#    plot_input_barcode_lines(EC_input2E[:,0,:], threshold, title=None, figsize=(4, 1), show_xaxis=False)
    
    # plot_scatter_weights(weights, assembly_mask=EC_input2E[:, 0, :].mean(axis=0) > threshold, aggregate='Avg')
    assembly_mask = EC_input2E[:, 0, :].mean(axis=0) > threshold
    plot_weight_histograms(weights, model, assembly_mask, spike_value=1e-6, n_bins=30)

    plot_population_rates(rates, mode="mean", figsize=(5,4))
    plot_population_rates(rates, mode="all", populations=["GCs", "PVIIs"], figsize=(5, 2)) # mode = {"mean", "all"}
    
    
def main(rerun=False):
    
    load_path = os.path.join("..", "results", "data_one_pattern.pkl")
    data = load_or_generate(load_path, generate_and_save_data, rerun=rerun)
    
    plot_data(data)


if __name__ == "__main__":
    main(rerun=True)
    