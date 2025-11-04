#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 13:58:23 2025

@author: loreen.hertaeg
"""

# %% import libraries

import numpy as np
import pickle
import os

from src.inputs import generate_training_input
from src.model import DentateGyrus
import src.default as default

from src.analysis import load_or_generate, plot_population_rates, plot_input_barcode_lines, plot_suppression_index


# %% stimulate network with alternating input pattern

def generate_and_save_data():
    
    np.random.seed(default.seed)
    
    # number of distinct assemblies shown in total
    n_assemblies_distinct = 2
    n_assemblies_total = 2
    n_repeat = 5
    
    # generate the input
    EC_input2E_2, EC_input2PVFF_2 = generate_training_input(default.T_assembly, 1, default.n_E, mean_exc=default.mean_exc, 
                                                            assembly_input=default.assembly_input, 
                                                            sparsity=default.sparsity, std_exc=default.std_exc, 
                                                            n_assemblies_distinct=n_assemblies_distinct, 
                                                            n_assemblies_total=n_assemblies_total,
                                                            coactivation=1, seed=None)
    
    EC_input2E = np.tile(EC_input2E_2, (n_repeat, 1, 1))
    EC_input2PVFF = np.tile(EC_input2PVFF_2, (n_repeat, 1))
    
    # define model
    model = DentateGyrus(default.time_const, default.total_weights, n_E=default.n_E)

    
    # run simulation
    T_total = default.T_assembly * n_assemblies_total * n_repeat

    results = model.simulate(EC_input2E=EC_input2E, EC_input2PVFF=EC_input2PVFF, T=T_total, dt=default.dt, 
                             eta=default.eta, w_max=default.w_max, plasticity=True)

    # save and return data    
    data = {"results": results, "EC_input2E": EC_input2E, "EC_input2PVFF": EC_input2PVFF}
    
    save_path = os.path.join("..", "results", "data_alternating_pattern.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(data, f)
        
        print(f"Data saved to {save_path}")
        
    return data


def plot_data(data):
    
    rates = data["results"]["rates"]
    # weights = data["results"]["weights"]
    EC_input2E = data["EC_input2E"]
    
    # plot_matrix_over_time(EC_input2E[:,0,:], title="External input to E neurons", ylabel="E neuron index", colorbar_label="Input strength")
    
    # define assemblies
    assembly_neurons_1 = np.where(EC_input2E[:default.T_assembly, 0, :].mean(axis=0) > 2)[0]
    assembly_neurons_2 = np.where(EC_input2E[default.T_assembly:2*default.T_assembly, 0, :].mean(axis=0) > 2)[0]
    
    # Code neurons: 0 = not in any assembly, 1 = assembly 1, 2 = assembly 2
    assembly_code = np.zeros(default.n_E, dtype=int)
    assembly_code[assembly_neurons_1] = 1
    assembly_code[assembly_neurons_2] = 2
    
    
    plot_population_rates(rates, mode="all", T_assembly=default.T_assembly, n_assemblies_total=2 * 5, figsize=(15,3)) 
    plot_suppression_index(rates, assembly_code, assembly_1=1, assembly_2=2, figsize=(5,3),
                           title="Suppression Index", xlabel="Time step", ylabel="SI") 
    
def main(rerun=False):
    
    load_path = os.path.join("..", "results", "data_alternating_pattern.pkl")
    data = load_or_generate(load_path, generate_and_save_data, rerun=rerun)

    plot_data(data)
    

if __name__ == "__main__":
    main(rerun=False)
    
    
    