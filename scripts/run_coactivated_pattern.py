#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 13:58:45 2025

@author: loreen.hertaeg
"""

# %% import libraries

import numpy as np
import os
import pickle

from src.inputs import generate_training_input
from src.model import DentateGyrus
import src.default as default

from src.analysis import load_or_generate, plot_suppression_index


# %% stimulate network with coactivated input pattern (one familair, one novel)

def generate_and_save_data():
    
    np.random.seed(default.seed)
    
    # number of distinct assemblies shown in a total
    n_assemblies_distinct = 1
    n_assemblies_total = 5
    
    # generate the input for familar and for novel
    EC_input2E_1, EC_input2PVFF_1 = generate_training_input(default.T_assembly, 1, default.n_E, mean_exc=default.mean_exc, 
                                                            assembly_input=default.assembly_input, 
                                                            sparsity=default.sparsity, std_exc=default.std_exc, 
                                                            n_assemblies_distinct=n_assemblies_distinct, 
                                                            n_assemblies_total=n_assemblies_total,
                                                            coactivation=1, seed=None)
    
    EC_input2E_2, EC_input2PVFF_2 = generate_training_input(default.T_assembly, 1, default.n_E, mean_exc=default.mean_exc, 
                                                            assembly_input=default.assembly_input, 
                                                            sparsity=default.sparsity, std_exc=default.std_exc, 
                                                            n_assemblies_distinct=n_assemblies_distinct, 
                                                            n_assemblies_total=n_assemblies_total,
                                                            coactivation=1, seed=None)
    
    EC_input2E_coactivated = np.maximum(EC_input2E_1, EC_input2E_2)
    EC_input2PVFF_coactivated = np.mean(EC_input2E_coactivated, axis=(1,2))
    
    # define model
    model = DentateGyrus(default.time_const, default.total_weights, n_E=default.n_E)

    
    # run simulation
    T_total = default.T_assembly * n_assemblies_total

    results_familar = model.simulate(EC_input2E=EC_input2E_1, EC_input2PVFF=EC_input2PVFF_1, T=T_total, dt=default.dt, 
                                     eta=default.eta, w_max=default.w_max, plasticity=True)
    
    results_coactivated = model.simulate(EC_input2E=EC_input2E_coactivated, EC_input2PVFF=EC_input2PVFF_coactivated, 
                                         T=T_total, dt=default.dt, eta=default.eta, w_max=default.w_max, plasticity=False)

    # save and return data 
    data = {"results": results_coactivated, "EC_input2E_coactivated": EC_input2E_coactivated, 
            "EC_input2PVFF_coactivated": EC_input2PVFF_coactivated, "EC_input2E_1": EC_input2E_1, 
            "EC_input2PVFF_1": EC_input2PVFF_1, "EC_input2E_2": EC_input2E_2, 
            "EC_input2PVFF_2": EC_input2PVFF_2}
    
    save_path = os.path.join("..", "results", "data_coactivated_pattern.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(data, f)
        
        print(f"Data saved to {save_path}")
        
    return data


def plot_data(data):
    
    rates = data["results"]["rates"]
    EC_input2E_1 = data["EC_input2E_1"]
    EC_input2E_2 = data["EC_input2E_2"]
    
    threshold = (default.mean_exc + default.assembly_input)//2
    assembly_neurons_1 = np.where(EC_input2E_1[:, 0, :].mean(axis=0) > threshold)[0]
    assembly_neurons_2 = np.where(EC_input2E_2[:, 0, :].mean(axis=0) > threshold)[0]
    
    # Code neurons: 0 = not in any assembly, 1 = assembly 1, 2 = assembly 2
    assembly_code = np.zeros(default.n_E, dtype=int)
    assembly_code[assembly_neurons_1] = 1
    assembly_code[assembly_neurons_2] = 2
    
    plot_suppression_index(rates, assembly_code, assembly_1=1, assembly_2=2,
                           figsize=(5,3), time_window=(0,100),
                           title="Suppression Index", xlabel="Time step", ylabel="SI") 
    
    
def main(rerun=False):
    
    load_path = os.path.join("..", "results", "data_coactivated_pattern.pkl")
    data = load_or_generate(load_path, generate_and_save_data, rerun=rerun)

    plot_data(data)
    

if __name__ == "__main__":
    main(rerun=True)
    
    
    