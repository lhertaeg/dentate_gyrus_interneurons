#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 09:03:05 2025

@author: loreen.hertaeg
"""

# %% import libraries

import numpy as np
import os
import pickle

from src.inputs import generate_training_input
from src.model import DentateGyrus
import src.default as default

from src.analysis import load_or_generate, plot_SI_heatmap


# %% stimulate network with coactivated input pattern (one familair, one novel)

def generate_and_save_data():
    
    np.random.seed(default.seed)
    
    # number of distinct assemblies shown in a total
    n_assemblies_distinct = 1
    n_assemblies_total = 3
    
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
    
    # define the assmblies
    threshold = (default.mean_exc + default.assembly_input)//2
    assembly_neurons_1 = np.where(EC_input2E_1[:, 0, :].mean(axis=0) > threshold)[0]
    assembly_neurons_2 = np.where(EC_input2E_2[:, 0, :].mean(axis=0) > threshold)[0]
    
    assembly_code = np.zeros(default.n_E, dtype=int)
    assembly_code[assembly_neurons_1] = 1
    assembly_code[assembly_neurons_2] = 2

    # define connection strengths to be tested
    num_pts_tested = 9
    strengths_PVIIs_to_GCs = np.linspace(1, 2, num_pts_tested)
    max_ws = np.linspace(1, 2, num_pts_tested)
    #strengths_PVIIs_to_PVIOs = np.linspace(0.5, 1.5, num_pts_tested)
    
    n_seeds = 10
    SIs_all = np.zeros((num_pts_tested, num_pts_tested, n_seeds), dtype=float)
    
    # compute supression index for all combinations
    for i, w in enumerate(strengths_PVIIs_to_GCs):
        #for j, v in enumerate(strengths_PVIIs_to_PVIOs):
        for j, w_max in enumerate(max_ws):
            for seed_idx in range(n_seeds):
                
                # deterministic reseed per repeat
                np.random.seed(default.seed + seed_idx)
            
                print(f"w={w:.3f}, w_max={w_max:.3f}, seed={seed_idx+1}/{n_seeds}  ", end="\r")
                
                total_weights = default.total_weights.copy()
                #total_weights[2,0] = w
                total_weights[0,2] = w
                #total_weights[1,2] = v
                
                default.w_max = w_max

                # define model
                model = DentateGyrus(default.time_const, total_weights, n_E=default.n_E)
            
                # run simulation
                T_total = default.T_assembly * n_assemblies_total
    
                results_familiar = model.simulate(EC_input2E=EC_input2E_1, EC_input2PVFF=EC_input2PVFF_1, T=T_total, 
                                   dt=default.dt, eta=default.eta, w_max=default.w_max, plasticity=True)
                
                results_coactivated = model.simulate(EC_input2E=EC_input2E_coactivated, EC_input2PVFF=EC_input2PVFF_coactivated, 
                                                     T=T_total, dt=default.dt, eta=default.eta, w_max=default.w_max, plasticity=False)
    
                GC_rates = results_coactivated["rates"]["GCs"]
                GC_rates_last = GC_rates[-50:, 0, :]
                mean_1 = GC_rates_last[:, assembly_code == 1].mean()
                mean_2 = GC_rates_last[:, assembly_code == 2].mean()
                SIs_all[i,j,seed_idx] = (mean_1 - mean_2) / (mean_1 + mean_2 + 1e-8)

    # save and return data 
    data = {"SIs": SIs_all.mean(axis=2), "EC_input2E_coactivated": EC_input2E_coactivated, 
            "EC_input2PVFF_coactivated": EC_input2PVFF_coactivated, "EC_input2E_1": EC_input2E_1, 
            "EC_input2PVFF_1": EC_input2PVFF_1, "EC_input2E_2": EC_input2E_2, 
            "EC_input2PVFF_2": EC_input2PVFF_2, "strengths_PVIIs_to_GCs": strengths_PVIIs_to_GCs,
            "max_ws":max_ws}
    
    save_path = os.path.join("..", "results", "data_SI_dependency.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(data, f)
        
        print(f"Data saved to {save_path}")
        
    return data


def plot_data(data):
    
    SIs = data["SIs"] 
    max_ws = data["max_ws"]
    strengths_PVIIs_to_GCs = data["strengths_PVIIs_to_GCs"]
    
    plot_SI_heatmap(SIs, annot=True, vmin=0, vmax=1, x_values=max_ws, y_values=strengths_PVIIs_to_GCs,
                    xlabel="maximal weight", ylabel="PVII → GC strength")
    
    
def main(rerun=False):
    
    load_path = os.path.join("..", "results", "data_SI_dependency.pkl")
    data = load_or_generate(load_path, generate_and_save_data, rerun=rerun)

    plot_data(data)
    

if __name__ == "__main__":
    main(rerun=False)
    
    
    