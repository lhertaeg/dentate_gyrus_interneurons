#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 13:59:07 2025

@author: loreen.hertaeg
"""

# %% import libraries

import numpy as np
import os
import pickle

from src.inputs import generate_training_input
from src.model import DentateGyrus
import src.default as default
import matplotlib.pyplot as plt

from src.analysis import plot_suppression_index


# %% Notes

# I think the overlap should have higher activity, not lower activity
# At the moment, we show one pattern and the overlap are the neurons that receive more input
# We could also do it differently: show two pattern (that share some neurons/inputs) alternating --> leanring, then show them together
# but I don't think this would change anything, the shared neurons would still receive less inhibition and hence will not be suppressed

# %% stimulate network with two overlapping inputs

def generate_and_save_data():
    
    np.random.seed(default.seed)
    
    # number of distinct assemblies shown in a total
    n_assemblies_distinct = 1
    n_assemblies_total = 10
    batch_size = 10
    
    sparsity = default.sparsity * 1.5  # two assemblies at the same time
    
    # generate two inputs
    EC_input2E, EC_input2PVFF = generate_training_input(default.T_assembly, batch_size, default.n_E, mean_exc=default.mean_exc, 
                                                        assembly_input=default.assembly_input, 
                                                        sparsity=sparsity, std_exc=default.std_exc, 
                                                        n_assemblies_distinct=n_assemblies_distinct, 
                                                        n_assemblies_total=n_assemblies_total,
                                                        coactivation=1, seed=None)

    thres = (default.assembly_input + default.mean_exc)/2
    mean_over_time = EC_input2E.mean(axis=0)
    
    assembly_neurons = [np.where(mean_over_time[b] > thres)[0] for b in range(batch_size)]
    overlap_neurons = [a[len(a)//3 : 2*len(a)//3] for a in assembly_neurons]
    
    EC_input2E_overlapping = EC_input2E.copy()
    for b in range(batch_size):
        EC_input2E_overlapping[:, b, overlap_neurons[b]] += default.assembly_input

    EC_input2PVFF_overlapping = EC_input2E_overlapping.mean(axis=(1, 2))
    
    # define model
    model = DentateGyrus(default.time_const, default.total_weights, n_E=default.n_E)

    
    # run simulation
    T_total = default.T_assembly * n_assemblies_total
    
    results_overlapping = model.simulate(EC_input2E=EC_input2E_overlapping, EC_input2PVFF=EC_input2PVFF_overlapping, 
                                         T=T_total, dt=default.dt, eta=default.eta, w_max=default.w_max, plasticity=False)

    # save and return data 
    data = {"results": results_overlapping, "EC_input2E_overlapping": EC_input2E_overlapping, 
            "EC_input2PVFF_overlapping": EC_input2PVFF_overlapping, "assembly_neurons": assembly_neurons,
            "overlap_neurons": overlap_neurons}
    
    save_path = os.path.join("..", "results", "data_overlapping_pattern.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(data, f)
        
        print(f"Data saved to {save_path}")
        
    return data


def plot_data(data):
    
    rates = data["results"]["rates"]            # shape: (T, B, n_E)
    assembly_neurons_all = data["assembly_neurons"]   # list of length B
    overlap_neurons_all = data["overlap_neurons"]     # list of length B
    B = len(assembly_neurons_all)
    
    mean_assembly_no_overlap_per_batch = []
    mean_overlap_per_batch = []
    mean_non_assembly_per_batch = []

    for b in range(B):

        assembly = assembly_neurons_all[b]
        overlap = overlap_neurons_all[b]

        non_overlap = np.setdiff1d(assembly, overlap)
        non_assembly = np.setdiff1d(np.arange(default.n_E), assembly)

        # mean across neurons in that category (for this batch)
        mean_no_overlap = rates['E'][-1, b, non_overlap].mean().item()
        mean_overlap = rates['E'][-1, b, overlap].mean().item()
        mean_non_assembly = rates['E'][-1, b, non_assembly].mean().item()

        # Store batch means
        mean_assembly_no_overlap_per_batch.append(mean_no_overlap)
        mean_overlap_per_batch.append(mean_overlap)
        mean_non_assembly_per_batch.append(mean_non_assembly)

    # plot boxplot
    data_to_plot = [
        mean_assembly_no_overlap_per_batch,
        mean_overlap_per_batch,
        mean_non_assembly_per_batch
    ]

    labels = ["Assembly no-overlap", "Overlap", "Non-assembly"]

    plt.figure(figsize=(7, 5))
    plt.boxplot(data_to_plot, labels=labels)
    plt.ylabel("Mean firing rate")
    plt.title("Distribution of mean rates across batches")
    plt.show()
    
    
def main():
    data = generate_and_save_data()
    plot_data(data)


if __name__ == "__main__":
    main()
    