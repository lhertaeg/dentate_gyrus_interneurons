#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 08:53:17 2025

@author: loreen.hertaeg
"""

# %% import libraries

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.inputs import generate_training_input
from src.model import DentateGyrus

from src.analysis import plot_matrix_over_time, plot_population_rates, plot_weight_evolution
from src.analysis import plot_input_barcode_lines, plot_weight_scatter, plot_suppression_index

# %% Run network with one assmbly, then re-run and alternate between novel and familiar assembly

do_run = True
if do_run:
    
    n_E = 1000
    batch = 1
    T_assembly = 500
    n_assemblies_distinct = 1 #1
    n_assemblies_total = 5
    
    EC_input2E_1, EC_input2PVFF_1 = generate_training_input(T_assembly, batch, n_E, mean_exc=0.1, assembly_input=4.0, sparsity=0.01,
                                                            std_exc=0.1, n_assemblies_distinct=n_assemblies_distinct, 
                                                            n_assemblies_total=n_assemblies_total,
                                                            coactivation=1, seed=None)
    
    time_const = [10.0, 5.0, 5.0]
    # total_weights = np.array([[0,1,1],[0,0,1],[1,0,0]])
    total_weights = np.array([[0,0.9546,1.98],[0,0,1.21],[1.98,0,0]])
    
    model = DentateGyrus(time_const, total_weights, n_E=n_E)#, c_E_PV_FB=100)
    
    
    # --- Step 3: Run simulation (training) ---
    T_total = T_assembly * n_assemblies_total
    
    results_training = model.simulate(EC_input2E=EC_input2E_1, EC_input2PVFF=EC_input2PVFF_1, T=T_total, dt=1.0, 
                                      eta=2.1e-4, w_max=1.98, plasticity=True)
    
    
    ### create a novel assembly
    n_assemblies_total = 1
    EC_input2E_2, EC_input2PVFF_2 = generate_training_input(T_assembly, batch, n_E, mean_exc=0.1, assembly_input=4.0, sparsity=0.01,
                                                        std_exc=0.1, n_assemblies_distinct=n_assemblies_distinct, 
                                                        n_assemblies_total=n_assemblies_total,
                                                        coactivation=1, seed=None)
    
    ### alternate between assembly 1 and 2
    n_assemblies_total = 10
    assembly_idx = np.random.choice([1,2],p=(0.4,0.6), size=n_assemblies_total)
    assembly_idx[0] = 2
    
    EC_input2E = np.zeros((T_assembly*n_assemblies_total, 1, n_E))
    EC_input2PVFF = np.zeros((T_assembly*n_assemblies_total, 1))
    
    E_sources = {
        1: EC_input2E_1[:T_assembly, 0, :],
        2: EC_input2E_2[:T_assembly, 0, :]
    }
    PVFF_sources = {
        1: EC_input2PVFF_1[:T_assembly, 0],
        2: EC_input2PVFF_2[:T_assembly, 0]
    }
    
    # Fill arrays efficiently
    for i, idx in enumerate(assembly_idx):
        start = i * T_assembly
        end = start + T_assembly
        EC_input2E[start:end, 0, :] = E_sources.get(idx, E_sources[1])
        EC_input2PVFF[start:end, 0] = PVFF_sources.get(idx, PVFF_sources[1])

    
    results_test = model.simulate(EC_input2E=EC_input2E, EC_input2PVFF=EC_input2PVFF, T=T_assembly * n_assemblies_total, 
                                  dt=1.0, eta=1e4, w_max=1., plasticity=True)
    
    rates_test = results_test["rates"] 
    
    plot_matrix_over_time(EC_input2E[:,0,:], title="External input to E neurons", ylabel="E neuron index", colorbar_label="Input strength")
    plot_population_rates(rates_test, mode="all") # mode = {"mean", "all"}


    ### show suppression index
    assembly_neurons_1 = np.where(EC_input2E_1[:, 0, :].mean(axis=0) > 2)[0]
    assembly_neurons_2 = np.where(EC_input2E_2[:, 0, :].mean(axis=0) > 2)[0]

    assembly_code = np.zeros(n_E, dtype=int)
    assembly_code[assembly_neurons_1] = 1
    assembly_code[assembly_neurons_2] = 2

    plot_suppression_index(rates_test, assembly_code, assembly_1=1, assembly_2=2,
                           figsize=(5,3), color="#C95D63", time_window=None,
                           title="Suppression Index", xlabel="Time step", ylabel="SI")
    

# %% show one distinct assembly during training, then test on the the same coactivated with one not seen before

do_run = False

if do_run:

    n_E = 1000
    batch = 1
    T_assembly = 500
    n_assemblies_distinct = 1 #1
    n_assemblies_total = 10
    
    EC_input2E, EC_input2PVFF = generate_training_input(T_assembly, batch, n_E, mean_exc=0.1, assembly_input=4.0, sparsity=0.01,
                                                        std_exc=0.1, n_assemblies_distinct=n_assemblies_distinct, 
                                                        n_assemblies_total=n_assemblies_total,
                                                        coactivation=1, seed=None)
    
    time_const = [10.0, 5.0, 5.0]
    # total_weights = np.array([[0,1,1],[0,0,1],[1,0,0]])
    total_weights = np.array([[0,0.9546,1.98],[0,0,1.21],[1.98,0,0]])
    
    model = DentateGyrus(time_const, total_weights, n_E=n_E)#, c_E_PV_FB=100)
    
    
    # --- Step 3: Run simulation (training) ---
    T_total = T_assembly * n_assemblies_total
    
    results_training = model.simulate(EC_input2E=EC_input2E, EC_input2PVFF=EC_input2PVFF, T=T_total, dt=1.0, eta=2.1e-4, 
                                      w_max=1.98, plasticity=True)
    
    # eta = 2.1e-4
    # w_max = 1.98
    
    rates_training = results_training["rates"] 
    weights_training = results_training["weights"]
    
    # --- Step 4: Inspect training results ---
    # plot_matrix_over_time(EC_input2E[:,0,:], title="External input to E neurons", ylabel="E neuron index", colorbar_label="Input strength")

    # plot_population_rates(rates_training, mode="mean", figsize=(5,4))
    # plot_population_rates(rates_training, mode="all") # mode = {"mean", "all"}
     
    # plot_weight_evolution(weights_training, mode="avg-post") # mode = {"avg-post", "avg-pre"}
    
    
    # --- Step 5: Run network again but show this assembly and another one together ---
    EC_input2E2, EC_input2PVFF2 = generate_training_input(T_assembly, batch, n_E, mean_exc=1, assembly_input=4.0, sparsity=0.01,
                                                          std_exc=0.1, n_assemblies_distinct=n_assemblies_distinct, 
                                                          n_assemblies_total=n_assemblies_total,
                                                          coactivation=1, seed=None)
    
    EC_input2E_coactivated = np.maximum(EC_input2E, EC_input2E2)
    EC_input2PVFF_coactivated = np.mean(EC_input2E_coactivated, axis=(1,2))
    
    results_test = model.simulate(EC_input2E=EC_input2E_coactivated, EC_input2PVFF=EC_input2PVFF_coactivated, 
                                  T=T_total, dt=1.0, eta=1e4, w_max=1., plasticity=False)
    
    rates_test = results_test["rates"] 
    
    # --- Step 6: Inspect test results ---
    # plot_matrix_over_time(EC_input2E_coactivated[:,0,:], title="External input to E neurons", ylabel="E neuron index", 
    #                       colorbar_label="Input strength")

    # plot_population_rates(rates_test, mode="mean", figsize=(5,4))
    # plot_population_rates(rates_test, mode="all") # mode = {"mean", "all"}
    
    
    
    # --- show the E activity of all E neurons but color-coded by the input/assembly
    # Identify neurons in each assembly
    assembly_neurons_1 = np.where(EC_input2E[:, 0, :].mean(axis=0) > 2)[0]
    assembly_neurons_2 = np.where(EC_input2E2[:, 0, :].mean(axis=0) > 2)[0]
    
    # Code neurons: 0 = not in any assembly, 1 = assembly 1, 2 = assembly 2
    assembly_code = np.zeros(n_E, dtype=int)
    assembly_code[assembly_neurons_1] = 1
    assembly_code[assembly_neurons_2] = 2
    
    # # Plot histogram of E neuron activity
    # plt.figure(figsize=(8,6))
    # plt.hist(rates_test["E"][-1,0,assembly_code==0], bins=20, color='k', alpha=0.5, label="Other neurons", density=True)
    # plt.hist(rates_test["E"][-1,0,assembly_code==1], bins=20, color='r', alpha=0.5, label="Assembly 1", density=True)
    # plt.hist(rates_test["E"][-1,0,assembly_code==2], bins=20, color='b', alpha=0.5, label="Assembly 2", density=True)
    # plt.xlabel("E neuron firing rate")
    # plt.ylabel("Number of neurons")
    # plt.title("Neuron activity histogram by assembly")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    
    # compute the suppression index: we simply use (rA-rB)/(rA+rB)
    plot_suppression_index(rates_test, assembly_code, assembly_1=1, assembly_2=2,
                           figsize=(5,3), color="#C95D63", time_window=(0,100),
                           title="Suppression Index", xlabel="Time step", ylabel="SI")


# %% many assemblies shown

do_run = False

if do_run:

    n_E = 1000
    batch = 1
    T_assembly = 500
    n_assemblies_distinct = 20 #1
    n_assemblies_total = 5
    assembly_input=4.0
    mean_exc=1
    
    EC_input2E, EC_input2PVFF = generate_training_input(T_assembly, batch, n_E, mean_exc=mean_exc, assembly_input=assembly_input,
                                                        sparsity=0.01, std_exc=0.1, n_assemblies_distinct=n_assemblies_distinct, 
                                                        n_assemblies_total=n_assemblies_total,
                                                        coactivation=1, seed=None)
    
    time_const = [10.0, 5.0, 5.0]
    #total_weights = np.array([[0.01,1,1],[0,0,1],[1,0,0]])
    total_weights = np.array([[0,1,1.5],[0,0,1],[1.5,0,0]])
    
    model = DentateGyrus(time_const, total_weights, n_E=n_E)
    
    
    # --- Step 3: Run simulation ---
    T_total = T_assembly * n_assemblies_total
    
    results = model.simulate(EC_input2E=EC_input2E, EC_input2PVFF=EC_input2PVFF, T=T_total, dt=1.0, eta=2e-4, 
                             w_max=1.5, plasticity=True)
    
    rates = results["rates"] 
    weights = results["weights"]
    
    # --- Step 4: Inspect results ---
    threshold = (mean_exc + assembly_input)/2
    for n in range(n_assemblies_total):
        plot_input_barcode_lines(EC_input2E[n*T_assembly:(n+1)*T_assembly,0,:], threshold, title=None, figsize=(4, 1), show_xaxis=False)
    
    # plot_matrix_over_time(EC_input2E[:,0,:], title="External input to E neurons", ylabel="E neuron index", colorbar_label="Input strength")

    # plot_population_rates(rates, mode="mean", figsize=(5,4), T_assembly=T_assembly, n_assemblies_total=n_assemblies_total)
    plot_population_rates(rates, mode="all", T_assembly=T_assembly, n_assemblies_total=n_assemblies_total, 
                          figsize=(15,3)) # mode = {"mean", "all"}
     
    # plot_weight_evolution(weights, mode="avg-post") # mode = {"avg-post", "avg-pre"}


# %% one assembly shown

do_run = False

if do_run:

    n_E = 1000
    batch = 1
    T_assembly = 500
    n_assemblies_distinct = 1 #1
    n_assemblies_total = 5
    mean_exc = 1
    assembly_input = 4
    
    EC_input2E, EC_input2PVFF = generate_training_input(T_assembly, batch, n_E, mean_exc=mean_exc, assembly_input=assembly_input, 
                                                        sparsity=0.01, std_exc=0.1, n_assemblies_distinct=n_assemblies_distinct, 
                                                        n_assemblies_total=n_assemblies_total,
                                                        coactivation=1, seed=None)
    
    time_const = [10.0, 5.0, 5.0]
    total_weights = np.array([[0.01,1,1],[0,0,1],[1,0,0]])
    
    model = DentateGyrus(time_const, total_weights, n_E=n_E)
    
    
    # --- Step 3: Run simulation ---
    T_total = T_assembly * n_assemblies_total
    
    results = model.simulate(EC_input2E=EC_input2E, EC_input2PVFF=EC_input2PVFF, T=T_total, dt=1.0, eta=1e-4, 
                             w_max=1., plasticity=True)
    
    rates = results["rates"] 
    weights = results["weights"]
    
    # --- Step 4: Inspect results ---
    threshold = (mean_exc + assembly_input)/2
    plot_input_barcode_lines(EC_input2E[:,0,:], threshold, title=None, figsize=(4, 1), show_xaxis=False)
    
    plot_population_rates(rates, mode="mean", figsize=(5,4))
    plot_population_rates(rates, mode="all", populations=["E", "PV_FB"], figsize=(5, 2)) # mode = {"mean", "all"}
    
    plot_weight_scatter(weights, assembly_mask=EC_input2E[:, 0, :].mean(axis=0) > threshold,
                        weight_pairs=["E_PV_FB", "PV_FB_E"], figsize=(5, 2.5))
    
    
    # plot_matrix_over_time(EC_input2E[:,0,:], title="External input to E neurons", ylabel="E neuron index", colorbar_label="Input strength")
    # plot_weight_evolution(weights, mode="avg-post") # mode = {"avg-post", "avg-pre"}
