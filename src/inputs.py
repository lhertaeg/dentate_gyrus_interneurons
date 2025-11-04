#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 16:41:31 2025

@author: loreen.hertaeg
"""


# %% load libraries

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# %% functions

def generate_training_input(T_assembly, batch, NE,
                            mean_exc=0.5,
                            assembly_input=1.0,
                            sparsity=0.05,
                            std_exc=0.1,
                            n_assemblies_distinct=1,
                            n_assemblies_total=1,
                            coactivation=1,
                            seed=None):
    """
    Generate excitatory neuron inputs over time with assembly structure.

    Args:
        T_assembly (int): number of time steps per assembly
        batch (int): batch size
        NE (int): number of excitatory neurons
        mean_exc (float): baseline input
        assembly_input (float): strength of assembly drive
        sparsity (float): probability of each neuron receiving assembly input
        std_exc (float): noise standard deviation
        n_assemblies_distinct (int): number of distinct assemblies
        n_assemblies_total (int): number of assemblies shown sequentially
        coactivation (int): number of assemblies shown simultaneously
        seed (int, optional): random seed

    Returns:
        inputs: ndarray of shape [T_assembly*n_assemblies_total, batch, NE]
        mean_inputs: ndarray of shape [T_assembly*n_assemblies_total, batch]
    """
    if seed is not None:
        np.random.seed(seed)
    
    T_total = T_assembly * n_assemblies_total
    inputs = mean_exc * np.ones((T_total, batch, NE))
    
    for b in range(batch):
        
        # Generate distinct assemblies
        assemblies = np.random.binomial(1, sparsity, size=(n_assemblies_distinct, NE))
        
        # Sequence of assemblies
        assembly_sequence = np.random.choice(n_assemblies_distinct, size=n_assemblies_total)
        
        # Fill inputs sequentially
        for i, assembly_idx in enumerate(assembly_sequence):
            t_start = i * T_assembly
            t_end = t_start + T_assembly
            
            # Determine assembly mask
            if coactivation > 1:
                coactive_idx = np.random.choice(n_assemblies_distinct, size=coactivation, replace=True)
                assembly_mask = assemblies[coactive_idx].sum(axis=0)
                assembly_mask = np.clip(assembly_mask, 0, 1)
            else:
                assembly_mask = assemblies[assembly_idx]
            
            # Apply assembly input
            inputs[t_start:t_end, b, :] += assembly_input * assembly_mask
    
    # Add Gaussian noise
    inputs += std_exc * np.random.randn(T_total, batch, NE)
    
    # Mean input over time
    inputs_reshaped = inputs.reshape(n_assemblies_total, T_assembly, batch, NE)
    mean_per_assembly = inputs_reshaped.mean(axis=-1)  # shape: [n_assemblies_total, T_assembly, batch]
    mean_inputs = mean_per_assembly.reshape(T_total, batch)  # shape: [T_total, batch]
    
    return inputs, mean_inputs

