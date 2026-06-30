#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 14:15:37 2025

@author: loreen.hertaeg
"""


# %% load libraries

import numpy as np

# %% define default values

n_E = 1000
T_assembly = 500 
mean_exc = 1
assembly_input = 5 
sparsity = 0.01
std_exc = 0.1

time_const = [10.0, 5.0, 5.0]

# order: E, PV_FF, PV_FB

total_weights = np.array([[0,1,1],[0,0,0.5],[1,0.5,0]])

dt = 1.0
eta = 2e-4
w_max = 2.0 

seed = 186