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
T_assembly = 500 # !!!!!!!!!!!!!!
mean_exc = 1
assembly_input = 4 # !!!!!!!!!!!!!!
sparsity = 0.01
std_exc = 0.1

time_const = [10.0, 5.0, 5.0]

# order: E, PV_FF, PV_FB

total_weights = np.array([[0,1,1.5],[0,0,0.5],[1.5,0.0,0]]) # !!!!!!!!!!!!!!
# total_weights = np.array([[0,1,1.5],[0,0,0.5],[1.5,0,0]]) # increase PV-FB to PV-FF if you want stronger suppression during co-activation
# total_weights = np.array([[0, 1, 2],[0, 0, 1.2],[2, 0, 0]])
# total_weights = np.array([[0.01,1,1],[0,0,1],[1,0,0]])
# total_weights = np.array([[0,0.9546,1.98],[0,0,1.21],[1.98,0,0]])

dt = 1.0
eta = 2e-4 # 2e-4
w_max = 1.5 # 2.0 # 1.5 # !!!!!!!!!!!!!!

seed = 186