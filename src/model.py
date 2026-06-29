# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 15:11:22 2025

@author: loreen.hertaeg
"""

# %% load libraries

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# %% functions

class DentateGyrus(nn.Module):
    """
    Naming convention: w_X_Y (and mask_X_Y) means Y -> X.
    Weight/mask shape: (n_presyn, n_postsyn) == (n_Y, n_X)
    """
    def __init__(self, time_const, total_weights, n_E=1000, n_PV_FF=1, n_PV_FB=100, c_E_PV_FB=50., 
                 alternative_flag = 0, device='cpu'):
        super().__init__()
        self.device = device
        
        # Cell counts
        self.n_E, self.n_PV_FF, self.n_PV_FB = n_E, n_PV_FF, n_PV_FB
        
        # Time constants
        self.tau_E = time_const[0]
        self.tau_PV_FF = time_const[1]
        self.tau_PV_FB = time_const[2]
        
        # number of connections
        self.c_EE = 2.0               # E --> E
        self.c_PV_FB_E = 50.0         # E --> PV_FB
        self.c_E_PV_FB = c_E_PV_FB    # PV_FB --> E
        self.c_E_PV_FF = n_PV_FF      # PV_FF --> E
        
        # Connectivity masks (sparse, shapes follow presyn x postsyn = n_Y x n_X)
        def sparse_mask(n_pre, n_post, p):
            return (torch.rand(n_pre, n_post, device=device) < p).float()
        
        # --- masks (mask_X_Y means Y -> X ---
        self.mask_EE = sparse_mask(n_E, n_E, self.c_EE / n_E)                
        self.mask_PV_FB_E = sparse_mask(n_E, n_PV_FB, self.c_PV_FB_E / n_E)
        self.mask_E_PV_FB = sparse_mask(n_PV_FB, n_E, self.c_E_PV_FB / n_PV_FB)
        self.mask_E_PV_FF = sparse_mask(n_PV_FF, n_E, self.c_E_PV_FF / n_PV_FF)                      
        self.mask_PV_FF_PV_FB = torch.ones(n_PV_FB, n_PV_FF, device=device)
        self.mask_PV_FB_PV_FB = torch.ones(n_PV_FB, n_PV_FB, device=device)
        self.mask_PV_FB_PV_FF = torch.ones(n_PV_FF, n_PV_FB, device=device)
    
        # --- weights: w_X_Y means Y -> X  ---
        w_EE = torch.rand(n_E, n_E, device=device) * self.mask_EE * 2 * total_weights[0,0] / self.c_EE                  
        self.w_PV_FB_E = nn.Parameter(torch.rand(n_E, n_PV_FB, device=device) * self.mask_PV_FB_E) * 2 * total_weights[2,0] / self.c_PV_FB_E        
        self.w_E_PV_FB = nn.Parameter(torch.rand(n_PV_FB, n_E, device=device) * self.mask_E_PV_FB) * 2 * total_weights[0,2] / self.c_E_PV_FB        
        w_E_PV_FF = torch.rand(n_PV_FF, n_E, device=device) * self.mask_E_PV_FF * 2 * total_weights[0,1] / self.c_E_PV_FF        
        w_PV_FF_PV_FB = torch.rand(n_PV_FB, n_PV_FF, device=device) * self.mask_PV_FF_PV_FB * 2 * total_weights[1,2] / n_PV_FB  
        w_PV_FB_PV_FB = torch.rand(n_PV_FB, n_PV_FB, device=device) * self.mask_PV_FB_PV_FB * 2 * total_weights[2,2] / n_PV_FB 
        w_PV_FB_PV_FF = torch.rand(n_PV_FF, n_PV_FB, device=device) * self.mask_PV_FB_PV_FF * 2 * total_weights[2,1] / n_PV_FF 
    
        self.register_buffer("w_EE", w_EE)
        self.register_buffer("w_E_PV_FF", w_E_PV_FF)
        self.register_buffer("w_PV_FF_PV_FB", w_PV_FF_PV_FB)
        self.register_buffer("w_PV_FB_PV_FB", w_PV_FB_PV_FB)
        self.register_buffer("w_PV_FB_PV_FF", w_PV_FB_PV_FF)
    
    def step(self, h_E, h_PV_FF, h_PV_FB, EC_E, EC_PV_FF, dt=1.0, plasticity=False, eta=0.01, w_max = 1.0, anti_hebb=True,
             alternative_flag=0):
        """
        Single Euler time step update
        h_* : internal state (batch, n_neurons)
        EC_input : (batch, n_neurons_post)
        Convention: w_X_Y means Y -> X. Compute inputs as r_Y @ w_X_Y
        """
        # Activities
        r_E = F.relu(h_E)
        r_PV_FF = F.relu(h_PV_FF)
        r_PV_FB = F.relu(h_PV_FB)
        
        # Inputs 
        I_E = (r_E @ self.w_EE) - (r_PV_FF @ self.w_E_PV_FF) - (r_PV_FB @ self.w_E_PV_FB) + EC_E
        I_PV_FF = EC_PV_FF - (r_PV_FB @ self.w_PV_FF_PV_FB)
        I_PV_FB = (r_E @ self.w_PV_FB_E) - (r_PV_FF @ self.w_PV_FB_PV_FF) # !!!!!!!!!!!!!!!!!!
        if alternative_flag==1:
            I_PV_FB += EC_PV_FF - r_PV_FB @ self.w_PV_FB_PV_FB
        
        # Euler integration
        dh_E = (-h_E + I_E) / self.tau_E
        dh_PV_FF = (-h_PV_FF + I_PV_FF) / self.tau_PV_FF
        dh_PV_FB = (-h_PV_FB + I_PV_FB) / self.tau_PV_FB
        
        h_E = h_E + dt * dh_E
        h_PV_FF = h_PV_FF + dt * dh_PV_FF
        h_PV_FB = h_PV_FB + dt * dh_PV_FB
        
        if plasticity:
            # Hebbian updates
            delta_w = eta * (r_E.T @ r_PV_FB)   
            self.w_PV_FB_E.data += delta_w * self.mask_PV_FB_E
            self.w_PV_FB_E.data.clamp_(min=0.0)
        
            # Anti-Hebbian updates
            delta_w_fb = eta * (r_PV_FB.T @ r_E)
            if anti_hebb:
                delta_w_fb = -delta_w_fb
            self.w_E_PV_FB.data += delta_w_fb * self.mask_E_PV_FB
            self.w_E_PV_FB.data.clamp_(min=0.0)
        
            #  Weight normalization
            sum_w = self.w_E_PV_FB.data.sum(dim=0, keepdim=True)
            self.w_E_PV_FB.data = self.w_E_PV_FB.data / (sum_w + 1e-12) * w_max
        
            sum_w = self.w_PV_FB_E.data.sum(dim=0, keepdim=True)
            self.w_PV_FB_E.data = self.w_PV_FB_E.data / (sum_w + 1e-12) * w_max
        
        return h_E, h_PV_FF, h_PV_FB
    
    def simulate(self, EC_input2E, EC_input2PVFF, T=500, dt=1.0, eta=0.01, w_max = 1.0, plasticity=False, 
                 alternative_flag=0):
        """
        Simulate network over time
        EC_input2E: [T, batch, n_E] external input over time for E neurons
        EC_input2PVFF: [T, batch, n_PV_FF] external input over time for PV-FF neurons
        Returns: trajectories of rates (T, batch, ...)
        """
        batch = EC_input2E.shape[1]
        
        # Initial states
        h_E = torch.zeros(batch, self.n_E, device=self.device)
        h_PV_FF = torch.zeros(batch, self.n_PV_FF, device=self.device)
        h_PV_FB = torch.zeros(batch, self.n_PV_FB, device=self.device)
        
        rE_hist, rFF_hist, rFB_hist = [], [], []
        w_E_PV_FB_hist, w_PV_FB_E_hist = [], []
        
        if plasticity:
            w_E_PV_FB_hist.append(self.w_E_PV_FB.detach().cpu().clone())
            w_PV_FB_E_hist.append(self.w_PV_FB_E.detach().cpu().clone())
        
        for t in range(T):
            EC_E_t = torch.tensor(EC_input2E[t], dtype=torch.float32, device=self.device)
            EC_PV_t = torch.tensor(EC_input2PVFF[t], dtype=torch.float32, device=self.device)
            h_E, h_PV_FF, h_PV_FB = self.step(h_E, h_PV_FF, h_PV_FB, EC_E_t, EC_PV_t, dt=dt, eta=eta, w_max=w_max, 
                                              plasticity=plasticity, alternative_flag=alternative_flag)
            
            rE_hist.append(F.relu(h_E).detach().cpu())
            rFF_hist.append(F.relu(h_PV_FF).detach().cpu())
            rFB_hist.append(F.relu(h_PV_FB).detach().cpu())
            
            # Save weights if plastic
            if plasticity:
                w_E_PV_FB_hist.append(self.w_E_PV_FB.detach().cpu().clone())
                w_PV_FB_E_hist.append(self.w_PV_FB_E.detach().cpu().clone())
        
        # Stack results
        results = {
            "rates": {
                "GCs": torch.stack(rE_hist),
                "PVIOs": torch.stack(rFF_hist),
                "PVIIs": torch.stack(rFB_hist),
            }
        }
        
        if plasticity:
            results["weights"] = {
                "E_PV_FB": torch.stack(w_E_PV_FB_hist),
                "PV_FB_E": torch.stack(w_PV_FB_E_hist),
            }
            
        return results

    
