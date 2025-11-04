#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 09:45:14 2025

@author: loreen.hertaeg
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from src.inputs import generate_training_input

# %% All functions etc.

def save_model_state(model, final_weights, assembly_A_mask, assembly_B_mask, history, save_dir="../results"):
    """
    Save model parameters, buffers, final plastic weights, connectivity masks,
    base weights, and training metadata.
    """
    os.makedirs(save_dir, exist_ok=True)

    # --- 1. Gather model configuration ---
    config = {
        "time_const": [model.tau_E, model.tau_PV_FF, model.tau_PV_FB],
        "n_E": model.n_E,
        "n_PV_FF": model.n_PV_FF,
        "n_PV_FB": model.n_PV_FB,
        "device": model.device,
    }

    # --- 2. Save everything in one .pt file ---
    torch.save({
        # configuration + state
        "config": config,
        "model_state_dict": model.state_dict(),
        "final_weights": final_weights,

        # connectivity structure
        "mask_EE": model.mask_EE,
        "mask_E_PV_FF": model.mask_E_PV_FF,
        "mask_PV_FB_E": model.mask_PV_FB_E,
        "mask_E_PV_FB": model.mask_E_PV_FB,
        "mask_PV_FF_PV_FB": model.mask_PV_FF_PV_FB,

        # base (fixed) weight tensors
        "W_EE_base": model.W_EE_base,
        "W_E_PV_FF_base": model.W_E_PV_FF_base,
        "W_PV_FF_PV_FB_base": model.W_PV_FF_PV_FB_base,

        # metadata
        "assembly_A_mask": assembly_A_mask,
        "assembly_B_mask": assembly_B_mask,
        "history": history,
    }, os.path.join(save_dir, "dentate_model.pt"))

    print(f"Model and metadata saved to {save_dir}/dentate_model.pt")


def load_model_state(path):
    """
    Reload a DentateGyrusDiff model and associated data from a .pt file.
    Returns:
        model (DentateGyrusDiff)
        final_weights (dict)
        assembly_A_mask, assembly_B_mask, history
    """
    checkpoint = torch.load(path, map_location="cpu")

    cfg = checkpoint["config"]
    total_weights = np.array([[0, 1, 1],
                              [0, 0, 1],
                              [1, 0, 0]])  # same as in training

    # Recreate model skeleton
    model = DentateGyrusDiff(
        time_const=cfg["time_const"],
        total_weights=total_weights,
        n_E=cfg["n_E"],
        n_PV_FF=cfg["n_PV_FF"],
        n_PV_FB=cfg["n_PV_FB"],
        device=cfg["device"]
    )

    # Load trained parameters
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    # --- restore masks and base weights exactly as saved ---
    model.mask_EE = checkpoint["mask_EE"]
    model.mask_E_PV_FF = checkpoint["mask_E_PV_FF"]
    model.mask_PV_FB_E = checkpoint["mask_PV_FB_E"]
    model.mask_E_PV_FB = checkpoint["mask_E_PV_FB"]
    model.mask_PV_FF_PV_FB = checkpoint["mask_PV_FF_PV_FB"]

    model.W_EE_base = checkpoint["W_EE_base"]
    model.W_E_PV_FF_base = checkpoint["W_E_PV_FF_base"]
    model.W_PV_FF_PV_FB_base = checkpoint["W_PV_FF_PV_FB_base"]

    # --- other saved components ---
    final_weights = checkpoint.get("final_weights", None)
    assembly_A_mask = checkpoint.get("assembly_A_mask", None)
    assembly_B_mask = checkpoint.get("assembly_B_mask", None)
    history = checkpoint.get("history", None)

    print(f"Model successfully loaded from {path}")
    return model, final_weights, assembly_A_mask, assembly_B_mask, history


# -------------------------
# Utility: positive param via softplus
# -------------------------
def positive(param_raw):
    # softplus to ensure >0
    return F.softplus(param_raw)

# -------------------------
# Differentiable DentateGyrus
# -------------------------
class DentateGyrusDiff(nn.Module):
    def __init__(self, time_const, total_weights, n_E=1000, n_PV_FF=1, n_PV_FB=100, c_E_PV_FB=50., device='cpu'):
        super().__init__()
        self.device = device
        self.n_E, self.n_PV_FF, self.n_PV_FB = n_E, n_PV_FF, n_PV_FB
        self.tau_E, self.tau_PV_FF, self.tau_PV_FB = time_const

        # Connectivity densities
        self.c_EE = 2.0
        self.c_PV_FB_E = 50.0
        self.c_E_PV_FB = c_E_PV_FB
        self.c_E_PV_FF = n_PV_FF

        # Masks
        def sparse_mask(n_pre, n_post, p):
            return (torch.rand(n_pre, n_post, device=device) < p).float()

        self.mask_EE = sparse_mask(n_E, n_E, self.c_EE / n_E)
        self.mask_PV_FB_E = sparse_mask(n_E, n_PV_FB, self.c_PV_FB_E / n_E)
        self.mask_E_PV_FB = sparse_mask(n_PV_FB, n_E, self.c_E_PV_FB / n_PV_FB)
        self.mask_E_PV_FF = sparse_mask(n_PV_FF, n_E, self.c_E_PV_FF / n_PV_FF)
        self.mask_PV_FF_PV_FB = torch.ones(n_PV_FB, n_PV_FF, device=device)

        # Initialize base weights (fixed)
        w_EE_init = (torch.rand(n_E, n_E, device=device) * self.mask_EE) * (2 * total_weights[0, 0] / self.c_EE)
        w_E_PV_FF_init = (torch.rand(n_PV_FF, n_E, device=device) * self.mask_E_PV_FF) * (2 * total_weights[0, 1] / self.c_E_PV_FF)
        w_PV_FF_PV_FB_init = (torch.rand(n_PV_FB, n_PV_FF, device=device) * self.mask_PV_FF_PV_FB) * (2 * total_weights[1, 2] / n_PV_FB)

        # Plastic weights (buffers, updated by Hebbian plasticity)
        w_PV_FB_E_init = (torch.rand(n_E, n_PV_FB, device=device) * self.mask_PV_FB_E) * (2 * total_weights[2, 0] / self.c_PV_FB_E)
        w_E_PV_FB_init = (torch.rand(n_PV_FB, n_E, device=device) * self.mask_E_PV_FB) * (2 * total_weights[0, 2] / self.c_E_PV_FB)

        self.register_buffer("w_PV_FB_E", w_PV_FB_E_init)
        self.register_buffer("w_E_PV_FB", w_E_PV_FB_init)

        # Fixed baseline weights (scaled by α)
        self.register_buffer("W_EE_base", w_EE_init)
        self.register_buffer("W_E_PV_FF_base", w_E_PV_FF_init)
        self.register_buffer("W_PV_FF_PV_FB_base", w_PV_FF_PV_FB_init)

        # Raw (trainable) alpha parameters (log-space for positivity)
        self.alpha_EE_raw = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.alpha_E_PV_FF_raw = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.alpha_PV_FF_PV_FB_raw = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        # Meta-parameters (η, w_max)
        init_eta, init_wmax = 1e-4, 1.0
        self.eta_raw = nn.Parameter(torch.tensor(np.log(np.expm1(init_eta + 1e-12)), dtype=torch.float32))
        self.w_max_raw = nn.Parameter(torch.tensor(np.log(np.expm1(init_wmax + 1e-12)), dtype=torch.float32))

        self.eps = 1e-12

    # --- Getters ---
    def get_eta(self): return positive(self.eta_raw)
    def get_w_max(self): return positive(self.w_max_raw)
    def get_alpha_EE(self): return torch.exp(self.alpha_EE_raw)
    def get_alpha_E_PV_FF(self): return torch.exp(self.alpha_E_PV_FF_raw)
    def get_alpha_PV_FF_PV_FB(self): return torch.exp(self.alpha_PV_FF_PV_FB_raw)

    def step_with_weights(self, h_E, h_PV_FF, h_PV_FB, EC_E, EC_PV_FF,
                          w_EE, w_E_PV_FF, w_PV_FF_PV_FB, w_PV_FB_E, w_E_PV_FB,
                          dt=1.0):
        """
        Single Euler time step that uses the weights passed in (so we can track updated weights
        as differentiable tensors).
        """
        # Activities
        r_E = F.relu(h_E)
        r_PV_FF = F.relu(h_PV_FF)
        r_PV_FB = F.relu(h_PV_FB)

        # Inputs (note shapes)
        # I_E = (r_E @ w_EE) - (r_PV_FF @ w_E_PV_FF) - (r_PV_FB @ w_E_PV_FB) + EC_E
        I_E = (r_E @ w_EE) - (r_PV_FF @ w_E_PV_FF) - (r_PV_FB @ w_E_PV_FB) + EC_E
        I_PV_FF = EC_PV_FF - (r_PV_FB @ w_PV_FF_PV_FB)
        I_PV_FB = (r_E @ w_PV_FB_E)

        # Euler integration
        dh_E = (-h_E + I_E) / self.tau_E
        dh_PV_FF = (-h_PV_FF + I_PV_FF) / self.tau_PV_FF
        dh_PV_FB = (-h_PV_FB + I_PV_FB) / self.tau_PV_FB

        h_E = h_E + dt * dh_E
        h_PV_FF = h_PV_FF + dt * dh_PV_FF
        h_PV_FB = h_PV_FB + dt * dh_PV_FB

        return h_E, h_PV_FF, h_PV_FB

    def simulate(self, EC_input2E, EC_input2PVFF, T=500, dt=1.0,
                 plasticity=False, anti_hebb=True,
                 initial_weights=None):
        """
        Simulate network over time. All weight updates done to local tensors which are differentiable.
        If initial_weights is provided, it should be a dict of tensors for:
            { "w_EE", "w_E_PV_FF", "w_PV_FF_PV_FB", "w_PV_FB_E", "w_E_PV_FB" }
        Otherwise use model params / buffers as initial.
        Returns dict with:
            - rates: {"E": tensor [T, batch, n_E], ...}
            - final_weights: updated weight tensors (same keys as initial_weights)
        """

        # ensure tensors on correct device
        device = self.device

        batch = EC_input2E.shape[1]

        # initial states
        h_E = torch.zeros(batch, self.n_E, device=device)
        h_PV_FF = torch.zeros(batch, self.n_PV_FF, device=device)
        h_PV_FB = torch.zeros(batch, self.n_PV_FB, device=device)

        rE_hist = torch.zeros(T, batch, self.n_E, device=device)
        rFF_hist = torch.zeros(T, batch, self.n_PV_FF, device=device) 
        rFB_hist = torch.zeros(T, batch, self.n_PV_FB, device=device)

        # get initial weights (as tensors that will be updated)
        if initial_weights is None:
            # Non-plastic base weights: detach to prevent gradient tracking
            W_EE = (self.W_EE_base * self.mask_EE).detach()
            W_E_PV_FF = (self.W_E_PV_FF_base * self.mask_E_PV_FF).detach()
            W_PV_FF_PV_FB = (self.W_PV_FF_PV_FB_base * self.mask_PV_FF_PV_FB).detach()
            
            # Multiply by trainable scalars (gradients tracked for alphas only)
            w_EE = self.get_alpha_EE() * W_EE
            w_E_PV_FF = self.get_alpha_E_PV_FF() * W_E_PV_FF
            w_PV_FF_PV_FB = self.get_alpha_PV_FF_PV_FB() * W_PV_FF_PV_FB
  
            w_PV_FB_E = self.w_PV_FB_E * self.mask_PV_FB_E
            w_E_PV_FB = self.w_E_PV_FB * self.mask_E_PV_FB
        else:
            # use provided updated weights (ensure on correct device / dtype)
            w_EE = initial_weights["w_EE"].to(device=device, dtype=torch.float32) * self.mask_EE
            w_E_PV_FF = initial_weights["w_E_PV_FF"].to(device=device, dtype=torch.float32) * self.mask_E_PV_FF
            w_PV_FF_PV_FB = initial_weights["w_PV_FF_PV_FB"].to(device=device, dtype=torch.float32) * self.mask_PV_FF_PV_FB
            w_PV_FB_E = initial_weights["w_PV_FB_E"].to(device=device, dtype=torch.float32) * self.mask_PV_FB_E
            w_E_PV_FB = initial_weights["w_E_PV_FB"].to(device=device, dtype=torch.float32) * self.mask_E_PV_FB


        # scalar meta-params
        eta = self.get_eta()
        w_max = self.get_w_max()

        # run simulation
        for t in range(T):
            EC_E_t = torch.tensor(EC_input2E[t], dtype=torch.float32, device=device)
            EC_PV_t = torch.tensor(EC_input2PVFF[t], dtype=torch.float32, device=device)

            # step
            h_E, h_PV_FF, h_PV_FB = self.step_with_weights(h_E, h_PV_FF, h_PV_FB,
                                                          EC_E_t, EC_PV_t,
                                                          w_EE=w_EE, w_E_PV_FF=w_E_PV_FF,
                                                          w_PV_FF_PV_FB=w_PV_FF_PV_FB,
                                                          w_PV_FB_E=w_PV_FB_E, w_E_PV_FB=w_E_PV_FB,
                                                          dt=dt)

            r_E = F.relu(h_E)
            r_PV_FF = F.relu(h_PV_FF)
            r_PV_FB = F.relu(h_PV_FB)

            rE_hist[t] = r_E
            rFF_hist[t] = r_PV_FF
            rFB_hist[t] = r_PV_FB

            if plasticity:
                # Hebbian updates 
                delta_w_PVFB_E = eta * r_E.T @ r_PV_FB
                w_PV_FB_E = torch.clamp(w_PV_FB_E + delta_w_PVFB_E * self.mask_PV_FB_E, min=0.0)
                
                delta_w_E_PVFB = eta * r_PV_FB.T @ r_E
                if anti_hebb:
                    delta_w_E_PVFB = -delta_w_E_PVFB
                w_E_PV_FB = torch.clamp(w_E_PV_FB + delta_w_E_PVFB * self.mask_E_PV_FB, min=0.0)
            
                # Column-wise normalization
                w_E_PV_FB = w_E_PV_FB / (w_E_PV_FB.sum(dim=0, keepdim=True) + self.eps) * w_max
                w_PV_FB_E = w_PV_FB_E / (w_PV_FB_E.sum(dim=0, keepdim=True) + self.eps) * w_max
            
                # Detach weights from graph after update
                # w_E_PV_FB = w_E_PV_FB.detach()
                # w_PV_FB_E = w_PV_FB_E.detach()


        # stack rates
        results = {
            "rates": {
                "E": rE_hist,        # [T, batch, n_E]
                "PV_FF": rFF_hist,
                "PV_FB": rFB_hist,
            },
            "final_weights": {
                "w_EE": w_EE,
                "w_E_PV_FF": w_E_PV_FF,
                "w_PV_FF_PV_FB": w_PV_FF_PV_FB,
                "w_PV_FB_E": w_PV_FB_E,
                "w_E_PV_FB": w_E_PV_FB,
            }
        }
        return results

# -------------------------
# Training loop for meta-parameters
# -------------------------
def train_meta_parameters(model: DentateGyrusDiff,
                          EC_train_A, EC_PV_train_A,
                          EC_test_coact, EC_PV_test_coact,
                          assembly_A_mask, assembly_B_mask,
                          epochs=200, lr=1e-3, device='cpu',
                          verbose=True, lambda_contrast=10):
    """
    Training routine:
      1) Run plastic simulation showing assembly A (plasticity=True).
      2) Take final weights from that run and run a non-plastic test with A+B coactivated.
      3) Compute loss = (mean(B_cells_activity) - mean(baseline_activity))^2.
      4) Backpropagate through the entire simulation chain to update meta-parameters:
         - eta (scalar), w_max (scalar) and optionally weight tensors (w_EE, w_E_PV_FF, w_PV_FF_PV_FB).
    Args:
        - model: DentateGyrusDiff
        - EC_train_A: np array [T_train, batch, n_E]
        - EC_PV_train_A: np array [T_train, batch, n_PV_FF]
        - EC_test_coact: np array [T_test, batch, n_E]  (A+B coactivated)
        - EC_PV_test_coact: np array [T_test, batch, n_PV_FF]
        - assembly_A_mask: boolean mask length n_E (1 where A neurons)
        - assembly_B_mask: boolean mask length n_E (1 where B neurons)
    Returns:
        - history: dict of loss per epoch and parameter traces
    """
    model.to(device)
    train_params = [model.eta_raw, model.w_max_raw, model.alpha_EE_raw, 
                    model.alpha_E_PV_FF_raw, model.alpha_PV_FF_PV_FB_raw]

    optimizer = torch.optim.Adam(train_params, lr=lr)

    history = {"loss": [], "eta": [], "w_max": []}

    # convert inputs to numpy->tensors will be done inside simulate
    for ep in range(epochs):
        optimizer.zero_grad()

        # Step 1: plastic training run with assembly A
        res_train = model.simulate(EC_train_A, EC_PV_train_A, T=EC_train_A.shape[0],
                                   plasticity=True, initial_weights=None)

        # get updated weights (these are differentiable tensors)
        updated_weights = res_train["final_weights"]

        # Step 2: non-plastic test run using updated weights
        res_test = model.simulate(EC_test_coact, EC_PV_test_coact, T=EC_test_coact.shape[0],
                                  plasticity=False, initial_weights=updated_weights)

        rates_test = res_test["rates"]["E"]  # [T_test, batch, n_E]
        final_rates = rates_test[-1, 0, :]   # use final time step and batch=0 (shape n_E)

        # compute baseline mean (neurons not in A nor B)
        mask_A = torch.tensor(assembly_A_mask, dtype=torch.bool, device=device)
        mask_B = torch.tensor(assembly_B_mask, dtype=torch.bool, device=device)
        mask_other = ~(mask_A | mask_B)

        if mask_B.sum() == 0 or mask_other.sum() == 0:
            raise ValueError("Assembly B or baseline (other) empty — check masks.")

        mean_A = final_rates[mask_A].mean()
        mean_B = final_rates[mask_B].mean()
        mean_baseline = final_rates[mask_other].mean()

        loss = (mean_B - mean_baseline) ** 2 + lambda_contrast * torch.exp(- (mean_A - mean_B))

        loss.backward()
        optimizer.step()

        # logging
        history["loss"].append(loss.item())
        history["eta"].append(model.get_eta().item())
        history["w_max"].append(model.get_w_max().item())

        if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
            print(f"Epoch {ep+1}/{epochs} loss={loss.item():.6e} eta={model.get_eta().item():.3e} w_max={model.get_w_max().item():.3f}")

    return history, updated_weights

# -------------------------
# Helper: compute assembly masks from EC inputs
# -------------------------
def assembly_mask_from_input(EC_input, threshold=2.0):
    # EC_input shape [T, batch, n_E]
    mean_over_time = EC_input.mean(axis=0).mean(axis=0)  # shape (n_E,)
    return (mean_over_time > threshold).astype(bool)


def recreate_coactivated_inputs(assembly_A_mask, assembly_B_mask, n_E, T=500, batch=1,
                                mean_exc=0.1, std_exc=0.1, assembly_input=4.0):
    
    EC_input2E = np.random.normal(mean_exc, std_exc, (T, batch, n_E))
    EC_input2E2 = np.random.normal(mean_exc, std_exc, (T, batch, n_E))
    EC_input2E[:, :, assembly_A_mask] += assembly_input
    EC_input2E2[:, :, assembly_B_mask] += assembly_input
    EC_input2E_coactivated = np.maximum(EC_input2E, EC_input2E2)
    EC_input2PVFF_coactivated = np.mean(EC_input2E_coactivated, axis=(1, 2)).reshape(T, batch, 1)
    return EC_input2E_coactivated, EC_input2PVFF_coactivated


# -------------------------
# Example usage (adapted from your script)
# -------------------------
if __name__ == "__main__":

    do_run = True
    if do_run:
        # parameters
        n_E = 1000
        batch = 1
        T_assembly = 500
        n_assemblies_distinct = 1
        n_assemblies_total = 5
        
        do_train = False
        
        if do_train:

            # reuse your generate_training_input function (assumed imported / defined)
            EC_input2E, EC_input2PVFF = generate_training_input(T_assembly, batch, n_E, mean_exc=0.1,
                                                                assembly_input=4.0, sparsity=0.01,
                                                                std_exc=0.1, n_assemblies_distinct=n_assemblies_distinct,
                                                                n_assemblies_total=n_assemblies_total,
                                                                coactivation=1, seed=None)
    
            # simplify: let training show a single assembly A for T_total
            T_total = T_assembly * n_assemblies_total
    
            time_const = [10.0, 5.0, 5.0]
            total_weights = np.array([[0,1,1],[0,0,1],[1,0,0]])
    
            device = 'cpu'
            model = DentateGyrusDiff(time_const, total_weights, n_E=n_E, device=device)
    
            # Create a second input (for B) and then coactivate A and B for testing
            EC_input2E2, EC_input2PVFF2 = generate_training_input(T_assembly, batch, n_E, mean_exc=0.1,
                                                                  assembly_input=4.0, sparsity=0.01,
                                                                  std_exc=0.1, n_assemblies_distinct=n_assemblies_distinct,
                                                                  n_assemblies_total=n_assemblies_total,
                                                                  coactivation=1, seed=42)
            # coactivate (elementwise maximum)
            EC_input2E_coactivated = np.maximum(EC_input2E, EC_input2E2)
            # Map to PV_FF input as simple mean (your original code did this; you may change)
            EC_input2PVFF_coactivated = np.mean(EC_input2E_coactivated, axis=(1,2)).reshape(T_total, batch, 1)
    
            # Compute masks for assembly membership
            assembly_A_mask = assembly_mask_from_input(EC_input2E, threshold=2.0)
            assembly_B_mask = assembly_mask_from_input(EC_input2E2, threshold=2.0)
    
            # For training, we want:
            # - a plastic training sequence showing only assembly A (EC_input2E)
            # - a test sequence with A+B coactivated (EC_input2E_coactivated)
            # Using previously created arrays:
            history, final_weights = train_meta_parameters(model, EC_train_A=EC_input2E, EC_PV_train_A=EC_input2PVFF,
                                                           EC_test_coact=EC_input2E_coactivated, 
                                                           EC_PV_test_coact=EC_input2PVFF_coactivated,
                                                           assembly_A_mask=assembly_A_mask, assembly_B_mask=assembly_B_mask,
                                                           epochs=1000, lr=1e-3, device=device, verbose=True, 
                                                           lambda_contrast=15)
    
            # After training you can inspect model.get_eta(), model.get_w_max(), history["loss"]
            print("Training done. Final eta:", model.get_eta().item(), " final w_max:", model.get_w_max().item())
            
            if final_weights is not None:
                model.w_PV_FB_E = final_weights["w_PV_FB_E"].detach().to(device)
                model.w_E_PV_FB = final_weights["w_E_PV_FB"].detach().to(device)
                
                
            #### Save
            save_model_state(model, final_weights, assembly_A_mask, assembly_B_mask, history)
            
        else:
            
            ### Load and initialise 
            model, final_weights, assembly_A_mask, assembly_B_mask, history = load_model_state("../results/dentate_model.pt")
    
            
        ###### test
        EC_input2E_coactivated, EC_input2PVFF_coactivated = recreate_coactivated_inputs(assembly_A_mask, 
                                                                                        assembly_B_mask, 
                                                                                        model.n_E, T=500,
                                                                                        mean_exc=0.1, std_exc=0.1,
                                                                                        assembly_input=4.0)
                                                                                                              
        res_test = model.simulate(EC_input2E_coactivated, EC_input2PVFF_coactivated, T=EC_input2E_coactivated.shape[0],
                                  plasticity=False, initial_weights=final_weights)
        
        rates_E = res_test["rates"]["E"]  # [T, batch, n_E]
        final_rates = rates_E[-1, 0, :]   # final timestep, batch 0

        mask_A = torch.tensor(assembly_A_mask, dtype=torch.bool)
        mask_B = torch.tensor(assembly_B_mask, dtype=torch.bool)
        mask_other = ~(mask_A | mask_B)

        activity_A = final_rates[mask_A].detach().cpu().numpy()
        activity_B = final_rates[mask_B].detach().cpu().numpy()
        activity_other = final_rates[mask_other].detach().cpu().numpy()

        plt.figure(figsize=(7,5))
        
        plt.hist(activity_A, bins=10, alpha=0.6, label="Assembly A", density=True)
        plt.hist(activity_B, bins=10, alpha=0.6, label="Assembly B", density=True)
        plt.hist(activity_other, bins=50, alpha=0.6, label="Other neurons", density=True)
        
        plt.xlabel("Firing rate (a.u.)")
        plt.ylabel("Number of neurons")
        plt.title("Activity distribution after learning")
        plt.legend()
        plt.show()
