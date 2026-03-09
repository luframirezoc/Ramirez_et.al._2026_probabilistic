"""
Figure 5 and Figure 6
---------------------
1. Loads binary population activity from either a hippocampus or retina dataset.
2. Computes the effective field h_eff for a target neuron conditioned on the
   activity state of a group of downstream neurons.
3. Estimates error bars for h_eff by repeated subsampling of experiments.
4. Fits a low-order interaction model (e.g. pairwise) to the inferred h_eff.
5. Produces summary plots.

Important
---------
This code expects external data files that are not included in the repository.
All file-loading sections are therefore written as documented placeholders.
You should replace the paths with your own data locations.

Expected input data structure
-----------------------------
1. Hippocampus dataset (.mat)
   Key required in the MATLAB file:
       'data'
   Expected shape:
       (n_neurons, n_time_bins)
   Contents:
       binary spiking activity, ideally in {0, 1}
       if your data contain -1, those values are converted to 0.

2. Retina dataset (.mat)
   Key required in the MATLAB file:
       'data'
   Expected shape:
       (n_experiments, n_neurons, n_time_bins_per_experiment)
   Contents:
       binary spiking activity in {0, 1} or {-1, 1}; -1 is converted to 0.

3. Pairwise / individual information matrix (.txt)
   Expected shape:
       (n_neurons, n_neurons)
   Contents:
       a square matrix used to sort neurons relative to a target neuron.
       Loaded and symmetrized as:
           indiv_info = indiv_info + indiv_info.T
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import scipy.optimize as spo


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class DatasetConfig:

    dataset_name: str
    activity_path: Path
    indiv_info_path: Path

@dataclass
class AnalysisConfig:
    """Main parameters controlling the effective-field analysis."""

    nd: int = 8                  # Size of the downstream group K
    nc: int = 1                  # Number of conditioned neurons (sigma_0 block)
    group: int = 0               # Which group of ND neurons to analyze
    n_iterations: int = 100      # Bootstrap/subsampling iterations
    threshold_stats: int = 70    # Min valid samples across iterations
    neuron_indices: Tuple[int, ...] = (6,)  # Target neurons to analyze
    fit_order: int = 2           # 0 const, 1 linear, 2 pairwise, ...
    fit_bounds: Tuple[float, float] = (-5.0, 5.0)


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------


def load_dataset(config: DatasetConfig) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Load neural activity and associated information matrix.

    Parameters
    ----------
    config : DatasetConfig
        Dataset specification and file paths.

    Returns
    -------
    S : np.ndarray
        Binary population matrix of shape (n_neurons, n_time_bins).
    indiv_info : np.ndarray
        Square neuron-neuron information matrix.
    n_experiments : int
        Number of experiments/trials.
        For hippocampus data this is set to 1.
    n_time_per_experiment : int
        Number of time bins per experiment.
        For hippocampus data this equals the full recording length.

    Notes
    -----
    Expected external files:

    - Hippocampus:
        .mat file with key 'binary_M', shape (n_neurons, n_time_bins)
    - Retina:
        .mat file with key 'bint', shape (n_experiments, n_neurons, n_time_bins)
    - Individual information matrix:
        .txt file with shape (n_neurons, n_neurons)

    Replace the placeholder paths in `main()` with your own file locations.
    """
    if config.dataset_name not in {"retina", "hippo"}:
        raise ValueError("dataset_name must be 'retina' or 'hippo'.")

    if config.dataset_name == "hippo":
        # ------------------------------------------------------------------
        # Expected file structure:
        #   MATLAB file containing variable 'binary_M'
        #   shape: (n_neurons, n_time_bins)
        # ------------------------------------------------------------------
        data = sio.loadmat(config.activity_path)
        S = np.array(data["binary_M"], dtype=int)

        n_neurons = S.shape[0]
        n_time_per_experiment = S.shape[1]
        n_experiments = 1

        # ------------------------------------------------------------------
        # Expected file structure:
        #   plain-text square matrix of shape (n_neurons, n_neurons)
        #   used to rank/sort neurons relative to a target neuron.
        # ------------------------------------------------------------------
        indiv_info = np.loadtxt(config.indiv_info_path)
        indiv_info = indiv_info + indiv_info.T

    else:
        # ------------------------------------------------------------------
        # Expected file structure:
        #   MATLAB file containing variable 'bint'
        #   shape: (n_experiments, n_neurons, n_time_bins_per_experiment)
        # ------------------------------------------------------------------
        data = sio.loadmat(config.activity_path)
        S_raw = np.array(data["bint"], dtype=int)

        n_experiments = S_raw.shape[0]
        n_neurons = S_raw.shape[1]
        n_time_per_experiment = S_raw.shape[2]

        # ------------------------------------------------------------------
        # Expected file structure:
        #   plain-text square matrix of shape (n_neurons, n_neurons)
        # ------------------------------------------------------------------
        indiv_info = np.loadtxt(config.indiv_info_path)
        indiv_info = indiv_info + indiv_info.T

        # Concatenate all experiments along the time axis so that the final
        # matrix has shape (n_neurons, total_time_bins).
        S = S_raw[0, :, :]
        for i in range(1, n_experiments):
            S = np.append(S, S_raw[i, :, :], axis=1)

    # Sort neurons by mean activity, highest first.
    order = np.argsort(-np.mean(S, axis=1))
    S = S[order, :]

    # Convert any -1 state to 0, preserving binary activity.
    S = np.where(S == -1, 0, S)

    return S, indiv_info, n_experiments, n_time_per_experiment


# -----------------------------------------------------------------------------
# Effective-field computation
# -----------------------------------------------------------------------------


def compute_group_state_codes(S_group: np.ndarray) -> np.ndarray:
    """Encode binary states of a neuron group as integers.

    Parameters
    ----------
    S_group : np.ndarray
        Binary matrix of shape (nd, n_time_bins).

    Returns
    -------
    np.ndarray
        Integer-encoded state per time bin, shape (n_time_bins,).

    Notes
    -----
    For ND neurons, each column is interpreted as a binary number. For example,
    with ND = 3 and state [1, 0, 1], the encoded integer is 1*2^0 + 0*2^1 + 1*2^2 = 5.
    """
    nd = S_group.shape[0]
    return np.sum([2**i * S_group[i, :] for i in range(nd)], axis=0)



def compute_effective_field_for_neuron(
    S: np.ndarray,
    indiv_info: np.ndarray,
    neuron_index: int,
    nd: int,
    group: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute h_eff over all observed group states for one target neuron.

    Parameters
    ----------
    S : np.ndarray
        Binary activity matrix, shape (n_neurons, n_time_bins).
    indiv_info : np.ndarray
        Pairwise information matrix, shape (n_neurons, n_neurons).
    neuron_index : int
        Target neuron whose conditional firing is analyzed.
    nd : int
        Number of downstream neurons in the group.
    group : int
        Index of the ND-sized group.

    Returns
    -------
    h_eff_all : np.ndarray
        Effective field for every possible state in [0, 2**nd - 1].
        Unobserved or ill-defined states are NaN.
    observed_states : np.ndarray
        Encoded states that were present in the data.
    state_inverse : np.ndarray
        Inverse map assigning each time bin to an observed-state index.
    """
    n_states = 2**nd

    # Activity of the conditioned neuron sigma_0.
    s_cu = np.copy(S[neuron_index, :])

    # Reorder neurons according to their relationship with the target neuron.
    sorted_idx = np.argsort(-indiv_info[neuron_index, :])
    S_sorted = np.copy(S[sorted_idx, :])

    # Select the downstream group and encode its binary state per time bin.
    start = group * nd
    stop = start + nd
    s_group = S_sorted[start:stop, :]
    s_down = compute_group_state_codes(s_group)

    observed_states, state_inverse = np.unique(s_down, return_inverse=True)

    h_eff_all = np.full(n_states, np.nan)

    for st_idx, st_code in enumerate(observed_states):
        idx = np.where(state_inverse == st_idx)[0]
        n_spikes = np.sum(s_cu[idx] == 1)

        if n_spikes > 1:
            p1 = n_spikes / idx.size
            if p1 not in (0, 1):
                h_eff_all[st_code] = np.log(p1 / (1 - p1))

    return h_eff_all, observed_states, state_inverse



def estimate_heff_errorbars(
    s_cu: np.ndarray,
    state_inverse: np.ndarray,
    observed_states: np.ndarray,
    n_experiments: int,
    n_time_per_experiment: int,
    n_iterations: int,
    threshold_stats: int,
    n_states_total: int,
) -> np.ndarray:
    n_states_data = observed_states.size

    s_cu_reshaped = np.reshape(s_cu, (n_experiments, n_time_per_experiment))
    state_inv_reshaped = np.reshape(state_inverse, (n_experiments, n_time_per_experiment))

    counter = np.zeros(n_states_data, dtype=int)
    h_eff_samples = np.zeros((n_iterations, n_states_data))

    for _ in range(n_iterations):
        idx_roll = np.arange(n_experiments)
        np.random.shuffle(idx_roll)

        n_train = int(n_experiments * 0.6)
        s_cu_sub = np.reshape(s_cu_reshaped[idx_roll[:n_train], :], n_train * n_time_per_experiment)
        state_sub = np.reshape(state_inv_reshaped[idx_roll[:n_train], :], n_train * n_time_per_experiment)

        for st in range(n_states_data):
            idx = np.where(state_sub == st)[0]
            n_spikes = np.sum(s_cu_sub[idx] == 1)

            if n_spikes > 1:
                p1 = n_spikes / idx.size
                if p1 not in (0, 1):
                    h_eff_samples[counter[st], st] = np.log(p1 / (1 - p1))
                    counter[st] += 1

    h_eff_stats = np.full((2, n_states_total), np.nan)
    for st, st_code in enumerate(observed_states):
        if counter[st] > threshold_stats:
            h_eff_stats[0, st_code] = np.mean(h_eff_samples[:counter[st], st])
            h_eff_stats[1, st_code] = np.std(h_eff_samples[:counter[st], st])

    return h_eff_stats


# -----------------------------------------------------------------------------
# Model construction and fitting
# -----------------------------------------------------------------------------


def number_parameters(order: int, n_sd: int) -> Tuple[np.ndarray, np.ndarray]:
    
    masks = []
    n_params = []

    for k in range(1, order + 1):
        combs = list(combinations(range(n_sd), k))
        n_params.append(len(combs))
        for comb in combs:
            mask = np.zeros(n_sd, dtype=int)
            mask[list(comb)] = 1
            masks.append(mask)

    return np.array(n_params, dtype=int), np.array(masks, dtype=int)



def matrix_sigmas(
    list_states: np.ndarray,
    n_sd: int,
    list_params: np.ndarray,
) -> np.ndarray:
    
    n_list_states = list_states.size
    n_params = list_params.shape[0]
    value = np.zeros((n_list_states, n_params))

    for st in range(n_list_states):
        state_bits = np.array([int(x) for x in np.binary_repr(list_states[st], n_sd)])

        for i in range(n_params):
            idx = np.where(list_params[i, :] == 1)[0]
            value[st, i] = np.prod(state_bits[idx])

    return value



def cost_fun(params: np.ndarray, design_matrix: np.ndarray, heff_true: np.ndarray) -> float:
    """Least-squares objective for fitting h_eff."""
    residual = design_matrix @ params - heff_true
    return np.sum(residual**2)


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------


def decode_ranked_states(sorted_state_indices: np.ndarray, nd: int) -> np.ndarray:
    """Convert ranked integer states into a binary state matrix."""
    state = np.zeros((sorted_state_indices.size, nd), dtype=int)
    for st, code in enumerate(sorted_state_indices):
        bits = np.binary_repr(code, nd)
        for i in range(nd):
            state[st, i] = int(bits[i])
    return state



def plot_heff_with_errorbars(h_eff_all: np.ndarray, h_eff_stats: np.ndarray) -> np.ndarray:
    """Plot ranked effective fields and return the ranking index."""
    idx = np.argsort(h_eff_all)

    plt.figure(figsize=(12, 4))
    plt.plot(
        np.arange(h_eff_all.size),
        h_eff_all[idx],
        label="Data",
        marker=".",
        color="darkred",
        markersize=6,
        linewidth=3,
    )
    plt.errorbar(
        np.arange(h_eff_all.size),
        h_eff_stats[0, idx],
        yerr=h_eff_stats[1, idx],
        linestyle="none",
        markersize=3,
        fillstyle="none",
        marker="o",
        capsize=10,
        color="k",
        label="Error means",
    )
    plt.tick_params(labelsize=25)
    plt.xlabel("Rank", fontsize=20)
    plt.ylabel(r"$h_{\rm eff}$", fontsize=20)
    plt.legend(fontsize=18)
    plt.xlim(-2, h_eff_all.size)
    plt.tight_layout()
    plt.show()

    return idx



def plot_ranked_states(idx: np.ndarray, nd: int) -> None:
    """Plot the binary pattern associated with each ranked state."""
    state = decode_ranked_states(idx, nd)

    plt.figure(figsize=(12, 2))
    plt.imshow(state.T, cmap="binary", extent=[0, idx.size, 0, nd], aspect="auto")
    plt.tick_params(labelsize=25)
    plt.xlabel("Rank", fontsize=20)
    plt.ylabel("Neuron", fontsize=20)
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# Missing legacy helpers from the original project
# -----------------------------------------------------------------------------


def matX_pairwise(array_final,N_states_final): 
    
    array_final=np.array(array_final,dtype='int')
    Order_arr=np.arange(3)
    N_parameters=[int(np.math.factorial(ND)/(np.math.factorial(i)*np.math.factorial(ND-i))) for i in Order_arr]
    N_parameters_total=np.sum(N_parameters)
    state=np.zeros(ND,dtype=int)
    X=np.zeros((N_states_final, N_parameters_total))
    
    X[:,0]=1
    for st in range(N_states_final):
        st_=array_final[st]
        State=np.binary_repr(st_,ND)
        for i in range(ND):
            state[i]=int(State[i])
            
        X[st,1:N_parameters[1]+1]=np.copy(state)
        matrix_p2=np.outer(state,state)
        X[st,N_parameters[1]+1:]=np.concatenate([matrix_p2[i,(i+1):] for i in range(ND)])
    return X


def Calculate_C(h_max,h_opt,h_app):    
    log2=1/np.log(2)
    
    #h_max correspond to the fit of the independent model 
    L1=log2*np.sum(h_max/(1+np.exp(-h_max)))
    L2=np.sum((1/(1+np.exp(-h_max)))*np.log2(1+np.exp(h_max)))+np.sum((1/(1+np.exp(h_max)))*np.log2(1+np.exp(h_max)))
    L_max=L2-L1

    #h_opt correspond to the estimated h_eff from data
    del L1,L2
    L1=log2*np.sum(h_opt/(1+np.exp(-h_opt)))
    L2=np.sum((1/(1+np.exp(-h_opt)))*np.log2(1+np.exp(h_opt)))+np.sum((1/(1+np.exp(h_opt)))*np.log2(1+np.exp(h_opt)))
    L_opt=L2-L1
    
    #h_app correspond to the fit with M parameters
    del L1,L2
    L1=log2*np.sum(h_app/(1+np.exp(-h_app)))
    L2=np.sum((1/(1+np.exp(-h_app)))*np.log2(1+np.exp(h_app)))+np.sum((1/(1+np.exp(h_app)))*np.log2(1+np.exp(h_app)))
    L_app=L2-L1

    return (L_app-L_opt)/(L_max-L_opt)


# -----------------------------------------------------------------------------
# Main analysis example
# -----------------------------------------------------------------------------


def main() -> None:
    # ------------------------------------------------------------------
    # EDIT THESE PATHS FOR YOUR LOCAL MACHINE OR CLUSTER.
    # These files are not part of the repository and should be provided by
    # the user.
    # ------------------------------------------------------------------
    dataset = DatasetConfig(
        dataset_name="retina",
        activity_path=Path("data/raw/data.mat"),
        indiv_info_path=Path("data/raw/inf_sorting_data.txt"),
    )

    analysis = AnalysisConfig()

    S, indiv_info, n_experiments, n_time_per_experiment = load_dataset(dataset)

    n_neurons, n_time = S.shape
    n_states = 2**analysis.nd

    print(f"n_neurons = {n_neurons}, n_time = {n_time}, n_experiments = {n_experiments}")

    for neuron in analysis.neuron_indices:
        h_eff_all, observed_states, state_inverse = compute_effective_field_for_neuron(
            S=S,
            indiv_info=indiv_info,
            neuron_index=neuron,
            nd=analysis.nd,
            group=analysis.group,
        )

        # Reconstruct the conditioned neuron's activity and state assignments.
        s_cu = np.copy(S[neuron, :])

        h_eff_stats = estimate_heff_errorbars(
            s_cu=s_cu,
            state_inverse=state_inverse,
            observed_states=observed_states,
            n_experiments=n_experiments,
            n_time_per_experiment=n_time_per_experiment,
            n_iterations=analysis.n_iterations,
            threshold_stats=analysis.threshold_stats,
            n_states_total=n_states,
        )

        idx = plot_heff_with_errorbars(h_eff_all, h_eff_stats)
        plot_ranked_states(idx, analysis.nd)

        # --------------------------------------------------------------
        # Fit a model up to `fit_order` on the non-NaN states.
        # --------------------------------------------------------------
        heff_true = np.copy(h_eff_all)
        idx_nan = np.where(np.isnan(heff_true))[0]

        heff_true = np.delete(heff_true, idx_nan)
        list_states = np.delete(np.arange(n_states), idx_nan)

        n_params, list_params = number_parameters(analysis.fit_order, analysis.nd)
        design_matrix = matrix_sigmas(list_states, analysis.nd, list_params)

        bounds = np.full((design_matrix.shape[1], 2), analysis.fit_bounds)
        params0 = np.random.random(design_matrix.shape[1])

        result = spo.minimize(
            cost_fun,
            params0,
            args=(design_matrix, heff_true),
            options={"disp": True},
            bounds=bounds,
        )
        params_final = result.x

        heff_learned = design_matrix @ params_final

        plt.figure(figsize=(8, 3))
        plt.plot(
            np.arange(heff_true.size),
            heff_learned,
            marker="o",
            linestyle="none",
            color="orange",
            markersize=4,
            markeredgecolor="k",
            alpha=0.9,
            label="Pairwise fit",
            markeredgewidth=0.9,
        )
        plt.plot(
            np.arange(heff_true.size),
            heff_true,
            color="k",
            linewidth=0.7,
            marker=".",
            markersize=1,
        )
        plt.tick_params(labelsize=14)
        plt.xlabel("Rank", fontsize=20)
        plt.ylabel(r"$h_{\rm eff}$", fontsize=20)
        plt.xlim(-1, n_states)
        plt.ylim(-4, 5.1)
        plt.legend(fontsize=12, frameon=False, loc=4)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
