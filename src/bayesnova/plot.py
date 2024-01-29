import copy
import pygtc
import matplotlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot as pl


def map_to_latex(label: str):
    map_dict = {
        "H0": r"$H_0$",
        "f_host_1": r"$f_1^{\mathrm{host}}$",
        "f_sn_1": r"$f_1^{\mathrm{SN}}$",
        "f_SN_1_max": r"$f_{1,\mathrm{max}}^{\mathrm{SN}}$",
        "M_int": r"$M_{\mathrm{int}}$",
        "delta_M_int": r"$\delta M_{\mathrm{int}}$",
        "M_int_scatter": r"$\sigma_{M_{\mathrm{int}}}$",
        "alpha": r"$\hat{\alpha}$",
        "X_int": r"$\hat{X}_{\mathrm{1,int}}$",
        "delta_X_int": r"$\delta X_{\mathrm{1,int}}$",
        "X_int_scatter": r"$\sigma_{X_{\mathrm{1,int}}}$",
        "delta_X_int_scatter": r"$\delta \sigma_{X_{\mathrm{1,int}}}$",
        "beta": r"$\hat{\beta}$",
        "C_int": r"$\hat{c}_{\mathrm{int}}$",
        "delta_C_int": r"$\delta c_{\mathrm{int}}$",
        "C_int_scatter": r"$\sigma_{c_{\mathrm{int}}}$",
        "delta_C_int_scatter": r"$\delta \sigma_{c_{\mathrm{int}}}$",
        "R_B": r"$\hat{R}_B$",
        "delta_R_B": r"$\delta R_B$",
        "R_B_scatter": r"$\sigma_{R_B}$",
        "delta_R_B_scatter": r"$\delta \sigma_{R_B}$",
        "gamma_EBV": r"$\gamma_{\mathrm{E(B-V)}}$",
        "tau_EBV": r"$\tau_{\mathrm{E(B-V)}}$",
        "delta_tau_EBV": r"$\delta \tau_{\mathrm{E(B-V)}}$",
        "M_host": r"$\hat{M}_{\mathrm{host}}$",
        "M_host_scatter": r"$\sigma_{M_{\mathrm{host}}}$",
        "scaling": r"$a$",
        "offset": r"$b$",
        "gp_sigma": r"$\sigma_{\mathrm{GP}}$",
        "gp_length": r"$\lambda_{\mathrm{GP}}$",
        "gp_noise": r"$\sigma_{\mathrm{noise}}^2$",
    }

    return map_dict[label]


def preprocess_samples(samples: dict, parameters: list):
    param_labels = []
    pop_labels = []
    filtered_samples = {}
    pop_1_samples = {}
    pop_2_samples = {}

    for param in parameters:
        if param not in samples.keys():
            continue
        param_labels.append(map_to_latex(param))
        filtered_samples[param] = np.asarray(samples[param])

        if "delta" in param:
            original_param = param.split("delta_")[1]
            pop_1_samples[original_param] = pop_1 = np.asarray(samples[original_param])
            delta = np.asarray(samples[param])
            if "scatter" in param or "tau" in param:
                pop_2_samples[original_param] = pop_1 * delta
            else:
                pop_2_samples[original_param] = pop_1 + delta
        else:
            pop_1_samples[param] = np.asarray(samples[param])
            pop_2_samples[param] = np.asarray(samples[param])
            pop_labels.append(map_to_latex(param))

    filtered_samples = np.array(list(filtered_samples.values())).T
    pop_1_samples = np.array(list(pop_1_samples.values())).T
    pop_2_samples = np.array(list(pop_2_samples.values())).T

    return filtered_samples, pop_1_samples, pop_2_samples, param_labels, pop_labels


def corner(
    chains: list,
    param_labels: list = None,
    chain_labels: list = None,
    n_countours: int = 2,
    make_transparent: bool = False,
    label_settings: dict = {"family": "Arial", "size": 12},
    tick_settings: dict = {"family": "Arial", "size": 6},
    dpi: int = 300,
    save_path="corner.png",
    **kwargs,
):
    GTC = pygtc.plotGTC(
        chains=chains,
        paramNames=param_labels,
        chainLabels=chain_labels,
        nContourLevels=2,
        legendMarker="All",
        customLabelFont=label_settings,
        customTickFont=tick_settings,
        **kwargs,
    )

    for axes in GTC.axes:
        for spine in axes.spines:
            axes.spines[spine].set_color("white")
        axes.tick_params(axis="both", colors="white")

    GTC.savefig(save_path, dpi=dpi, transparent=make_transparent)
