"""
AESTRA Prediction and Analysis Pipeline

Streamlined prediction module with automated testing suite for:
- Model prediction and RV series generation
- Periodogram analysis
- MCMC orbital inference
- Correlation analysis
- Latent space visualization
"""

# Core imports
import numpy as np
import torch
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks

# Local imports
from src.modeling.train import load_experiment_checkpoint
from torch.utils.data import DataLoader
from src.dataset import generate_collate_fn
import matplotlib.pyplot as plt
from src.ccf import get_full_ccf_analysis
from sklearn.linear_model import LinearRegression
import os
import argparse
import csv

# ==== PERIODOGRAM ANALYSIS ====


def _calculate_period_grid(
    time_values, min_period=1.0, max_period=None, n_periods=5000
):
    """Calculate logarithmic period grid for Lomb-Scargle periodogram."""
    t = np.asarray(time_values, dtype=float)

    if max_period is None:
        baseline = t.max() - t.min()
        max_period = max(baseline / 3.0, min_period * 1.1)

    # Assurer un minimum raisonnable pour min_period
    time_step = np.median(np.diff(np.sort(t)))
    min_period_safe = max(
        min_period, 2 * time_step
    )  # Au moins 2x l'échantillonnage temporel

    # print(
    #     f"Period grid: min={min_period_safe:.3f}, max={max_period:.3f}, n_periods={n_periods}"
    # )
    # print(f"Time baseline: {baseline:.3f}, median time step: {time_step:.3f}")

    periods = np.logspace(np.log10(min_period_safe), np.log10(max_period), n_periods)
    return periods


def _compute_lomb_scargle(
    time_values, y_values, periods, fit_mean=True, center_data=True
):
    """Compute Lomb-Scargle periodogram power."""
    t = np.asarray(time_values, dtype=float)
    y = np.asarray(y_values, dtype=float)

    # Diagnostic des données d'entrée
    # print(f"Time values: min={t.min():.3f}, max={t.max():.3f}, shape={t.shape}")
    # print(
    #     f"Y values: min={y.min():.3f}, max={y.max():.3f}, shape={y.shape}, std={y.std():.6f}"
    # )
    # print(f"Time NaN count: {np.sum(np.isnan(t))}, Y NaN count: {np.sum(np.isnan(y))}")
    # print(f"Time Inf count: {np.sum(np.isinf(t))}, Y Inf count: {np.sum(np.isinf(y))}")

    # Vérification des données
    if np.any(np.isnan(t)) or np.any(np.isnan(y)):
        print("ERROR: Input data contains NaN values")
        # Nettoyage des données
        mask = np.isfinite(t) & np.isfinite(y)
        t = t[mask]
        y = y[mask]
        print(f"After cleaning: t.shape={t.shape}, y.shape={y.shape}")

    if len(t) < 3:
        print("ERROR: Not enough valid data points for periodogram")
        return None, np.full_like(periods, np.nan)

    if np.std(y) == 0:
        print("ERROR: Y values have zero variance")
        return None, np.full_like(periods, 0.0)

    # Diagnostic plus poussé si variance très faible
    if np.std(y) < 1e-10:
        print(f"WARNING: Very low variance in Y values: {np.std(y):.2e}")
        print(f"Y range: {np.ptp(y):.2e}")
        # Essayer de rescaler les données
        y_rescaled = (
            y - np.mean(y)
        ) * 1e6  # Rescaler pour éviter les problèmes numériques
        print(f"Rescaled Y std: {np.std(y_rescaled):.6f}")
        y = y_rescaled

    frequencies = 1.0 / periods
    # print(f"Frequency range: {frequencies.min():.6f} to {frequencies.max():.6f}")
    # print(f"Period range: {periods.min():.3f} to {periods.max():.3f}")

    try:
        # Essayer avec des paramètres plus robustes
        ls = LombScargle(
            t, y, fit_mean=fit_mean, center_data=center_data, normalization="standard"
        )

        # Calculer sur une grille plus restreinte d'abord pour tester
        test_freqs = frequencies[::100]  # Sous-échantillonner pour test
        test_power = ls.power(test_freqs)

        if np.any(np.isnan(test_power)):
            print(
                "ERROR: Even test frequencies produce NaN, trying different normalization"
            )
            ls = LombScargle(
                t, y, fit_mean=fit_mean, center_data=center_data, normalization="psd"
            )
            test_power = ls.power(test_freqs)

            if np.any(np.isnan(test_power)):
                print("ERROR: All normalizations fail, returning zeros")
                return None, np.full_like(periods, 0.0)

        # Si le test passe, calculer sur toute la grille
        power = ls.power(frequencies)

        # print(f"Power: min={np.nanmin(power):.6f}, max={np.nanmax(power):.6f}")
        # print(
        #     f"Power NaN count: {np.sum(np.isnan(power))}, Inf count: {np.sum(np.isinf(power))}"
        # )

        return ls, power
    except Exception as e:
        print(f"ERROR in LombScargle computation: {e}")
        return None, np.full_like(periods, np.nan)


def _get_planet_window_mask(periods, P_inj, exclude_width_frac=0.05):
    """Create mask for planet period window."""
    if P_inj is not None and P_inj > 0:
        return np.abs(periods - P_inj) <= exclude_width_frac * P_inj
    return np.zeros_like(periods, dtype=bool)


def _analyze_planet_detection(
    periods, power, mask_planet_window, P_inj, ls, ls_method="baluev"
):
    """Analyze planet detection metrics within the planet window."""
    if P_inj is None or P_inj <= 0:
        return None, None, None, None

    idx_window = np.where(mask_planet_window)[0]
    if idx_window.size == 0:
        return None, None, None, None

    # Best peak in planet window
    i_best = idx_window[np.argmax(power[idx_window])]
    P_detected = periods[i_best]
    P_planet_power = power[i_best]

    # Calculate FAP and power ratio
    fap_at_Pinj = float(ls.false_alarm_probability(P_planet_power, method=ls_method))

    power_ratio = None
    if np.any(~mask_planet_window):
        max_outside = float(np.max(power[~mask_planet_window]))
        if max_outside > 0:
            power_ratio = float(P_planet_power / max_outside)

    delta_P = float(abs(P_detected - P_inj))

    return fap_at_Pinj, power_ratio, P_detected, delta_P


def _count_significant_peaks_outside(
    periods,
    power,
    mask_planet_window,
    ls,
    fap_threshold=0.01,
    peak_prominence=None,
    ls_method="baluev",
):
    """Count significant peaks outside the planet window."""
    if peak_prominence is None:
        peak_prominence = 0.5 * np.std(power)

    power_out = power[~mask_planet_window]
    peaks_out, _ = find_peaks(power_out, prominence=peak_prominence)

    if peaks_out.size == 0:
        return 0

    faps_out = ls.false_alarm_probability(power_out[peaks_out], method=ls_method)
    return int(np.sum(faps_out < fap_threshold))


def compute_periodogram_metrics(
    y_values,
    time_values,
    P_inj=None,
    min_period=2.0,
    max_period=None,
    n_periods=5000,
    fap_threshold=0.01,
    exclude_width_frac=0.05,
    peak_prominence=None,
    ls_method="baluev",
    fit_mean=True,
    center_data=True,
):
    """
    Compute Lomb-Scargle periodogram and detection metrics.

    Parameters
    ----------
    y_values : array
        RV series or residuals to analyze
    time_values : array
        Time values (typically in days)
    P_inj : float, optional
        Injected period (days). If None, P_inj-related metrics will be None
    min_period, max_period : float
        Period bounds (days). If max_period=None, set to (tmax-tmin)/3
    n_periods : int
        Size of logarithmic period grid
    fap_threshold : float
        FAP threshold for counting significant peaks
    exclude_width_frac : float
        Relative width of exclusion window around P_inj
    peak_prominence : float, optional
        Peak prominence for detection. If None, auto-set to 0.5*std(power)
    ls_method : str
        FAP method ('baluev', 'naive', 'bootstrap')
    fit_mean, center_data : bool
        LombScargle options

    Returns
    -------
    periods : array
        Period grid (days)
    power : array
        Corresponding LS power
    metrics : dict
        Detection metrics
    """
    # Calculate period grid and periodogram
    periods = _calculate_period_grid(time_values, min_period, max_period, n_periods)
    ls, power = _compute_lomb_scargle(
        time_values, y_values, periods, fit_mean, center_data
    )

    # Vérifier si le calcul a échoué
    if ls is None or np.all(np.isnan(power)):
        print("ERROR: Periodogram computation failed, returning empty metrics")
        metrics = {
            "fap_at_Pinj": None,
            "power_ratio": None,
            "n_sig_peaks_outside": 0,
            "P_detected": None,
            "delta_P": None,
        }
        return periods, power, metrics

    # Create planet window mask
    mask_planet_window = _get_planet_window_mask(periods, P_inj, exclude_width_frac)

    # Analyze planet detection
    fap_at_Pinj, power_ratio, P_detected, delta_P = _analyze_planet_detection(
        periods, power, mask_planet_window, P_inj, ls, ls_method
    )

    # Count significant peaks outside planet window
    n_sig_peaks_outside = _count_significant_peaks_outside(
        periods,
        power,
        mask_planet_window,
        ls,
        fap_threshold,
        peak_prominence,
        ls_method,
    )

    metrics = {
        "fap_at_Pinj": fap_at_Pinj,
        "power_ratio": power_ratio,
        "n_sig_peaks_outside": n_sig_peaks_outside,
        "P_detected": P_detected,
        "delta_P": delta_P,
    }

    return periods, power, metrics


def plot_periodogram(
    periods,
    power,
    metrics=None,
    P_inj=None,
    fap_threshold=0.01,
    exclude_width_frac=0.05,
    peak_prominence=None,
    title="Lomb–Scargle Periodogram",
    save_path=None,
    show_plot=False,
    xlim=None,
):
    """
    Plot periodogram with detection metrics and annotations.
    Supports multiple injected periods.

    Parameters
    ----------
    periods : array
        Period grid (days)
    power : array
        LS power values
    metrics : dict or list of dict, optional
        Detection metrics from compute_periodogram_metrics.
        If list, should match length of P_inj list.
    P_inj : float or list of float, optional
        Injected period(s) for reference
    fap_threshold : float
        FAP threshold for display
    exclude_width_frac : float
        Planet window width fraction
    peak_prominence : float, optional
        Peak prominence for outside peak detection
    title : str
        Plot title
    save_path : str, optional
        If provided, save the figure to this path
    show_plot : bool
        If True, display the figure instead of closing
    xlim : tuple, optional
        If provided, apply x-axis limits (min, max)
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Base periodogram plot
    ax.semilogx(periods, power, lw=1.6, color="blue")
    ax.set_xlabel("Période [jours]")
    ax.set_ylabel("Puissance Lomb–Scargle")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)

    # Add FAP threshold line with validation
    if np.any(np.isnan(power)) or np.any(np.isinf(power)):
        print("Warning: power array contains NaN or Inf values")
        power_clean = power[np.isfinite(power)]
        if len(power_clean) == 0:
            print("Error: All power values are NaN or Inf")
            max_power = 1.0  # fallback value
        else:
            max_power = np.max(power_clean)
    else:
        max_power = np.max(power)

    if np.isnan(max_power) or np.isinf(max_power) or max_power <= 0:
        print(f"Warning: max_power is invalid ({max_power}), using fallback")
        max_power = 1.0

    fap_level = max_power * fap_threshold

    # print(
    #     f"max_power: {max_power:.3f}, fap_threshold: {fap_threshold}, fap_level: {fap_level:.3f}"
    # )

    if np.isfinite(fap_level) and fap_level > 0:
        ax.axhline(
            fap_level,
            ls="--",
            lw=1.2,
            color="red",
            label=f"Seuil FAP = {int(fap_threshold * 100)}%",
        )
    else:
        print(f"Skipping FAP threshold line due to invalid fap_level: {fap_level}")

    # Handle multiple periods
    if P_inj is not None:
        # Convert to lists for uniform handling
        P_inj_list = P_inj if isinstance(P_inj, (list, tuple, np.ndarray)) else [P_inj]

        if metrics is not None:
            metrics_list = metrics if isinstance(metrics, list) else [metrics]
            # Ensure metrics list matches P_inj list length
            if len(metrics_list) != len(P_inj_list):
                metrics_list = (
                    [metrics_list[0]] * len(P_inj_list)
                    if metrics_list
                    else [None] * len(P_inj_list)
                )
        else:
            metrics_list = [None] * len(P_inj_list)

        # Color palette for multiple planets
        colors = plt.cm.Set1(np.linspace(0, 1, max(len(P_inj_list), 3)))

        # Combined exclusion mask for all planets
        combined_mask = np.zeros_like(periods, dtype=bool)

        for i, (P_planet, planet_metrics) in enumerate(zip(P_inj_list, metrics_list)):
            if P_planet is None or P_planet <= 0:
                continue

            color = colors[i % len(colors)]

            # Planet exclusion band
            ax.axvspan(
                P_planet * (1 - exclude_width_frac),
                P_planet * (1 + exclude_width_frac),
                alpha=0.15,
                color=color,
                label=f"Planète {i + 1}: P = {P_planet:.3f} j",
            )

            # Update combined mask
            planet_mask = np.abs(periods - P_planet) <= exclude_width_frac * P_planet
            combined_mask |= planet_mask

            # Reference line for P_inj
            ax.axvline(P_planet, color=color, lw=1.5, alpha=0.7)

            # Detected peak marker if metrics available
            if planet_metrics and planet_metrics.get("P_detected") is not None:
                P_detected = planet_metrics["P_detected"]
                idx = np.argmin(np.abs(periods - P_detected))
                ax.plot(
                    periods[idx],
                    power[idx],
                    marker="o",
                    ms=8,
                    mec=color,
                    mfc="none",
                    mew=2,
                    label=f"Détecté {i + 1}: {P_detected:.3f} j",
                )
    else:
        combined_mask = np.zeros_like(periods, dtype=bool)

    # Mark significant peaks outside all planet windows
    if peak_prominence is None:
        peak_prominence = 0.5 * np.std(power)

    if np.any(~combined_mask):
        p_out = power[~combined_mask]
        per_out = periods[~combined_mask]
        peaks_out, _ = find_peaks(p_out, prominence=peak_prominence)

        if peaks_out.size > 0:
            ax.plot(
                per_out[peaks_out],
                p_out[peaks_out],
                "x",
                ms=6,
                color="orange",
                label="Pics significatifs",
            )

    # Add metrics text box (for first planet if multiple)
    if metrics is not None:
        first_metrics = metrics if not isinstance(metrics, list) else metrics[0]
        if first_metrics:
            lines = []
            if first_metrics.get("fap_at_Pinj") is not None:
                lines.append(f"FAP: {first_metrics['fap_at_Pinj']:.3g}")
            if first_metrics.get("power_ratio") is not None:
                lines.append(f"Ratio: {first_metrics['power_ratio']:.3g}")
            if first_metrics.get("n_sig_peaks_outside") is not None:
                lines.append(f"Pics ext.: {first_metrics['n_sig_peaks_outside']}")
            if first_metrics.get("delta_P") is not None:
                lines.append(f"ΔP: {first_metrics['delta_P']:.3g} j")

            if lines:
                txt = "\n".join(lines)
                ax.text(
                    0.02,
                    0.98,
                    txt,
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    bbox=dict(
                        boxstyle="round,pad=0.4", fc="white", ec="0.7", alpha=0.9
                    ),
                    fontsize=10,
                )

    if xlim is not None:
        ax.set_xlim(*xlim)

    ax.legend(loc="best", frameon=True)
    plt.tight_layout()

    # Save or show
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        if show_plot:
            plt.show()
        else:
            plt.close()


# ==== PREDICTION  ====


def predict(model, dataset, batch_size=64, perturbation_value=1.0):
    """Extract latent vectors and RV values from model and dataset."""
    all_s = []
    all_saug = []
    all_vobs_list = []
    rv_pred_aug_list = []
    all_vaug_list = []
    all_yact = []
    all_yact_aug = []
    all_yobs_prime = []
    all_yact_perturbed = {}

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=generate_collate_fn(dataset=dataset),
    )

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch_yobs, batch_yaug, batch_voffset_true, batch_wavegrid = batch

            batch_vobs_pred, batch_vaug_pred = model.get_rvestimator_pred(
                batch_yobs=batch_yobs, batch_yaug=batch_yaug
            )

            batch_yobs_prime, batch_yact, batch_yact_aug, batch_s, batch_saug = (
                model.get_spender_pred(
                    batch_yobs=batch_yobs,
                    batch_yaug=batch_yaug,
                    batch_wavegrid=batch_wavegrid,
                    batch_vobs_pred=batch_vobs_pred,
                )
            )

            latent_dim = batch_s.shape[1]

            if not all_yact_perturbed:
                for dim in range(latent_dim):
                    all_yact_perturbed[dim] = []

            for dim in range(latent_dim):
                batch_s_dim_perturbed = batch_s.clone()
                batch_s_dim_perturbed[:, dim] += perturbation_value
                batch_yact_perturbed = model.spender.decoder(batch_s_dim_perturbed)
                all_yact_perturbed[dim].append(
                    batch_yact_perturbed.cpu().detach().numpy()
                )

            all_s.append(batch_s.cpu().detach().numpy())
            all_saug.append(batch_saug.cpu().detach().numpy())
            all_vaug_list.append(batch_voffset_true.cpu().detach().numpy())
            all_vobs_list.append(batch_vobs_pred.cpu().detach().numpy())
            rv_pred_aug_list.append(batch_vaug_pred.cpu().detach().numpy())
            all_yact.append(batch_yact.cpu().detach().numpy())
            all_yact_aug.append(batch_yact_aug.cpu().detach().numpy())
            all_yobs_prime.append(batch_yobs_prime.cpu().detach().numpy())

    # Concatenate results
    all_s = np.concatenate(all_s, axis=0)
    all_saug = np.concatenate(all_saug, axis=0)
    all_vaug = np.concatenate(all_vaug_list, axis=0)
    all_vobs = np.concatenate(all_vobs_list, axis=0)
    rv_pred_aug = np.concatenate(rv_pred_aug_list, axis=0)
    all_yact = np.concatenate(all_yact, axis=0)
    all_yact_aug = np.concatenate(all_yact_aug, axis=0)
    all_yobs_prime = np.concatenate(all_yobs_prime, axis=0)

    latent_dim = len(all_yact_perturbed)
    all_yact_perturbed_array = np.array(
        [np.concatenate(all_yact_perturbed[dim], axis=0) for dim in range(latent_dim)]
    )

    return {
        "all_s": all_s,
        "all_saug": all_saug,
        "all_vobs": all_vobs,
        "rv_pred_aug": rv_pred_aug,
        "all_vaug": all_vaug,
        "all_rvs": all_vaug,  # backwards compatibility
        "all_yact": all_yact,
        "all_yact_aug": all_yact_aug,
        "all_yobs_prime": all_yobs_prime,
        "all_yact_perturbed": all_yact_perturbed_array,
    }


def get_vapparent(dataset, CCF_params):
    full_analysis = get_full_ccf_analysis(
        spectra=dataset.spectra.cpu().detach().numpy(),
        wavegrid=dataset.wavegrid.cpu().detach().numpy(),
        **CCF_params,
    )
    v_apparent = full_analysis["rv"]
    depth = full_analysis["depth"]
    span = full_analysis["span"]
    fwhm = full_analysis["fwhm"]

    return v_apparent, depth, span, fwhm


def get_vref(dataset, CCF_params):
    full_analysis = get_full_ccf_analysis(
        spectra=dataset.spectra_no_activity.cpu().detach().numpy(),
        wavegrid=dataset.wavegrid.cpu().detach().numpy(),
        **CCF_params,
    )
    v_ref = full_analysis["rv"]
    depth = full_analysis["depth"]
    span = full_analysis["span"]
    fwhm = full_analysis["fwhm"]

    return v_ref, depth, span, fwhm


def get_vtraditionnal(v_apparent, depth, span, fwhm):
    X = np.column_stack([fwhm, depth, span])
    model = LinearRegression().fit(X, v_apparent)
    v_pred = model.predict(X)
    return v_apparent - v_pred


# ==== CORRELATION MATRIX ====


def plot_correlation_matrix(
    v_apparent,
    v_correct,
    v_traditionnal,
    v_ref,
    depth,
    span,
    fwhm,
    latent_vectors,
    save_path=None,
    show_plot=False,
):
    """Plot correlation matrix between RV methods and activity indicators/latent dimensions."""

    # Prepare data
    n_latent = latent_vectors.shape[1]

    # X-axis variables (indicators + latent dims)
    x_vars = {
        "FWHM": fwhm,
        "Span": span,
        "Depth": depth,
    }
    for i in range(n_latent):
        x_vars[f"s_{i + 1}"] = latent_vectors[:, i]

    # Y-axis variables (RV methods)
    y_vars = {
        "v_apparent": v_apparent,
        "v_correct": v_correct,
        "v_traditionnal": v_traditionnal,
        "v_ref": v_ref,
    }

    # Compute correlations
    x_names = list(x_vars.keys())
    y_names = list(y_vars.keys())
    corr_matrix = np.zeros((len(y_names), len(x_names)))

    for i, y_name in enumerate(y_names):
        for j, x_name in enumerate(x_names):
            corr_matrix[i, j] = np.corrcoef(y_vars[y_name], x_vars[x_name])[0, 1]

    # Plot
    fig, ax = plt.subplots(figsize=(max(8, len(x_names) * 0.8), 6))
    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    # Labels
    ax.set_xticks(range(len(x_names)))
    ax.set_xticklabels(x_names, rotation=45, ha="right")
    ax.set_yticks(range(len(y_names)))
    ax.set_yticklabels(y_names)

    # Add correlation values
    for i in range(len(y_names)):
        for j in range(len(x_names)):
            ax.text(
                j,
                i,
                f"{corr_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=10,
            )

    # Colorbar and title
    plt.colorbar(im, ax=ax, label="Corrélation")
    ax.set_title("Matrice de corrélation : Méthodes RV vs Indicateurs d'activité")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        if show_plot:
            plt.show()
        else:
            plt.close()


# === LATENT SPACE VISUALIZATION ===


def compute_latent_distances(all_s, all_saug, seed=None):
    """Compute latent distances for random pairs and augmented pairs."""
    n = all_s.shape[0]
    if seed is not None:
        np.random.seed(seed)

    inds = np.array([np.random.choice(n, size=2, replace=False) for _ in range(n)])
    delta_s_rand = np.linalg.norm(all_s[inds[:, 0]] - all_s[inds[:, 1]], axis=1)
    delta_s_aug = np.linalg.norm(all_s - all_saug, axis=1)
    return delta_s_rand, delta_s_aug


def plot_latent_distance_distribution(
    delta_s_rand, delta_s_aug, save_path=None, show_plot=False
):
    """
    Crée le plot de la Figure 3 avec les distributions de distances latentes.

    Args:
        delta_s_rand: Distances latentes pour les paires aléatoires
        delta_s_aug: Distances latentes pour les paires augmentées
        save_path: Chemin pour sauvegarder la figure (optionnel)
    """
    # Configuration du plot
    plt.figure(figsize=(8, 6))

    # Calcul des statistiques pour les légendes
    mean_rand = np.mean(delta_s_rand)
    mean_aug = np.mean(delta_s_aug)

    # Détermination automatique de la plage des valeurs
    all_values = np.concatenate([delta_s_rand, delta_s_aug])
    min_val = np.min(
        all_values[all_values > 0]
    )  # Éviter les valeurs nulles pour le log
    max_val = np.max(all_values)

    # Extension de la plage pour une meilleure visualisation
    x_min = min_val * 0.5
    x_max = max_val * 2.0

    # Création des histogrammes avec bins adaptés aux données réelles
    bins = np.logspace(np.log10(x_min), np.log10(x_max), 50)

    plt.hist(
        delta_s_rand,
        bins=bins,
        alpha=0.7,
        color="blue",
        label=f"(∆s_rand): {mean_rand:.3e}",
        density=False,
    )
    plt.hist(
        delta_s_aug,
        bins=bins,
        alpha=0.7,
        color="red",
        label=f"(∆s_aug): {mean_aug:.3e}",
        density=False,
    )

    # Configuration des axes et labels
    plt.xlabel("latent distance ∆s", fontsize=12)
    plt.ylabel("N", fontsize=12)
    plt.xscale("log")
    plt.xlim(x_min, x_max)

    # Ajout de la légende
    plt.legend(fontsize=12)

    # Configuration de la grille
    plt.grid(True, alpha=0.3)

    plt.subplots_adjust(bottom=0.15)

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Figure sauvegardée: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_activity_perturbation(
    y_act_original,
    y_act_perturbed_list,
    wavelength,
    save_path=None,
    show_plot=False,
    wave_range=(5000, 5010),
):
    """
    Visualisation de l'effet de la perturbation du vecteur latent pour TOUTES les dimensions fournies.

    Args:
        y_act_original: Spectre d'activité original (1D)
        y_act_perturbed_list: Liste des spectres d'activité perturbés (liste de 1D)
        wavelength: Grille de longueurs d'onde (1D)
        save_path: Chemin de sauvegarde (optionnel)
        show_plot: Afficher le plot (False par défaut)
        wave_range: Gamme de longueurs d'onde à afficher (tuple)
    """
    n_dims = len(y_act_perturbed_list)
    if n_dims == 0:
        return

    # Palette de couleurs
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, n_dims)))

    # Filtre pour la gamme de longueurs d'onde
    wave_mask = (wavelength >= wave_range[0]) & (wavelength <= wave_range[1])
    wave_filtered = wavelength[wave_mask]
    y_original_filtered = y_act_original[wave_mask]

    # Création de la figure avec subplots pour chaque perturbation
    fig, axes = plt.subplots(n_dims, 1, figsize=(10, max(6, 2 * n_dims)), sharex=True)
    if n_dims == 1:
        axes = [axes]

    fig.suptitle(
        "Visualization of the effect of perturbing the latent vector",
        fontsize=14,
        fontweight="bold",
    )

    for i, y_perturbed in enumerate(y_act_perturbed_list):
        ax = axes[i]
        color = colors[i % len(colors)]
        label = f"Perturb s_{i + 1}"

        y_perturbed_filtered = y_perturbed[wave_mask]

        # Plot du spectre original (courbe noire)
        ax.plot(wave_filtered, y_original_filtered, "k-", linewidth=1.2, alpha=0.8)

        # Plot du spectre perturbé (courbe colorée)
        ax.plot(
            wave_filtered,
            y_perturbed_filtered,
            color=color,
            linewidth=1.2,
            alpha=0.95,
            label=label,
        )

        # Configuration des axes
        ax.set_ylabel("y_act", fontsize=11)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

        # Ajustement des limites y pour bien voir les différences
        y_min = min(y_original_filtered.min(), y_perturbed_filtered.min())
        y_max = max(y_original_filtered.max(), y_perturbed_filtered.max())
        margin = (y_max - y_min) * 0.1 if y_max > y_min else 1e-3
        ax.set_ylim(y_min - margin, y_max + margin)

    # Configuration de l'axe x pour le dernier subplot
    axes[-1].set_xlabel("Restframe wavelength (Å)", fontsize=11)

    plt.tight_layout(rect=[0, 0.02, 1, 0.98])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure sauvegardée: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_latent_marginal_distributions(all_s, save_path=None, show_plot=False):
    """Plot histograms of each latent coordinate distribution."""
    n, d = all_s.shape
    n_cols = min(4, d)
    n_rows = int(np.ceil(d / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols + 1, 2.4 * n_rows))
    axes = np.array(axes).reshape(n_rows, n_cols)

    for i in range(d):
        r, c = divmod(i, n_cols)
        ax = axes[r, c]
        ax.hist(all_s[:, i], bins=40, color="steelblue", alpha=0.85, edgecolor="k")
        ax.set_title(f"s_{i + 1}", fontsize=10)
        ax.grid(True, alpha=0.25)

    # Hide unused axes
    for j in range(d, n_rows * n_cols):
        r, c = divmod(j, n_cols)
        axes[r, c].axis("off")

    fig.suptitle("Latent coordinates distributions", fontsize=14)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        if show_plot:
            plt.show()
        else:
            plt.close()


def plot_latent_space_3d(
    latent_s,
    rv_values,
    save_path=None,
    show_plot=False,
    decorrelated=False,
):
    """Plot a 3D latent space with 2D projections, colored by RV values.

    Args:
        latent_s (np.ndarray): latent vectors of shape (N, D)
        rv_values (np.ndarray): RV values of shape (N,)
        save_path (str, optional): path to save the figure
        show_plot (bool): whether to display the plot
        decorrelated (bool): flag to indicate decorrelated RV (for title)

    Returns:
        bool: True if plotted, False if latent space < 3D
    """
    if latent_s.shape[1] < 3:
        print(f"⚠️  L'espace latent n'est que {latent_s.shape[1]}D, plot 3D impossible")
        return False

    if save_path is None:
        save_path = "reports/figures/latent_space_3d.png"

    s1, s2, s3 = latent_s[:, 0], latent_s[:, 1], latent_s[:, 2]

    print(f"RV range: [{np.min(rv_values):.3f}, {np.max(rv_values):.3f}] m/s")
    print(f"Nombre de spectres: {len(rv_values)}")

    fig = plt.figure(figsize=(20, 15))

    # 3D scatter
    ax_3d = fig.add_subplot(2, 3, (1, 4), projection="3d")
    scatter_3d = ax_3d.scatter(
        s1, s2, s3, c=rv_values, cmap="viridis", s=20, alpha=0.7, edgecolors="none"
    )
    ax_3d.set_xlabel("S₁")
    ax_3d.set_ylabel("S₂")
    ax_3d.set_zlabel("S₃")
    ax_3d.set_title(
        f"Espace latent 3D coloré par V_encode [m/s] {'DECORRELATED' if decorrelated else ''}"
    )
    cbar_3d = plt.colorbar(scatter_3d, ax=ax_3d, shrink=0.6)
    cbar_3d.set_label("V_encode [m/s]")

    def add_correlation_analysis(ax, x_data, y_data, x_label, y_label):
        corr = np.corrcoef(x_data, y_data)[0, 1]
        slope, intercept = np.polyfit(x_data, y_data, 1)
        x_range = np.linspace(x_data.min(), x_data.max(), 100)
        y_fit = slope * x_range + intercept
        ax.plot(
            x_range,
            y_fit,
            "r--",
            linewidth=2,
            alpha=0.8,
            label=f"R={corr:.3f}, slope={slope:.3f}",
        )
        ax.set_title(f"{ax.get_title()}\nR={corr:.3f}")
        ax.legend(fontsize=8, loc="best")
        return corr, slope

    # 2D projections
    ax_12 = fig.add_subplot(2, 3, 2)
    ax_12.scatter(
        s1, s2, c=rv_values, cmap="viridis", s=15, alpha=0.7, edgecolors="none"
    )
    ax_12.set_xlabel("S₁")
    ax_12.set_ylabel("S₂")
    ax_12.set_title("Projection S₁-S₂")
    ax_12.grid(True, alpha=0.3)
    corr_12, slope_12 = add_correlation_analysis(ax_12, s1, s2, "S₁", "S₂")

    ax_13 = fig.add_subplot(2, 3, 3)
    ax_13.scatter(
        s1, s3, c=rv_values, cmap="viridis", s=15, alpha=0.7, edgecolors="none"
    )
    ax_13.set_xlabel("S₁")
    ax_13.set_ylabel("S₃")
    ax_13.set_title("Projection S₁-S₃")
    ax_13.grid(True, alpha=0.3)
    corr_13, slope_13 = add_correlation_analysis(ax_13, s1, s3, "S₁", "S₃")

    ax_23 = fig.add_subplot(2, 3, 6)
    ax_23.scatter(
        s2, s3, c=rv_values, cmap="viridis", s=15, alpha=0.7, edgecolors="none"
    )
    ax_23.set_xlabel("S₂")
    ax_23.set_ylabel("S₃")
    ax_23.set_title("Projection S₂-S₃")
    ax_23.grid(True, alpha=0.3)
    corr_23, slope_23 = add_correlation_analysis(ax_23, s2, s3, "S₂", "S₃")

    # RV histogram and stats
    ax_hist = fig.add_subplot(2, 3, 5)
    ax_hist.hist(
        rv_values,
        bins=50,
        alpha=0.8,
        color="skyblue",
        edgecolor="black",
        label="Distribution RV",
    )
    ax_hist.set_xlabel("V_encode [m/s]")
    ax_hist.set_ylabel("Fréquence")
    ax_hist.set_title("Distribution des vitesses radiales")
    ax_hist.grid(True, alpha=0.3)
    ax_hist.legend(fontsize=8)

    corr_s1_rv = np.corrcoef(s1, rv_values)[0, 1]
    corr_s2_rv = np.corrcoef(s2, rv_values)[0, 1]
    corr_s3_rv = np.corrcoef(s3, rv_values)[0, 1]

    stats_text = f"""Statistiques:
N spectres: {len(rv_values)}
RV:
  Min: {np.min(rv_values):.3f} m/s
  Max: {np.max(rv_values):.3f} m/s
  Mean: {np.mean(rv_values):.3f} m/s
  Std: {np.std(rv_values):.3f} m/s
Dim latente: {latent_s.shape[1]}D

Corrélations entre dimensions:
  S₁-S₂: R={corr_12:.3f}
  S₁-S₃: R={corr_13:.3f}
  S₂-S₃: R={corr_23:.3f}

Corrélations avec RV:
  S₁-RV: R={corr_s1_rv:.3f}
  S₂-RV: R={corr_s2_rv:.3f}
  S₃-RV: R={corr_s3_rv:.3f}"""

    ax_hist.text(
        0.05,
        0.95,
        stats_text,
        transform=ax_hist.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot 3D de l'espace latent sauvegardé: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return True


# ==== MAIN EXECUTION ====

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AESTRA postprocessing: compute metrics, periodograms, and plots"
    )
    parser.add_argument(
        "experiment_name",
        type=str,
        help="Nom de l'expérience (dossier sous experiments/)",
    )
    parser.add_argument(
        "--p-inj",
        dest="p_inj",
        type=float,
        nargs="*",
        default=None,
        help="Périodes injectées (jours). Optionnel, une ou plusieurs valeurs.",
    )
    parser.add_argument(
        "--fap-threshold", type=float, default=0.01, help="Seuil FAP (par défaut 1%)."
    )
    parser.add_argument(
        "--exclude-width-frac",
        type=float,
        default=0.05,
        help="Largeur relative de la fenêtre autour de P_inj (±fraction).",
    )
    parser.add_argument(
        "--min-period",
        type=float,
        default=2.0,
        help="Période minimale (jours) pour les périodogrammes.",
    )
    parser.add_argument(
        "--max-period",
        type=float,
        default=None,
        help="Période maximale (jours). Par défaut ~1/3 de la base temporelle.",
    )
    parser.add_argument(
        "--n-periods",
        type=int,
        default=5000,
        help="Nombre de points de la grille de périodes (log).",
    )
    parser.add_argument(
        "--zoom-frac",
        type=float,
        default=0.15,
        help="Fraction pour le zoom autour de P_inj (±fraction).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Taille de batch pour la prédiction.",
    )
    parser.add_argument(
        "--perturbation-value",
        type=float,
        default=1.0,
        help="Amplitude de la perturbation appliquée à chaque dimension latente.",
    )
    args = parser.parse_args()

    experiment_name = args.experiment_name

    # Paths
    exp_dir = os.path.join("experiments", experiment_name)
    ckpt_path = os.path.join(exp_dir, "models", "aestra_final.pth")
    out_root = os.path.join(exp_dir, "postprocessing")
    fig_dir = os.path.join(out_root, "figures")
    fig_periodo_v = os.path.join(fig_dir, "periodograms", "v")
    fig_periodo_s = os.path.join(fig_dir, "periodograms", "s")
    fig_latent = os.path.join(fig_dir, "latent")
    fig_corr = os.path.join(fig_dir, "correlations")
    data_dir = os.path.join(out_root, "data")

    for d in [fig_periodo_v, fig_periodo_s, fig_latent, fig_corr, data_dir]:
        os.makedirs(d, exist_ok=True)

    # Load experiment
    exp = load_experiment_checkpoint(ckpt_path)
    model = exp["model"]
    dataset = exp["dataset"]

    # Predictions
    prediction_results = predict(
        model=model,
        dataset=dataset,
        batch_size=args.batch_size,
        perturbation_value=args.perturbation_value,
    )

    v_correct = prediction_results["all_vobs"]
    times_values = dataset.time_values

    # CCF analysis
    CCF_params = {
        "v_grid": np.arange(-20000, 20000, 250),
        "window_size_velocity": 820,
        "mask_type": "G2",
        "verbose": False,
        "batch_size": args.batch_size,
        "normalize": True,
    }

    print("Calcul des vitesses par CCF classique...")
    v_apparent, depth, span, fwhm = get_vapparent(dataset, CCF_params)
    print("Calcul des vitesses de référence...")
    v_ref, depth_ref, span_ref, fwhm_ref = get_vref(dataset, CCF_params)
    print("Calcul des vitesses corrigées par méthode traditionnelle...")
    v_traditionnal = get_vtraditionnal(v_apparent, depth, span, fwhm)

    # Prepare series
    v_series = {
        "v_apparent": v_apparent,
        "v_correct": v_correct,
        "v_traditionnal": v_traditionnal,
        "v_ref": v_ref,
    }

    # Store periodograms
    periodo_store = {}

    # CSV metrics rows
    csv_rows = []

    def add_metric(row_type, series, component, metric, value, P_inj=None):
        csv_rows.append(
            {
                "row_type": row_type,
                "series": series,
                "component": component,
                "metric": metric,
                "value": value if value is None or np.isscalar(value) else float(value),
                "P_inj": P_inj,
            }
        )

    # Helper to compute metrics for multiple injected periods using a single LS
    def compute_metrics_multi(
        periods, power, ls, P_inj_list, exclude_width_frac, fap_threshold
    ):
        metrics_list = []
        if not P_inj_list:
            return metrics_list
        for P_val in P_inj_list:
            mask = _get_planet_window_mask(periods, P_val, exclude_width_frac)
            fap, ratio, P_det, dP = _analyze_planet_detection(
                periods, power, mask, P_val, ls
            )
            n_sig = _count_significant_peaks_outside(
                periods, power, mask, ls, fap_threshold=fap_threshold
            )
            metrics_list.append(
                {
                    "fap_at_Pinj": fap,
                    "power_ratio": ratio,
                    "n_sig_peaks_outside": n_sig,
                    "P_detected": P_det,
                    "delta_P": dP,
                }
            )
        return metrics_list

    # Periodograms for RV series with optional zooms
    for name, y in v_series.items():
        print(f"Périodogramme: {name}")
        periods = _calculate_period_grid(
            times_values, args.min_period, args.max_period, args.n_periods
        )
        ls, power = _compute_lomb_scargle(
            times_values, y, periods, fit_mean=True, center_data=True
        )
        periodo_store[f"{name}_periods"] = periods
        periodo_store[f"{name}_power"] = power

        # Metrics for each injected period
        metrics_list = (
            compute_metrics_multi(
                periods,
                power,
                ls,
                args.p_inj if args.p_inj is not None else [],
                args.exclude_width_frac,
                args.fap_threshold,
            )
            if ls is not None
            else []
        )

        # Save full plot
        plot_periodogram(
            periods=periods,
            power=power,
            metrics=metrics_list if metrics_list else None,
            P_inj=args.p_inj,
            fap_threshold=args.fap_threshold,
            exclude_width_frac=args.exclude_width_frac,
            title=f"LS Periodogram - {name}",
            save_path=os.path.join(fig_periodo_v, f"{name}_periodogram.png"),
            show_plot=False,
        )

        # Save zooms for each injected period
        if args.p_inj:
            for i, P_val in enumerate(args.p_inj):
                xlim = (P_val * (1 - args.zoom_frac), P_val * (1 + args.zoom_frac))
                plot_periodogram(
                    periods=periods,
                    power=power,
                    metrics=metrics_list[i] if metrics_list else None,
                    P_inj=P_val,
                    fap_threshold=args.fap_threshold,
                    exclude_width_frac=args.exclude_width_frac,
                    title=f"LS Periodogram - {name} (zoom P={P_val:.3f} j)",
                    save_path=os.path.join(
                        fig_periodo_v, f"{name}_periodogram_zoom_P{P_val:.3f}.png"
                    ),
                    show_plot=False,
                    xlim=xlim,
                )

        # Register metrics in CSV rows
        if args.p_inj:
            for i, P_val in enumerate(args.p_inj):
                m = metrics_list[i] if i < len(metrics_list) else {}
                add_metric(
                    "periodogram",
                    name,
                    "rv",
                    "fap_at_Pinj",
                    m.get("fap_at_Pinj"),
                    P_val,
                )
                add_metric(
                    "periodogram",
                    name,
                    "rv",
                    "power_ratio",
                    m.get("power_ratio"),
                    P_val,
                )
                add_metric(
                    "periodogram",
                    name,
                    "rv",
                    "n_sig_peaks_outside",
                    m.get("n_sig_peaks_outside"),
                    P_val,
                )
                add_metric(
                    "periodogram", name, "rv", "delta_P", m.get("delta_P"), P_val
                )

    # Periodograms for each latent coordinate s_i
    all_s = prediction_results["all_s"]
    n_latent = all_s.shape[1]
    for i in range(n_latent):
        name = f"s_{i + 1}"
        y = all_s[:, i]
        print(f"Périodogramme: {name}")
        periods = _calculate_period_grid(
            times_values, args.min_period, args.max_period, args.n_periods
        )
        ls, power = _compute_lomb_scargle(
            times_values, y, periods, fit_mean=True, center_data=True
        )
        periodo_store[f"{name}_periods"] = periods
        periodo_store[f"{name}_power"] = power

        metrics_list = (
            compute_metrics_multi(
                periods,
                power,
                ls,
                args.p_inj if args.p_inj is not None else [],
                args.exclude_width_frac,
                args.fap_threshold,
            )
            if ls is not None
            else []
        )

        plot_periodogram(
            periods=periods,
            power=power,
            metrics=metrics_list if metrics_list else None,
            P_inj=args.p_inj,
            fap_threshold=args.fap_threshold,
            exclude_width_frac=args.exclude_width_frac,
            title=f"LS Periodogram - {name}",
            save_path=os.path.join(fig_periodo_s, f"{name}_periodogram.png"),
            show_plot=False,
        )

        if args.p_inj:
            for j, P_val in enumerate(args.p_inj):
                xlim = (P_val * (1 - args.zoom_frac), P_val * (1 + args.zoom_frac))
                plot_periodogram(
                    periods=periods,
                    power=power,
                    metrics=metrics_list[j] if j < len(metrics_list) else None,
                    P_inj=P_val,
                    fap_threshold=args.fap_threshold,
                    exclude_width_frac=args.exclude_width_frac,
                    title=f"LS Periodogram - {name} (zoom P={P_val:.3f} j)",
                    save_path=os.path.join(
                        fig_periodo_s, f"{name}_periodogram_zoom_P{P_val:.3f}.png"
                    ),
                    show_plot=False,
                    xlim=xlim,
                )

        if args.p_inj:
            for j, P_val in enumerate(args.p_inj):
                m = metrics_list[j] if j < len(metrics_list) else {}
                add_metric(
                    "periodogram",
                    name,
                    "latent",
                    "fap_at_Pinj",
                    m.get("fap_at_Pinj"),
                    P_val,
                )
                add_metric(
                    "periodogram",
                    name,
                    "latent",
                    "power_ratio",
                    m.get("power_ratio"),
                    P_val,
                )
                add_metric(
                    "periodogram",
                    name,
                    "latent",
                    "n_sig_peaks_outside",
                    m.get("n_sig_peaks_outside"),
                    P_val,
                )
                add_metric(
                    "periodogram", name, "latent", "delta_P", m.get("delta_P"), P_val
                )

    # Save NPZ with all periodograms
    npz_periodo_path = os.path.join(data_dir, "periodograms.npz")
    np.savez(npz_periodo_path, **periodo_store)

    # Latent distance distribution
    delta_s_rand, delta_s_aug = compute_latent_distances(
        prediction_results["all_s"], prediction_results["all_saug"], seed=42
    )

    # Save latent distance plot
    plot_latent_distance_distribution(
        delta_s_rand=delta_s_rand,
        delta_s_aug=delta_s_aug,
        save_path=os.path.join(fig_latent, "latent_distance_distribution.png"),
        show_plot=False,
    )

    # Also plot marginal distributions of latent coordinates
    plot_latent_marginal_distributions(
        prediction_results["all_s"],
        save_path=os.path.join(fig_latent, "latent_marginal_distributions.png"),
        show_plot=False,
    )

    # Save latent distances arrays
    np.savez(
        os.path.join(data_dir, "latent_distances.npz"),
        delta_s_rand=delta_s_rand,
        delta_s_aug=delta_s_aug,
    )

    # Add latent distance summary stats to CSV
    add_metric(
        "latent_distance",
        "latent",
        "delta_s",
        "delta_s_rand_mean",
        float(np.mean(delta_s_rand)),
    )
    add_metric(
        "latent_distance",
        "latent",
        "delta_s",
        "delta_s_rand_std",
        float(np.std(delta_s_rand)),
    )
    add_metric(
        "latent_distance",
        "latent",
        "delta_s",
        "delta_s_rand_median",
        float(np.median(delta_s_rand)),
    )
    add_metric(
        "latent_distance",
        "latent",
        "delta_s",
        "delta_s_aug_mean",
        float(np.mean(delta_s_aug)),
    )
    add_metric(
        "latent_distance",
        "latent",
        "delta_s",
        "delta_s_aug_std",
        float(np.std(delta_s_aug)),
    )
    add_metric(
        "latent_distance",
        "latent",
        "delta_s",
        "delta_s_aug_median",
        float(np.median(delta_s_aug)),
    )

    # Plot activity perturbation effects for all latent dimensions on the first spectrum
    y_act_original = prediction_results["all_yact"][0]
    latent_dim = prediction_results["all_yact_perturbed"].shape[0]
    y_act_perturbed_list = [
        prediction_results["all_yact_perturbed"][dim][0] for dim in range(latent_dim)
    ]
    wave = dataset.wavegrid.cpu().detach().numpy()

    plot_activity_perturbation(
        y_act_original=y_act_original,
        y_act_perturbed_list=y_act_perturbed_list,
        wavelength=wave,
        save_path=os.path.join(fig_latent, "activity_perturbations.png"),
        show_plot=False,
        wave_range=(5000, 5010),
    )

    # Correlation matrix plot and correlation metrics into CSV
    plot_correlation_matrix(
        v_apparent=v_apparent,
        v_correct=v_correct,
        v_traditionnal=v_traditionnal,
        v_ref=v_ref,
        depth=depth,
        span=span,
        fwhm=fwhm,
        latent_vectors=prediction_results["all_s"],
        save_path=os.path.join(fig_corr, "correlation_matrix.png"),
        show_plot=False,
    )

    # Also export correlation values into CSV rows
    x_vars = {
        "FWHM": fwhm,
        "Span": span,
        "Depth": depth,
    }
    for i in range(n_latent):
        x_vars[f"s_{i + 1}"] = prediction_results["all_s"][:, i]
    y_vars = {
        "v_apparent": v_apparent,
        "v_correct": v_correct,
        "v_traditionnal": v_traditionnal,
        "v_ref": v_ref,
    }
    for y_name, y_vals in y_vars.items():
        for x_name, x_vals in x_vars.items():
            corr = float(np.corrcoef(y_vals, x_vals)[0, 1])
            add_metric("correlation", y_name, x_name, "pearson_r", corr)

    # Optional latent 3D visualization if available
    if prediction_results["all_s"].shape[1] >= 3:
        try:
            plot_latent_space_3d(
                latent_s=prediction_results["all_s"],
                rv_values=v_correct,
                save_path=os.path.join(fig_latent, "latent_space_3d.png"),
                show_plot=False,
            )
        except Exception as e:
            print(f"Erreur lors du plot 3D de l'espace latent: {e}")

    # Write final metrics CSV
    csv_path = os.path.join(data_dir, "metrics.csv")
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["row_type", "series", "component", "metric", "value", "P_inj"],
        )
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    print(f"Tous les résultats ont été enregistrés dans: {out_root}")
