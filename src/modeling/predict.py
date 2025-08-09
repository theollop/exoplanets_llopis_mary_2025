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
import os
import json
import argparse
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks

# Local imports
from src.modeling.train import load_experiment_checkpoint
from torch.utils.data import DataLoader
from src.dataset import generate_collate_fn
from src.utils import clear_gpu_memory
from src.ccf import get_full_ccf_analysis
from src.plots_aestra_clean import (
    plot_periodogram_analysis,
    plot_mcmc_posteriors,
    plot_latent_distance_distribution,
    plot_yact_perturbed,
    plot_latent_analysis_for_series,
)
import matplotlib.pyplot as plt

# ==== PERIODOGRAM ANALYSIS ====


def compute_periodogram_metrics(
    y_values,
    time_values,
    P_inj=None,
    min_period=1.0,
    max_period=None,  # si None → (tmax - tmin)/3
    n_periods=10000,
    fap_threshold=0.01,  # 1% par défaut
    exclude_width_frac=0.05,  # fenêtre ±5% autour de P_inj
    peak_prominence=None,  # None → auto (0.5 * std(power))
    ls_method="baluev",  # méthode FAP: "baluev" (rapide, robuste)
    fit_mean=True,
    center_data=True,
):
    """
    Calcule le périodogramme Lomb–Scargle et un set de métriques de détection.

    Paramètres
    ----------
    y_values : array (N,)
        Série RV (ou résidus) à analyser.
    time_values : array (N,)
        Temps (mêmes unités que celles attendues par LS, typiquement jours).
    P_inj : float | None
        Période injectée (jour). Si None, les métriques liées à P_inj seront None.
    min_period, max_period : float
        Bornes de période (jours). Si max_period=None → (tmax - tmin)/3.
    n_periods : int
        Taille de la grille logarithmique en période.
    fap_threshold : float
        Seuil FAP pour compter les pics “significatifs”.
    exclude_width_frac : float
        Largeur relative de la fenêtre à exclure autour de P_inj pour les stats “hors planète”.
        Exemple: 0.05 → exclut [P_inj*(1-0.05), P_inj*(1+0.05)].
    peak_prominence : float | None
        Prominence pour détecter des pics (scipy.signal.find_peaks).
        None → 0.5 * std(power).
    ls_method : str
        Méthode de FAP (‘baluev’, ‘naive’, ‘bootstrap’). ‘baluev’ recommandé.
    fit_mean, center_data : bool
        Options LombScargle.

    Renvoie
    -------
    periods : (M,)
        Grille de périodes (jours).
    power : (M,)
        Puissance LS correspondante.
    metrics : dict
        {
          "fap_at_Pinj": float|None,
          "power_ratio": float|None,   # P_planète / max(P_hors_planète)
          "n_sig_peaks_outside": int,  # nb de pics hors fenêtre avec FAP < fap_threshold
          "P_detected": float|None,    # meilleur pic dans la fenêtre de P_inj (si P_inj fourni)
          "delta_P": float|None        # |P_detected - P_inj|
        }
    """
    t = np.asarray(time_values, dtype=float)
    y = np.asarray(y_values, dtype=float)

    # borne supérieure par défaut : ~1/3 de la base temporelle
    if max_period is None:
        baseline = t.max() - t.min()
        max_period = max(baseline / 3.0, min_period * 1.1)

    # grille log en période → fréquences
    periods = np.logspace(np.log10(min_period), np.log10(max_period), n_periods)
    frequencies = 1.0 / periods

    # Lomb–Scargle
    ls = LombScargle(t, y, fit_mean=fit_mean, center_data=center_data)
    power = ls.power(frequencies)

    # Détection auto de prominence si non fournie
    if peak_prominence is None:
        peak_prominence = 0.5 * np.std(power)

    # Fenêtre autour de P_inj (pour exclure la planète des stats “hors planète”)
    if P_inj is not None and P_inj > 0:
        mask_planet_window = np.abs(periods - P_inj) <= exclude_width_frac * P_inj
    else:
        mask_planet_window = np.zeros_like(periods, dtype=bool)

    # --- Mesures autour de P_inj ---
    fap_at_Pinj = None
    power_ratio = None
    P_detected = None
    delta_P = None

    if P_inj is not None and P_inj > 0:
        # Pic détecté le plus puissant DANS la fenêtre de la planète
        idx_window = np.where(mask_planet_window)[0]
        if idx_window.size > 0:
            i_best_in_win = idx_window[np.argmax(power[idx_window])]
            P_detected = periods[i_best_in_win]
            P_planet_power = power[i_best_in_win]
            # FAP au niveau de puissance du pic planétaire
            fap_at_Pinj = float(
                ls.false_alarm_probability(P_planet_power, method=ls_method)
            )

            # Rapport de puissance : P_planète / max(P_hors_planète)
            if np.any(~mask_planet_window):
                max_outside = float(np.max(power[~mask_planet_window]))
                # éviter division par ~0
                denom = max_outside if max_outside > 0 else np.nan
                power_ratio = (
                    float(P_planet_power / denom) if denom == denom else None
                )  # nan-check
            else:
                power_ratio = None

            # Erreur de période
            delta_P = float(abs(P_detected - P_inj))
        else:
            # si la fenêtre est trop étroite pour contenir un point de grille
            fap_at_Pinj = None
            power_ratio = None
            P_detected = None
            delta_P = None

    # --- Nombre de pics “significatifs” hors planète (FAP < seuil) ---
    # On détecte d’abord les pics hors fenêtre, puis on évalue leur FAP individuelle
    periods_out = periods[~mask_planet_window]
    power_out = power[~mask_planet_window]

    # Attention: find_peaks travaille sur un signal indexé régulièrement.
    # Ici la grille en période est régulière en log, mais ça reste OK pour détecter des maxima locaux.
    peaks_out, _ = find_peaks(power_out, prominence=peak_prominence)
    if peaks_out.size:
        faps_out = ls.false_alarm_probability(power_out[peaks_out], method=ls_method)
        n_sig_peaks_outside = int(np.sum(faps_out < fap_threshold))
    else:
        n_sig_peaks_outside = 0

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
    metrics,
    P_inj=None,
    fap_threshold=0.01,
    exclude_width_frac=0.05,
    peak_prominence=None,
    title="Lomb–Scargle Periodogram",
):
    """
    Affiche le périodogramme avec:
    - ligne de seuil FAP (ex: 1%),
    - bande d'exclusion autour de P_inj,
    - marquage du meilleur pic détecté,
    - annotation des métriques principales.

    Note: Utilise les données déjà calculées par compute_periodogram_metrics
    """
    # Calcul du seuil FAP approximatif basé sur la puissance maximale
    # (approximation raisonnable pour l'affichage)
    max_power = np.max(power)
    fap_level = max_power * fap_threshold  # Approximation simple pour l'affichage

    # Détection de pics hors fenêtre pour marquage visuel
    if P_inj is not None:
        mask_excl = np.abs(periods - P_inj) <= exclude_width_frac * P_inj
    else:
        mask_excl = np.zeros_like(periods, dtype=bool)

    if peak_prominence is None:
        peak_prominence = 0.5 * np.std(power)

    p_out = power[~mask_excl]
    per_out = periods[~mask_excl]
    peaks_out, _ = find_peaks(p_out, prominence=peak_prominence)

    # Prépare la figure
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.semilogx(periods, power, lw=1.6)
    ax.set_xlabel("Période [jours]")
    ax.set_ylabel("Puissance Lomb–Scargle")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)

    # Seuil FAP
    ax.axhline(
        fap_level, ls="--", lw=1.2, label=f"Seuil FAP = {int(fap_threshold * 100)}%"
    )

    # Bande d’exclusion autour de P_inj
    if P_inj is not None and P_inj > 0:
        ax.axvspan(
            P_inj * (1 - exclude_width_frac),
            P_inj * (1 + exclude_width_frac),
            alpha=0.15,
            label=f"Bande autour de $P_{{inj}}$ = {P_inj:.4g} j",
        )

        # Meilleur pic détecté dans la fenêtre (issu des métriques)
        if metrics.get("P_detected") is not None:
            P_detected = metrics["P_detected"]
            # positionne un marqueur sur le pic détecté
            idx = np.argmin(np.abs(periods - P_detected))
            ax.plot(
                periods[idx],
                power[idx],
                marker="o",
                ms=7,
                mec="k",
                mfc="none",
                label=f"Pic détecté @ {P_detected:.4g} j",
            )
            # Ligne verticale P_inj (référence)
            ax.axvline(P_inj, color="k", lw=1.0, alpha=0.5)
    else:
        P_detected = None

    # Marque les pics “hors planète”
    if peaks_out.size:
        ax.plot(
            per_out[peaks_out], p_out[peaks_out], "x", ms=6, label="Pics (hors planète)"
        )

    # Petit encart avec les métriques
    lines = []
    if metrics.get("fap_at_Pinj") is not None:
        lines.append(f"FAP @ P_inj: {metrics['fap_at_Pinj']:.3g}")
    if metrics.get("power_ratio") is not None:
        lines.append(f"Power ratio: {metrics['power_ratio']:.3g}")
    if metrics.get("n_sig_peaks_outside") is not None:
        lines.append(f"# pics FAP<1% (hors): {metrics['n_sig_peaks_outside']}")
    if metrics.get("delta_P") is not None:
        lines.append(f"ΔP: {metrics['delta_P']:.3g} j")
    if lines:
        txt = "\n".join(lines)
        ax.text(
            0.02,
            0.98,
            txt,
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7", alpha=0.9),
            fontsize=10,
        )

    ax.legend(loc="best", frameon=True)
    plt.tight_layout()
    plt.show()


# ==== CORE PREDICTION ====


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


if __name__ == "__main__":
    experiment_name = "aestra_local_experiment_range_50"

    exp = load_experiment_checkpoint(
        "experiments/" + experiment_name + "/models/aestra_final.pth"
    )

    model = exp["model"]
    dataset = exp["dataset"]

    prediction_results = predict(
        model=model,
        dataset=dataset,
    )

    rvs = prediction_results["all_vobs"]

    times_values = dataset.time_values

    # Periodogram analysis
    periods, power, metrics = compute_periodogram_metrics(
        y_values=rvs,
        time_values=times_values,
        P_inj=100.0,  # Example injected period
        min_period=1.0,
        max_period=None,  # Automatically set to ~1/3 of the time range
        n_periods=10000,
        fap_threshold=0.01,  # 1% by default
        exclude_width_frac=0.05,  # ±5% around P_inj
        peak_prominence=None,  # Automatically set to 0.5 * std(power)
        ls_method="baluev",  # Fast and robust FAP method
        fit_mean=True,
        center_data=True,
    )

    # Plot periodogram
    plot_periodogram(
        periods=periods,
        power=power,
        metrics=metrics,
        P_inj=100.0,  # Example injected period
        fap_threshold=0.01,
        exclude_width_frac=0.05,
        peak_prominence=None,  # Automatically set to 0.5 * std(power)
        title="Lomb–Scargle Periodogram Analysis",
    )
