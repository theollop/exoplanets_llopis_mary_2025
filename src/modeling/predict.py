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
from src.ccf import get_full_ccf_analysis
from sklearn.linear_model import LinearRegression
import os
import glob
import csv
from src.plots_aestra import (
    plot_periodogram,
    plot_latent_distance_distribution,
    plot_latent_marginal_distributions,
    plot_activity_perturbation,
    plot_correlation_matrix,
    plot_latent_space_3d,
)
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
            "fap_at_PNj": None,
            "power_ratio": None,
            "n_sig_peaks_outside": 0,
            "P_detected": None,
            "delta_P": None,
        }
        return periods, power, metrics

    # Create planet window mask
    mask_planet_window = _get_planet_window_mask(periods, P_inj, exclude_width_frac)

    # Analyze planet detection
    fap_at_PNj, power_ratio, P_detected, delta_P = _analyze_planet_detection(
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
        "fap_at_PNj": fap_at_PNj,
        "power_ratio": power_ratio,
        "n_sig_peaks_outside": n_sig_peaks_outside,
        "P_detected": P_detected,
        "delta_P": delta_P,
    }

    return periods, power, metrics


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


def compute_latent_distances(all_s, all_saug, seed=None):
    """Compute latent distances for random pairs and augmented pairs."""
    n = all_s.shape[0]
    if seed is not None:
        np.random.seed(seed)

    inds = np.array([np.random.choice(n, size=2, replace=False) for _ in range(n)])
    delta_s_rand = np.linalg.norm(all_s[inds[:, 0]] - all_s[inds[:, 1]], axis=1)
    delta_s_aug = np.linalg.norm(all_s - all_saug, axis=1)
    return delta_s_rand, delta_s_aug


def rank_metrics_across_experiments(
    experiments_root: str = "experiments",
    metrics_rel_path: str = os.path.join("postprocessing", "data", "metrics.csv"),
    top_k: int = 5,
):
    """
    Parcourt experiments/*/postprocessing/data/metrics.csv et produit un classement simple par métrique.

    Règles:
    - pearson_r: on classe par |valeur| (plus grand est meilleur)
    - power_ratio: plus grand est meilleur
    - fap_at_PNj, delta_P, n_sig_peaks_outside: plus petit est meilleur

    Retourne un dict {metric: [ {experiment, value, row}, ... ]} trié, et imprime le top-k.
    """
    pattern = os.path.join(experiments_root, "*", metrics_rel_path)
    csv_paths = glob.glob(pattern)
    if not csv_paths:
        print(f"Aucun metrics.csv trouvé (pattern: {pattern})")
        return {}

    higher_is_better = {
        "pearson_r": True,  # on utilisera la valeur absolue
        "power_ratio": True,
    }

    bucket = {}
    for csv_file in csv_paths:
        exp_name = os.path.basename(
            os.path.dirname(os.path.dirname(os.path.dirname(csv_file)))
        )
        try:
            with open(csv_file, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    metric = row.get("metric")
                    val_str = row.get("value")
                    if not metric or val_str in (None, ""):
                        continue
                    try:
                        val = float(val_str)
                    except Exception:
                        continue
                    rank_val = abs(val) if metric == "pearson_r" else val
                    if np.isnan(rank_val):
                        continue
                    bucket.setdefault(metric, {}).setdefault(exp_name, []).append(
                        (rank_val, row)
                    )
        except Exception as e:
            print(f"Erreur en lisant {csv_file}: {e}")

    rankings = {}
    for metric, expmap in bucket.items():
        rows = []
        for exp_name, items in expmap.items():
            if not items:
                continue
            if higher_is_better.get(metric, False):
                best_val, best_row = max(items, key=lambda x: x[0])
            else:
                best_val, best_row = min(items, key=lambda x: x[0])
            rows.append({"experiment": exp_name, "value": best_val, "row": best_row[1]})
        rows.sort(key=lambda d: d["value"], reverse=higher_is_better.get(metric, False))
        rankings[metric] = rows

    for metric, ranked in rankings.items():
        print(f"\n=== Classement {metric} ===")
        for i, entry in enumerate(ranked[:top_k], start=1):
            try:
                print(f"{i:>2}. {entry['experiment']}: {entry['value']:.6g}")
            except Exception:
                print(f"{i:>2}. {entry['experiment']}: {entry['value']}")

    return rankings


def main(
    experiment_dir: str,
    fap_threshold: float = 0.01,
    exclude_width_frac: float = 0.05,
    min_period: float = 2.0,
    max_period: float = None,
    n_periods: int = 5000,
    zoom_frac: float = 0.15,
    batch_size: int = 64,
    perturbation_value: float = 1.0,
):
    # Paths
    ckpt_path = os.path.join(experiment_dir, "models", "aestra_final.pth")
    out_root = os.path.join(experiment_dir, "postprocessing")
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
        batch_size=batch_size,
        perturbation_value=perturbation_value,
    )

    v_correct = prediction_results["all_vobs"]
    times_values = dataset.time_values

    # CCF analysis
    CCF_params = {
        "v_grid": np.arange(-20000, 20000, 250),
        "window_size_velocity": 820,
        "mask_type": "G2",
        "verbose": False,
        "batch_size": batch_size,
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

    # Helper to compute metrics for multiple injected periods
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

    # === Periodograms for RV series ===
    for name, y in v_series.items():
        print(f"Périodogramme: {name}")
        periods = _calculate_period_grid(
            times_values, min_period, max_period, n_periods
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
                dataset.planets_periods or [],
                exclude_width_frac,
                fap_threshold,
            )
            if ls is not None
            else []
        )

        plot_periodogram(
            periods=periods,
            power=power,
            metrics=metrics_list if metrics_list else None,
            P_inj=dataset.planets_periods,
            fap_threshold=fap_threshold,
            exclude_width_frac=exclude_width_frac,
            title=f"LS Periodogram - {name}",
            save_path=os.path.join(fig_periodo_v, f"{name}_periodogram.png"),
            show_plot=False,
        )

        if dataset.planets_periods:
            for i, P_val in enumerate(dataset.planets_periods):
                m = metrics_list[i] if i < len(metrics_list) else {}
                add_metric(
                    "periodogram", name, "rv", "fap_at_PNj", m.get("fap_at_PNj"), P_val
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

    # === Periodograms for latent coordinates ===
    all_s = prediction_results["all_s"]
    n_latent = all_s.shape[1]
    for i in range(n_latent):
        name = f"s_{i + 1}"
        y = all_s[:, i]
        print(f"Périodogramme: {name}")
        periods = _calculate_period_grid(
            times_values, min_period, max_period, n_periods
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
                dataset.planets_periods or [],
                exclude_width_frac,
                fap_threshold,
            )
            if ls is not None
            else []
        )

        plot_periodogram(
            periods=periods,
            power=power,
            metrics=metrics_list if metrics_list else None,
            P_inj=None,
            fap_threshold=fap_threshold,
            exclude_width_frac=exclude_width_frac,
            title=f"LS Periodogram - {name}",
            save_path=os.path.join(fig_periodo_s, f"{name}_periodogram.png"),
            show_plot=False,
        )

        if dataset.planets_periods:
            for j, P_val in enumerate(dataset.planets_periods):
                m = metrics_list[j] if j < len(metrics_list) else {}
                add_metric(
                    "periodogram",
                    name,
                    "latent",
                    "fap_at_PNj",
                    m.get("fap_at_PNj"),
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

    # Save NPZ with periodograms
    np.savez(os.path.join(data_dir, "periodograms.npz"), **periodo_store)

    # Latent distance distribution
    delta_s_rand, delta_s_aug = compute_latent_distances(
        prediction_results["all_s"], prediction_results["all_saug"], seed=42
    )

    plot_latent_distance_distribution(
        delta_s_rand=delta_s_rand,
        delta_s_aug=delta_s_aug,
        save_path=os.path.join(fig_latent, "latent_distance_distribution.png"),
        show_plot=False,
    )

    plot_latent_marginal_distributions(
        prediction_results["all_s"],
        save_path=os.path.join(fig_latent, "latent_marginal_distributions.png"),
        show_plot=False,
    )

    np.savez(
        os.path.join(data_dir, "latent_distances.npz"),
        delta_s_rand=delta_s_rand,
        delta_s_aug=delta_s_aug,
    )

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

    # Activity perturbations
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

    # Correlation matrix
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

    # Export correlations to CSV
    x_vars = {"FWHM": fwhm, "Span": span, "Depth": depth}
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

    # Optional latent 3D plot
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

    # Write metrics CSV
    with open(os.path.join(data_dir, "metrics.csv"), mode="w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["row_type", "series", "component", "metric", "value", "P_inj"],
        )
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    print(f"Tous les résultats ont été enregistrés dans: {out_root}")


if __name__ == "__main__":
    main(
        experiment_dir="experiments/exp1",
    )
