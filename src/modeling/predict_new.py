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


# ==== CORE PREDICTION ====


def predict(model, dataset, batch_size=64, perturbation_value=1.0):
    """Extract latent vectors and RV values from model and dataset."""
    all_s = []
    all_saug = []
    rv_pred_obs_list = []
    rv_pred_aug_list = []
    rv_true_aug_list = []
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
            rv_true_aug_list.append(batch_voffset_true.cpu().detach().numpy())
            rv_pred_obs_list.append(batch_vobs_pred.cpu().detach().numpy())
            rv_pred_aug_list.append(batch_vaug_pred.cpu().detach().numpy())
            all_yact.append(batch_yact.cpu().detach().numpy())
            all_yact_aug.append(batch_yact_aug.cpu().detach().numpy())
            all_yobs_prime.append(batch_yobs_prime.cpu().detach().numpy())

    # Concatenate results
    all_s = np.concatenate(all_s, axis=0)
    all_saug = np.concatenate(all_saug, axis=0)
    rv_true_aug = np.concatenate(rv_true_aug_list, axis=0)
    rv_pred_obs = np.concatenate(rv_pred_obs_list, axis=0)
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
        "rv_pred_obs": rv_pred_obs,
        "rv_pred_aug": rv_pred_aug,
        "rv_true_aug": rv_true_aug,
        "all_rvs": rv_true_aug,  # backwards compatibility
        "all_yact": all_yact,
        "all_yact_aug": all_yact_aug,
        "all_yobs_prime": all_yobs_prime,
        "all_yact_perturbed": all_yact_perturbed_array,
    }


# ==== RV SERIES COMPUTATION ====


def get_vapparent(dataset: dict, CCFs_params: dict = None):
    """Compute v_apparent from CCF analysis on spectra."""
    res = get_full_ccf_analysis(
        spectra=dataset["spectra"], wavegrid=dataset["wavegrid"], **CCFs_params
    )
    return res["rv"], res["fwhm"], res["depth"], res["span"]


def get_vref(dataset: dict, CCFs_params: dict = None):
    """Compute v_ref from CCF analysis on activity-free spectra."""
    res = get_full_ccf_analysis(
        spectra=dataset["spectra_no_activity"],
        wavegrid=dataset["wavegrid"],
        **CCFs_params,
    )
    return res["rv"]


def get_vtraditionnal(v_apparent, fwhm, depth, bis_span):
    """Compute v_traditionnal using traditional activity correction."""
    X = np.column_stack([fwhm, depth, bis_span])
    model = LinearRegression().fit(X, v_apparent)
    v_pred = model.predict(X)
    return v_apparent - v_pred


# ==== ANALYSIS FUNCTIONS ====


def compute_latent_distances(all_s, all_saug, seed=None):
    """Compute latent distances for random pairs and augmented pairs."""
    n = all_s.shape[0]
    if seed is not None:
        np.random.seed(seed)

    inds = np.array([np.random.choice(n, size=2, replace=False) for _ in range(n)])
    delta_s_rand = np.linalg.norm(all_s[inds[:, 0]] - all_s[inds[:, 1]], axis=1)
    delta_s_aug = np.linalg.norm(all_s - all_saug, axis=1)
    return delta_s_rand, delta_s_aug


def compute_periodogram_metrics(
    y_values,
    time_values,
    P_inj=None,
    min_period=1.0,
    max_period=None,
    n_periods=10000,
    fap_threshold=0.01,
    exclude_width_frac=0.05,
    peak_prominence=None,
    ls_method="baluev",
    fit_mean=True,
    center_data=True,
):
    """Compute Lomb-Scargle periodogram with detection metrics."""
    t = np.asarray(time_values, dtype=float)
    y = np.asarray(y_values, dtype=float)

    if max_period is None:
        baseline = t.max() - t.min()
        max_period = max(baseline / 3.0, min_period * 1.1)

    periods = np.logspace(np.log10(min_period), np.log10(max_period), n_periods)
    frequencies = 1.0 / periods

    ls = LombScargle(t, y, fit_mean=fit_mean, center_data=center_data)
    power = ls.power(frequencies)

    if peak_prominence is None:
        peak_prominence = 0.5 * np.std(power)

    if P_inj is not None and P_inj > 0:
        mask_planet_window = np.abs(periods - P_inj) <= exclude_width_frac * P_inj
    else:
        mask_planet_window = np.zeros_like(periods, dtype=bool)

    # Planet detection metrics
    fap_at_Pinj = None
    power_ratio = None
    P_detected = None
    delta_P = None

    if P_inj is not None and P_inj > 0:
        idx_window = np.where(mask_planet_window)[0]
        if idx_window.size > 0:
            i_best_in_win = idx_window[np.argmax(power[idx_window])]
            P_detected = periods[i_best_in_win]
            P_planet_power = power[i_best_in_win]
            fap_at_Pinj = float(
                ls.false_alarm_probability(P_planet_power, method=ls_method)
            )

            if np.any(~mask_planet_window):
                max_outside = float(np.max(power[~mask_planet_window]))
                denom = max_outside if max_outside > 0 else np.nan
                power_ratio = float(P_planet_power / denom) if denom == denom else None

            delta_P = float(abs(P_detected - P_inj))

    # False alarms outside planet window
    power_out = power[~mask_planet_window]
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


def compute_correlations_for_series(all_s, y_series, fwhm, depth, bis_span):
    """Compute correlations between latent space, velocities, and activity indicators."""
    y = np.asarray(y_series).reshape(-1)
    S = all_s.shape[1]

    def safe_corr(a, b):
        a = np.asarray(a).reshape(-1)
        b = np.asarray(b).reshape(-1)
        if np.std(a) == 0 or np.std(b) == 0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    latent_vs_velocity = np.array([safe_corr(all_s[:, k], y) for k in range(S)])

    activity_vs_velocity = {
        "fwhm": safe_corr(fwhm, y),
        "depth": safe_corr(depth, y),
        "span": safe_corr(bis_span, y),
    }

    activity_vs_latent = np.zeros((3, S), dtype=float)
    for k in range(S):
        activity_vs_latent[0, k] = safe_corr(fwhm, all_s[:, k])
        activity_vs_latent[1, k] = safe_corr(depth, all_s[:, k])
        activity_vs_latent[2, k] = safe_corr(bis_span, all_s[:, k])

    return {
        "latent_vs_velocity": latent_vs_velocity,
        "activity_vs_velocity": activity_vs_velocity,
        "activity_vs_latent": activity_vs_latent,
    }


# ==== MCMC ANALYSIS ====


def run_mcmc_orbit(
    y_values,
    time_values,
    P_prior=None,
    P_bounds=None,
    nwalkers=32,
    nsteps=1000,
    burnin=500,
    seed=42,
):
    """MCMC inference for simple Keplerian model: v(t) = K * sin(2π t / P + phi)."""
    try:
        import importlib

        emcee = importlib.import_module("emcee")
    except Exception as e:
        raise RuntimeError("emcee is not installed. Please `pip install emcee`.") from e

    rng = np.random.default_rng(seed)
    t = np.asarray(time_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    y = y - np.mean(y)

    # Robust sigma estimate
    mad = np.median(np.abs(y - np.median(y)))
    sigma = 1.4826 * mad if mad > 0 else np.std(y) + 1e-8

    # Parameter bounds
    if P_bounds is None:
        baseline = t.max() - t.min()
        P_min = max(1.0, baseline / 1000)
        P_max = max(baseline / 3.0, P_min * 1.1)
    else:
        P_min, P_max = P_bounds

    if P_prior is None:
        P_prior = (P_min + P_max) / 2.0
    K_guess = 0.5 * (np.percentile(y, 97.5) - np.percentile(y, 2.5))

    def rv_model(t, P, K, phi):
        return K * np.sin(2.0 * np.pi * t / P + phi)

    def log_prior(theta):
        P, K, phi = theta
        if not (P_min < P < P_max):
            return -np.inf
        if not (0.0 < K < 10.0 * max(1.0, np.std(y))):
            return -np.inf
        if not (-np.pi <= phi <= np.pi):
            return -np.inf
        return 0.0

    def log_likelihood(theta):
        P, K, phi = theta
        model = rv_model(t, P, K, phi)
        resid = y - model
        return -0.5 * np.sum((resid / sigma) ** 2 + np.log(2 * np.pi * sigma**2))

    def log_probability(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta)

    # Initialize walkers
    p0 = np.vstack(
        [
            rng.normal(P_prior, 0.02 * P_prior, size=nwalkers),
            np.abs(
                rng.normal(
                    K_guess if K_guess > 0 else np.std(y),
                    0.5 * np.std(y),
                    size=nwalkers,
                )
            ),
            rng.uniform(-np.pi, np.pi, size=nwalkers),
        ]
    ).T

    sampler = emcee.EnsembleSampler(nwalkers, 3, log_probability)
    p0, _, _ = sampler.run_mcmc(p0, burnin, progress=False)
    sampler.reset()
    sampler.run_mcmc(p0, nsteps, progress=False)

    samples = sampler.get_chain(flat=True)

    def credible_interval(x):
        q16, q50, q84 = np.percentile(x, [16, 50, 84])
        return float(q50), float(q16), float(q84)

    med_P, lo_P, hi_P = credible_interval(samples[:, 0])
    med_K, lo_K, hi_K = credible_interval(samples[:, 1])
    med_phi, lo_phi, hi_phi = credible_interval(samples[:, 2])

    summary = {
        "P": {"median": med_P, "q16": lo_P, "q84": hi_P},
        "K": {"median": med_K, "q16": lo_K, "q84": hi_K},
        "phi": {"median": med_phi, "q16": lo_phi, "q84": hi_phi},
        "sigma_est": float(sigma),
        "nwalkers": nwalkers,
        "nsteps": nsteps,
        "burnin": burnin,
        "acceptance_fraction": float(np.mean(sampler.acceptance_fraction)),
    }

    return samples, summary


def circular_diff(a, b):
    """Circular difference for phase angles."""
    d = (a - b + np.pi) % (2 * np.pi) - np.pi
    return d


def summarize_mcmc_vs_truth(summary, truths):
    """Compute biases and posterior diagnostics."""
    out = {"parameters": {}, "diagnostics": {}}

    for key in ("P", "K", "phi"):
        s = summary[key]
        med, q16, q84 = s["median"], s["q16"], s["q84"]
        true = truths.get(key)

        if key == "phi":
            bias = float(circular_diff(med, true))
            width = float(circular_diff(q84, q16))
        else:
            bias = float(med - true) if true is not None else None
            width = float(q84 - q16)

        centered = None if true is None else (abs(bias) <= width / 2)
        rel_width = None
        if key in ("P", "K") and med != 0:
            rel_width = float(width / abs(med))

        out["parameters"][key] = {
            "median": med,
            "q16": q16,
            "q84": q84,
            "bias": bias,
            "centered": centered,
            "width": width,
            "relative_width": rel_width,
        }

    out["diagnostics"]["unimodal_expected"] = True
    return out


# ==== CSV EXPORT ====


def _to_float_safe(x):
    """Safe conversion to float."""
    try:
        return float(x)
    except Exception:
        return x


def build_csv_row(exp_name, results, correlation_summaries, latent_dim):
    """Flatten results into a single CSV row for model comparison."""
    row = {
        "experiment": exp_name,
        "P_inj": results.get("injection", {}).get("P"),
        "K_inj": results.get("injection", {}).get("K"),
        "phi_inj": results.get("injection", {}).get("phi"),
        "latent_dim": int(latent_dim),
    }

    # Periodogram metrics
    for label in ("v_correct", "v_apparent", "v_traditionnal", "v_ref"):
        metrics = results.get("periodograms", {}).get(label, {})
        for key in (
            "fap_at_Pinj",
            "power_ratio",
            "n_sig_peaks_outside",
            "P_detected",
            "delta_P",
        ):
            row[f"{label}_{key}"] = _to_float_safe(metrics.get(key))

    # MCMC summary
    mcmc = results.get("mcmc_v_correct", {})
    if "error" in mcmc:
        row["mcmc_error"] = mcmc.get("error")
    else:
        for p in ("P", "K", "phi"):
            if p in mcmc:
                row[f"mcmc_{p}_median"] = _to_float_safe(mcmc[p].get("median"))
                row[f"mcmc_{p}_q16"] = _to_float_safe(mcmc[p].get("q16"))
                row[f"mcmc_{p}_q84"] = _to_float_safe(mcmc[p].get("q84"))
        row["mcmc_sigma_est"] = _to_float_safe(mcmc.get("sigma_est"))
        row["mcmc_acceptance_fraction"] = _to_float_safe(
            mcmc.get("acceptance_fraction")
        )

        params = mcmc.get("parameters", {})
        for p in ("P", "K", "phi"):
            pm = params.get(p)
            if pm:
                row[f"mcmc_{p}_bias"] = _to_float_safe(pm.get("bias"))
                row[f"mcmc_{p}_centered"] = pm.get("centered")
                row[f"mcmc_{p}_width"] = _to_float_safe(pm.get("width"))
                row[f"mcmc_{p}_rel_width"] = _to_float_safe(pm.get("relative_width"))

    # Correlations
    for label, corr in correlation_summaries.items():
        lat = np.array(corr.get("latent_vs_velocity", []), dtype=float)
        if lat.size:
            row[f"{label}_corr_latent_max_abs"] = float(np.max(np.abs(lat)))
            row[f"{label}_corr_latent_max_k"] = int(np.argmax(np.abs(lat)) + 1)
            row[f"{label}_corr_latent_mean_abs"] = float(np.mean(np.abs(lat)))
            for i, v in enumerate(lat, start=1):
                row[f"{label}_corr_latent_s{i}"] = float(v)

        act_vel = corr.get("activity_vs_velocity", {})
        row[f"{label}_corr_y_fwhm"] = _to_float_safe(act_vel.get("fwhm"))
        row[f"{label}_corr_y_depth"] = _to_float_safe(act_vel.get("depth"))
        row[f"{label}_corr_y_span"] = _to_float_safe(act_vel.get("span"))

        act_lat = np.array(corr.get("activity_vs_latent", []), dtype=float)
        if act_lat.size:
            names = ["fwhm", "depth", "span"]
            for idx, name in enumerate(names):
                vec = act_lat[idx] if idx < act_lat.shape[0] else None
                if vec is not None and vec.size:
                    row[f"{label}_corr_{name}_vs_latent_max_abs"] = float(
                        np.max(np.abs(vec))
                    )
                    row[f"{label}_corr_{name}_vs_latent_max_k"] = int(
                        np.argmax(np.abs(vec)) + 1
                    )
                    row[f"{label}_corr_{name}_vs_latent_mean_abs"] = float(
                        np.mean(np.abs(vec))
                    )

    # Latent distances
    latent_dist = results.get("latent_distances", {})
    row["delta_s_rand_mean"] = _to_float_safe(latent_dist.get("delta_s_rand_mean"))
    row["delta_s_rand_std"] = _to_float_safe(latent_dist.get("delta_s_rand_std"))
    row["delta_s_aug_mean"] = _to_float_safe(latent_dist.get("delta_s_aug_mean"))
    row["delta_s_aug_std"] = _to_float_safe(latent_dist.get("delta_s_aug_std"))

    return row


def write_summary_csv(csv_path, row):
    """Create or update CSV with the given row."""
    import csv

    old_rows = []
    old_fields = []
    if os.path.exists(csv_path):
        try:
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                old_fields = reader.fieldnames or []
                old_rows = list(reader)
        except Exception:
            old_rows, old_fields = [], []

    new_fields = list(row.keys())
    union_fields = list(dict.fromkeys((old_fields or []) + new_fields))

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=union_fields)
        writer.writeheader()
        for r in old_rows:
            r_complete = {k: r.get(k, "") for k in union_fields}
            writer.writerow(r_complete)
        row_complete = {k: row.get(k, "") for k in union_fields}
        writer.writerow(row_complete)


# ==== MAIN PIPELINE ====


def main(
    experiment_name: str,
    device: str = "cuda",
    mcmc_steps: int = 1000,
    mcmc_burn: int = 500,
):
    """Run complete analysis pipeline for a given experiment."""
    exp_name = experiment_name

    # Setup directories
    exp_dir = os.path.join("experiments", exp_name)
    models_dir = os.path.join(exp_dir, "models")
    figures_dir = os.path.join(exp_dir, "figures")
    logs_dir = os.path.join(exp_dir, "logs")
    post_dir = os.path.join(figures_dir, "postprocessing")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(post_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    clear_gpu_memory()

    # Load model and dataset
    ckpt_path = os.path.join(models_dir, "aestra_final.pth")
    checkpoint = load_experiment_checkpoint(path=ckpt_path, device=device)
    model = checkpoint["model"]
    dataset = checkpoint["dataset"]

    # Run predictions
    prediction = predict(model, dataset, batch_size=32)
    v_correct = prediction["rv_pred_obs"]
    all_s = prediction["all_s"]
    all_saug = prediction["all_saug"]
    all_yact_perturbed = prediction["all_yact_perturbed"]

    # Extract dataset info
    time_values = dataset.time_values.cpu().detach().numpy()
    P_inj = float(dataset.planets_periods[0])
    Kp_inj = float(dataset.planets_amplitudes[0])
    Phi_inj = float(dataset.planets_phases[0])

    # CCF parameters
    CCFs_params = {
        "v_grid": np.arange(-20000, 20000, 100),
        "window_size_velocity": 820,
        "mask_type": "G2",
        "verbose": False,
        "batch_size": 64,
        "normalize": True,
    }

    # Compute alternative RV series
    raw_dataset = np.load(dataset.dataset_filepath, allow_pickle=True)
    v_apparent, fwhm, depth, bis_span = get_vapparent(raw_dataset, CCFs_params)
    v_traditionnal = get_vtraditionnal(v_apparent, fwhm, depth, bis_span)
    v_ref = get_vref(raw_dataset, CCFs_params)

    # Initialize results
    results = {
        "experiment": exp_name,
        "injection": {"P": P_inj, "K": Kp_inj, "phi": Phi_inj},
        "periodograms": {},
    }

    # Periodogram analysis for all RV series
    rv_series = [
        (v_correct, "v_correct"),
        (v_apparent, "v_apparent"),
        (v_traditionnal, "v_traditionnal"),
        (v_ref, "v_ref"),
    ]

    for y, label in rv_series:
        periods, power, metrics = compute_periodogram_metrics(
            y_values=y,
            time_values=time_values,
            P_inj=P_inj,
            min_period=10.0,
            n_periods=10000,
            fap_threshold=0.01,
            exclude_width_frac=0.1,
        )
        results["periodograms"][label] = metrics
        plot_periodogram_analysis(
            periods=periods,
            power=power,
            metrics=metrics,
            P_inj=P_inj,
            title=f"Lomb–Scargle Periodogram of {label}",
            save_path=os.path.join(post_dir, f"periodogram_{label}.png"),
        )

    # Correlation analysis and latent plots
    correlation_summaries = {}
    for y, label in rv_series:
        corr = compute_correlations_for_series(all_s, y, fwhm, depth, bis_span)
        correlation_summaries[label] = {
            "latent_vs_velocity": corr["latent_vs_velocity"].tolist(),
            "activity_vs_velocity": corr["activity_vs_velocity"],
            "activity_vs_latent": corr["activity_vs_latent"].tolist(),
        }
        plot_latent_analysis_for_series(
            all_s, y, label, os.path.join(post_dir, f"latent_{label}"), corr
        )

    results["correlations"] = correlation_summaries

    # Latent distance analysis
    delta_s_rand, delta_s_aug = compute_latent_distances(all_s, all_saug, seed=42)
    plot_latent_distance_distribution(
        delta_s_rand,
        delta_s_aug,
        os.path.join(post_dir, "latent_distances_distribution.png"),
    )

    # Perturbed activations plot
    plot_yact_perturbed(
        all_yact_perturbed,
        dataset.wavegrid.cpu().detach().numpy(),
        os.path.join(post_dir, "yact_perturbed_analysis.png"),
    )

    # Store latent distances
    results["latent_distances"] = {
        "delta_s_rand_mean": float(np.mean(delta_s_rand)),
        "delta_s_rand_std": float(np.std(delta_s_rand)),
        "delta_s_aug_mean": float(np.mean(delta_s_aug)),
        "delta_s_aug_std": float(np.std(delta_s_aug)),
    }

    # MCMC orbit inference
    baseline = time_values.max() - time_values.min()
    P_bounds = (10.0, max(baseline / 3.0, 11.0))

    try:
        samples, base_summary = run_mcmc_orbit(
            y_values=v_correct,
            time_values=time_values,
            P_prior=P_inj,
            P_bounds=P_bounds,
            nwalkers=32,
            nsteps=mcmc_steps,
            burnin=mcmc_burn,
            seed=42,
        )
        truths = {"P": P_inj, "K": Kp_inj, "phi": Phi_inj}
        comp = summarize_mcmc_vs_truth(base_summary, truths)
        mcmc_summary = {**base_summary, **comp}

        plot_mcmc_posteriors(
            samples, truths, os.path.join(post_dir, "mcmc_posteriors_v_correct.png")
        )
    except RuntimeError as e:
        mcmc_summary = {"error": str(e)}

    results["mcmc_v_correct"] = mcmc_summary

    # Save results
    with open(os.path.join(logs_dir, "tests_summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    csv_row = build_csv_row(exp_name, results, correlation_summaries, all_s.shape[1])
    write_summary_csv(os.path.join(post_dir, "tests_summary.csv"), csv_row)

    print(f"Analysis completed. Results in {logs_dir}, figures in {post_dir}")


def main_from_checkpoint(checkpoint_path: str):
    """Backward compatibility function for direct checkpoint loading."""
    exp_data = load_experiment_checkpoint(path=checkpoint_path)
    model = exp_data["model"]
    dataset = exp_data["dataset"]
    _ = predict(model, dataset)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AESTRA automated analysis pipeline")
    parser.add_argument(
        "--experiment", "-e", required=True, help="Experiment name under experiments/"
    )
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--mcmc-steps", type=int, default=1000, help="MCMC steps")
    parser.add_argument("--mcmc-burn", type=int, default=500, help="MCMC burn-in")
    args = parser.parse_args()

    main(args.experiment, args.device, args.mcmc_steps, args.mcmc_burn)
