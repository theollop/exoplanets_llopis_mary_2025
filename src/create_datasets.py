from __future__ import annotations

import gc
import os
import pickle
import re
import sys
import tempfile
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from scipy.ndimage import uniform_filter1d

from src.interpolate import shift_spectra_linear
from src.rassine import normalize_with_rassine

# ============================================================
# ---------------------- Config types ------------------------
# ============================================================


@dataclass
class IndexSplit:
    train_start: int
    train_end: int
    val_start: Optional[int] = None
    val_end: Optional[int] = None


@dataclass
class PlanetParams:
    amplitudes: Sequence[float]
    periods: Sequence[float]
    phases: Sequence[float]


@dataclass
class PreprocessParams:
    wavemin: float
    wavemax: float
    downscaling_factor: int = 2
    smooth_after_downscaling: bool = False
    smooth_kernel_size: int = 3


@dataclass
class NoiseParams:
    add_photon_noise: bool = False
    snr_target: Optional[float] = None
    seed: Optional[int] = None


# ============================================================
# ---------------------- Utils génériques --------------------
# ============================================================


def _slugify(text: str, max_len: int = 80) -> str:
    text = (
        unicodedata.normalize("NFKD", str(text))
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    text = re.sub(r"[^a-zA-Z0-9._-]+", "-", text).strip("-._").lower()
    return (text[:max_len] or "dataset").strip("-._")


def _fmt_num(x: float) -> str:
    xf = float(x)
    return str(int(xf)) if xf.is_integer() else f"{xf:g}".replace(".", "p")


def _fmt_list(lst: Sequence[float]) -> str:
    return "+".join(_fmt_num(x) for x in list(lst))


def auto_filename(
    output_dir: str,
    n_train: int,
    n_val: int,
    wavemin: float,
    wavemax: float,
    prep: PreprocessParams,
    noise: NoiseParams,
    planets: Optional[PlanetParams],
) -> str:
    bits = [f"nst{n_train}", f"nsv{n_val}", f"{int(wavemin)}-{int(wavemax)}"]
    if prep.downscaling_factor and prep.downscaling_factor > 1:
        bits.append(f"dx{prep.downscaling_factor}")
    if prep.smooth_after_downscaling:
        bits.append(f"sm{prep.smooth_kernel_size}")
    if noise.add_photon_noise:
        bits.append(
            "noise" if noise.snr_target is None else f"snr{_fmt_num(noise.snr_target)}"
        )
    if (
        planets is not None
        and len(planets.periods)
        and len(planets.amplitudes)
        and len(planets.phases)
    ):
        bits += [
            f"P{_fmt_list(planets.periods)}",
            f"K{_fmt_list(planets.amplitudes)}",
            f"Phi{_fmt_list(planets.phases)}",
        ]
    base = _slugify("soapgpu_" + "_".join(bits), max_len=80)
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"{base}.npz")


# ============================================================
# ---------------------- I/O & slicing -----------------------
# ============================================================


def load_template_npz(tmp_filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    spec_data = np.load(tmp_filepath)
    return spec_data["spec"], spec_data["wavelength"]


def build_mask(wavegrid: np.ndarray, wavemin: float, wavemax: float) -> np.ndarray:
    if wavemin is None:
        wavemin = float(wavegrid.min())
    if wavemax is None:
        wavemax = float(wavegrid.max())
    return (wavegrid >= wavemin) & (wavegrid <= wavemax)


def load_spectra_selection(
    spectra_filepath: str,
    mask: np.ndarray,
    split: IndexSplit,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, int]:
    """Retourne spectra_train, spectra_val (ou None), time_values_sel_concat, n_file_total."""
    with h5py.File(spectra_filepath, "r") as f:
        n_file_total = f["spec_cube"].shape[0]
        spectra_train = f["spec_cube"][split.train_start : split.train_end, :][:, mask]
        if split.val_start is not None and split.val_end is not None:
            spectra_val = f["spec_cube"][split.val_start : split.val_end, :][:, mask]
        else:
            spectra_val = None

    n_tot_indices = np.arange(n_file_total)
    if "filtered" in spectra_filepath:
        # cohérence temporelle si fichier déjà filtré
        removed = np.array([246, 249, 1196, 1453, 2176], dtype=int)
        n_tot_indices = np.delete(n_tot_indices, removed)

    t_train = n_tot_indices[split.train_start : split.train_end]
    t_val = (
        n_tot_indices[split.val_start : split.val_end]
        if split.val_start is not None and split.val_end is not None
        else np.array([], dtype=t_train.dtype)
    )
    t_all = np.concatenate([t_train, t_val])
    spectra_sel = (
        spectra_train
        if spectra_val is None
        else np.concatenate([spectra_train, spectra_val], axis=0)
    )
    assert t_all.shape[0] == spectra_sel.shape[0], "Mismatch time vs spectra selection"
    return spectra_sel, spectra_val, t_all, n_file_total


# ============================================================
# ---------------------- Pré-traitements ---------------------
# ============================================================


def downscale_mean_1d(x: np.ndarray, factor: int) -> np.ndarray:
    n_bins = x.size // factor
    return x[: n_bins * factor].reshape(n_bins, factor).mean(axis=1)


def downscale_mean_2d(X: np.ndarray, factor: int) -> Tuple[np.ndarray, int]:
    """X shape (N, P) -> (N, P//factor)"""
    N, P = X.shape
    n_bins = P // factor
    X_ds = X[:, : n_bins * factor].reshape(N, n_bins, factor).mean(axis=2)
    return X_ds, n_bins


def maybe_smooth_inplace(X: np.ndarray, size: int):
    for i in range(X.shape[0]):
        X[i] = uniform_filter1d(X[i], size=size, mode="reflect")


def compute_activity_pre_noise(
    spectra_ds: np.ndarray, template_ds: np.ndarray
) -> np.ndarray:
    return spectra_ds - template_ds


def _add_photon_noise(spectrum, snr_target=None, default_snr=100.0):
    """
    Ajoute du bruit de Poisson en conservant l'échelle du spectre.
    - Si snr_target est donné: SNR(médiane) ~= snr_target
    - Sinon: SNR(médiane) ~= default_snr
    """
    spec = np.asarray(spectrum, dtype=float)
    # éviter lambda=0 et valeurs négatives
    spec = np.maximum(spec, 1e-12)

    # flux de référence pour fixer le SNR (médiane plus robuste que la moyenne)
    mu = (
        float(np.median(spec)) if np.isfinite(np.median(spec)) else float(np.mean(spec))
    )

    if mu <= 0:
        # fallback: pas de bruit si flux pathologique
        return spec.copy()

    S = float(snr_target) if (snr_target is not None) else float(default_snr)
    S = max(S, 1.0)  # borne basse raisonnable

    # facteur de mise à l'échelle vers des "comptes"
    k = (S * S) / mu  # => SNR(médiane) ≈ S

    lam = k * spec
    # prudence: éviter les lambdas énormes qui peuvent ralentir/overflow
    # (optionnel) lam = np.clip(lam, 0, 1e8)

    counts = np.random.poisson(lam)
    noisy = counts / k  # revenir à l'échelle originale

    return noisy


def add_photon_noise_batch(
    X: np.ndarray, snr_target: Optional[float], seed: Optional[int] = None
):
    if seed is not None:
        np.random.seed(int(seed))
    for i in range(X.shape[0]):
        X[i] = _add_photon_noise(X[i], snr_target)


def _normalize_spectrum_with_rassine(wave, flux, config=None):
    """
    Fonction helper pour normaliser un spectre avec Rassine.

    Parameters
    ----------
    wave : np.ndarray
        Grille de longueurs d'onde
    flux : np.ndarray
        Flux du spectre
    config : dict
        Configuration Rassine

    Returns
    -------
    np.ndarray
        Spectre normalisé
    """
    # Ajouter le chemin Rassine si nécessaire
    rassine_path = os.path.join(os.path.dirname(__file__), "..", "Rassine_public")
    rassine_path = os.path.abspath(rassine_path)
    if rassine_path not in sys.path:
        sys.path.insert(0, rassine_path)

    try:
        if config is None:
            config = {
                "axes_stretching": "auto_0.3",
                "vicinity_local_max": 5,
                "smoothing_box": 3,
                "smoothing_kernel": "gaussian",
                "fwhm_ccf": "auto",
                "CCF_mask": "master",
                "RV_sys": 0,
                "mask_telluric": [[6275, 6330], [6470, 6577], [6866, 8000]],
                "mask_broadline": [[3960, 3980], [6560, 6562], [10034, 10064]],
                "min_radius": "auto",
                "max_radius": "auto",
                "model_penality_radius": "poly_0.5",
                "denoising_dist": 3,
                "number_of_cut": 2,
                "number_of_cut_outliers": 1,
                "interpol": "linear",
                "feedback": False,
                "only_print_end": True,
                "plot_end": False,
                "save_last_plot": False,
                "outputs_interpolation_save": "linear",
                "outputs_denoising_save": "undenoised",
                "light_file": True,
                "speedup": 0.5,
                "float_precision": "float64",
                "column_wave": "wave",
                "column_flux": "flux",
                "synthetic_spectrum": False,
                "anchor_file": "",
            }
        # Préparation des données pour Rassine
        spectrum_data = {config["column_wave"]: wave, config["column_flux"]: flux}

        # Créer un fichier temporaire pour le spectre
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".p", delete=False
        ) as tmp_file:
            pickle.dump(spectrum_data, tmp_file)
            tmp_spectrum_path = tmp_file.name

        try:
            # Appel simplifié de l'algorithme Rassine
            # Simulation de l'algorithme principal sans interface graphique

            # Charger les données
            spectrei = np.array(spectrum_data[config["column_flux"]])
            grid = np.array(spectrum_data[config["column_wave"]])

            # Tri par longueur d'onde
            sorting = grid.argsort()
            grid = grid[sorting]
            spectrei = spectrei[sorting]
            spectrei[spectrei < 0] = 0  # Remplacer les valeurs négatives

            # Calcul de la normalisation
            len_x = grid.max() - grid.min()
            len_y = spectrei.max() - spectrei.min()
            normalisation = float(len_y) / float(len_x)
            spectre = spectrei / normalisation

            # Calcul des maxima locaux avec rolling quantile
            dgrid = (grid[1] - grid[0]) / 5

            # Sigma clipping itératif conservatif
            import pandas as pd

            for iteration in range(2):  # Réduction à 2 itérations
                maxi_roll = np.ravel(
                    pd.DataFrame(spectre)
                    .rolling(
                        int(50 / dgrid), min_periods=1, center=True
                    )  # Fenêtre plus petite
                    .quantile(0.95)  # Quantile plus conservatif (95% au lieu de 99%)
                )
                Q3 = np.ravel(
                    pd.DataFrame(spectre)
                    .rolling(
                        int(3 / dgrid), min_periods=1, center=True
                    )  # Fenêtre plus petite
                    .quantile(0.75)
                )
                Q2 = np.ravel(
                    pd.DataFrame(spectre)
                    .rolling(
                        int(3 / dgrid), min_periods=1, center=True
                    )  # Fenêtre plus petite
                    .quantile(0.50)
                )
                IQ = 2 * (Q3 - Q2)
                sup = Q3 + 2.0 * IQ  # Seuil plus élevé (2.0 au lieu de 1.5)

                mask = (spectre > sup) & (spectre > maxi_roll)
                if np.sum(mask) == 0:
                    break
                spectre[mask] = Q2[mask]

            # Détection des maxima locaux conservative
            from scipy.signal import find_peaks

            peaks, _ = find_peaks(
                spectre, height=np.percentile(spectre, 70)
            )  # Seuil plus élevé

            # Si pas assez de pics, baisser progressivement le seuil
            if len(peaks) < 15:
                peaks, _ = find_peaks(spectre, height=np.percentile(spectre, 60))
            if len(peaks) < 10:
                peaks, _ = find_peaks(spectre, height=np.percentile(spectre, 50))

            # Sélectionner des points d'ancrage de manière plus conservative
            n_anchors = min(
                len(peaks), max(15, len(grid) // 80)
            )  # Moins de points d'ancrage
            if len(peaks) > n_anchors:
                indices = np.linspace(0, len(peaks) - 1, n_anchors, dtype=int)
                peaks = peaks[indices]

            wave_anchors = grid[peaks]
            flux_anchors = spectre[peaks] * normalisation

            # Interpolation conservative pour le continuum
            from scipy.interpolate import interp1d

            if len(wave_anchors) >= 2:
                # Utiliser interpolation linéaire par défaut (plus conservative)
                interpolator = interp1d(
                    wave_anchors,
                    flux_anchors,
                    kind="linear",  # Toujours linéaire pour être conservatif
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                continuum = interpolator(grid)

                # Éviter les valeurs aberrantes du continuum avec des limites plus strictes
                continuum = np.clip(
                    continuum,
                    np.percentile(
                        spectrei, 10
                    ),  # Limites plus strictes (10% au lieu de 5%)
                    np.percentile(spectrei, 90),  # et 90% au lieu de 99%
                )

                # Normalisation finale conservative
                normalized_spectrum = spectrei / continuum

                # Nettoyage avec des limites plus strictes
                normalized_spectrum = np.clip(
                    normalized_spectrum, 0.2, 1.8
                )  # Plus conservatif

            else:
                # Fallback: normalisation simple par la médiane
                normalized_spectrum = spectrei / np.median(spectrei)

            return normalized_spectrum

        finally:
            # Nettoyer le fichier temporaire
            try:
                os.unlink(tmp_spectrum_path)
            except Exception:
                pass

    except Exception as e:
        print(
            f"⚠️  Erreur Rassine pour un spectre, utilisation de normalisation simple: {e}"
        )
        # Fallback: normalisation simple
        return flux / np.median(flux)


# ============================================================
# ---------------------- Injection planètes ------------------
# ============================================================


def compute_velocities(time_values: np.ndarray, planets: PlanetParams) -> np.ndarray:
    v = np.zeros(len(time_values), dtype=float)
    for Kp, P, phase in zip(planets.amplitudes, planets.periods, planets.phases):
        v += Kp * np.sin(2 * np.pi * time_values / P + phase)
    return v


def inject_with_velocities(
    spectra: torch.Tensor,
    wavegrid: torch.Tensor,
    velocities: torch.Tensor,
    batch_size: Optional[int] = None,
) -> torch.Tensor:
    if batch_size is None:
        injected = shift_spectra_linear(
            spectra=spectra,
            wavegrid=wavegrid.unsqueeze(0).expand(spectra.shape[0], -1).contiguous(),
            velocities=velocities,
        )
        return injected

    out_chunks = []
    for i in range(0, spectra.shape[0], batch_size):
        end = min(i + batch_size, spectra.shape[0])
        injected = shift_spectra_linear(
            spectra=spectra[i:end],
            wavegrid=wavegrid.unsqueeze(0).expand(end - i, -1).contiguous(),
            velocities=velocities[i:end],
        )
        out_chunks.append(injected)
    return torch.cat(out_chunks, dim=0)


# ============================================================
# ---------------------- Sauvegarde --------------------------
# ============================================================


def build_metadata(
    n_file_total: int,
    n_sel: int,
    n_train: int,
    n_val: int,
    wavemin: float,
    wavemax: float,
    wavegrid_ds: np.ndarray,
    prep: PreprocessParams,
    noise: NoiseParams,
    batch_size: Optional[int],
    original_pixels: int,
    downscaled_pixels: int,
    planets: Optional[PlanetParams],
) -> Dict[str, Any]:
    return {
        "n_spectra_file": int(n_file_total),
        "n_spectra": int(n_sel),
        "n_spectra_train": int(n_train),
        "n_spectra_val": int(n_val),
        "n_pixels": int(len(wavegrid_ds)),
        "wavemin": float(wavemin),
        "wavemax": float(wavemax),
        "downscaling_factor": int(prep.downscaling_factor),
        "smooth_after_downscaling": bool(prep.smooth_after_downscaling),
        "smooth_kernel_size": int(prep.smooth_kernel_size),
        "add_photon_noise": bool(noise.add_photon_noise),
        "snr_target": (
            float(noise.snr_target) if noise.snr_target is not None else None
        ),
        "noise_seed": (int(noise.seed) if noise.seed is not None else None),
        "original_pixels": int(original_pixels),
        "downscaled_pixels": int(downscaled_pixels),
        "batch_size": (int(batch_size) if batch_size is not None else None),
        "planets_periods": (list(planets.periods) if planets else None),
        "planets_amplitudes": (list(planets.amplitudes) if planets else None),
        "planets_phases": (list(planets.phases) if planets else None),
        "activity_definition": "activity = spectra_pre_noise_pre_planets - template",
    }


def save_npz(path: str, payload: Dict[str, Any]):
    np.savez_compressed(path, **payload)


# ============================================================
# ---------------------- Pipeline principal ------------------
# ============================================================


def create_soap_gpu_paper_dataset(
    spectra_filepath: str,
    tmp_filepath: str,
    output_dir: str,
    output_filename: Optional[str] = None,
    idx_train_start: int = 0,
    idx_train_end: int = 100,
    idx_val_start: Optional[int] = 100,
    idx_val_end: Optional[int] = 200,
    wavemin: float = 5000,
    wavemax: float = 5050,
    downscaling_factor: int = 2,
    add_photon_noise: bool = False,
    snr_target: Optional[float] = None,
    noise_seed: Optional[int] = None,
    planets_amplitudes: Optional[Sequence[float]] = None,
    planets_periods: Optional[Sequence[float]] = None,
    planets_phases: Optional[Sequence[float]] = None,
    batch_size: int = 100,
    smooth_after_downscaling: bool = False,
    smooth_kernel_size: int = 3,
    use_rassine=False,
):
    print("🔄 Création du dataset SOAP GPU Paper...")

    # ---- Load template & build mask
    template, wavegrid = load_template_npz(tmp_filepath)
    mask = build_mask(wavegrid, wavemin, wavemax)
    template_masked = template[mask]
    wavegrid_masked = wavegrid[mask]

    # ---- Split config
    split = IndexSplit(idx_train_start, idx_train_end, idx_val_start, idx_val_end)

    # ---- Load spectra selection (+ time)
    spectra_sel, spectra_val, time_values_tot, n_file_total = load_spectra_selection(
        spectra_filepath, mask, split
    )
    n_train = split.train_end - split.train_start
    n_val = (
        0
        if (split.val_start is None or split.val_end is None)
        else (split.val_end - split.val_start)
    )
    n_sel = spectra_sel.shape[0]
    print(
        f"Données chargées: fichier={n_file_total} | sélection={n_sel} ({n_train} train / {n_val} val)"
    )
    print(f"Gamme spectrale: {wavemin:.1f} - {wavemax:.1f} Å")

    if use_rassine:
        # ---- Normalisation avec Rassine ----*
        print("🐍 Normalisation des spectres avec Rassine...")
        for i, spec in enumerate(spectra_sel):
            spectra_sel[i] = normalize_with_rassine(wavegrid_masked, spec)
        template_masked = normalize_with_rassine(wavegrid_masked, template_masked)

    # ---- Downscaling
    Npix = wavegrid_masked.size
    wavegrid_ds = downscale_mean_1d(wavegrid_masked, downscaling_factor)
    template_ds = downscale_mean_1d(template_masked, downscaling_factor)
    spectra_ds, n_bins = downscale_mean_2d(spectra_sel, downscaling_factor)
    print(f"📐 Downscaling: {Npix} → {n_bins} (factor {downscaling_factor})")

    # ---- Optional smoothing
    if smooth_after_downscaling:
        print(f"🔄 Lissage kernel={smooth_kernel_size}...")
        template_ds = uniform_filter1d(
            template_ds, size=smooth_kernel_size, mode="reflect"
        )
        maybe_smooth_inplace(spectra_ds, size=smooth_kernel_size)


    # ---- Activity (pre-noise, pre-planets)
    activity_ds = compute_activity_pre_noise(spectra_ds, template_ds)

    # ---- Noise ----
    noise = NoiseParams(add_photon_noise, snr_target, noise_seed)
    if noise.add_photon_noise:
        print("🔊 Bruit photonique...")
        add_photon_noise_batch(spectra_ds, noise.snr_target, noise.seed)

    # ---- Planets injection (optional)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v_true_tot = np.zeros_like(time_values_tot)
    spectra_ds_no_activity = None

    if (
        planets_amplitudes is not None
        and planets_periods is not None
        and planets_phases is not None
        and len(planets_amplitudes)
        and len(planets_periods)
        and len(planets_phases)
    ):
        print("🌌 Injection du signal planétaire...")
        planets = PlanetParams(planets_amplitudes, planets_periods, planets_phases)
        v_np = compute_velocities(time_values_tot, planets)
        v_true_tot = v_np

        # tensors
        dtype = torch.float32
        spectra_t = torch.tensor(spectra_ds, device=device, dtype=dtype)
        wave_t = torch.tensor(wavegrid_ds, device=device, dtype=dtype)
        v_t = torch.tensor(v_np, device=device, dtype=dtype)

        # inject on dataset with activity+noise
        spectra_inj = inject_with_velocities(
            spectra_t, wave_t, v_t, batch_size=batch_size
        )
        spectra_ds = spectra_inj.detach().cpu().numpy()

        # inject template only (no activity)
        tmpl_t = torch.tensor(template_ds, device=device, dtype=dtype)
        tmpl_batch = tmpl_t.unsqueeze(0).expand(spectra_ds.shape[0], -1).contiguous()
        spectra_noact_inj = inject_with_velocities(
            tmpl_batch, wave_t, v_t, batch_size=batch_size
        )
        spectra_ds_no_activity = spectra_noact_inj.detach().cpu().numpy()

    # ---- Train/val splits for save
    spectra_train = spectra_ds[:n_train]
    spectra_val = spectra_ds[n_train : n_train + n_val]
    activity_train = activity_ds[:n_train]
    activity_val = activity_ds[n_train : n_train + n_val]
    v_true_train = v_true_tot[:n_train]
    v_true_val = v_true_tot[n_train : n_train + n_val]

    # ---- Output filename
    prep = PreprocessParams(
        wavemin,
        wavemax,
        downscaling_factor,
        smooth_after_downscaling,
        smooth_kernel_size,
    )
    planets_obj = (
        PlanetParams(planets_amplitudes, planets_periods, planets_phases)
        if (planets_amplitudes and planets_periods and planets_phases)
        else None
    )
    if not output_filename:
        output_filepath = auto_filename(
            output_dir, n_train, n_val, wavemin, wavemax, prep, noise, planets_obj
        )
    else:
        os.makedirs(output_dir, exist_ok=True)
        output_filepath = os.path.join(output_dir, output_filename)

    # ---- Save
    metadata = build_metadata(
        n_file_total,
        n_sel,
        n_train,
        n_val,
        wavemin,
        wavemax,
        wavegrid_ds,
        prep,
        noise,
        batch_size,
        original_pixels=Npix,
        downscaled_pixels=n_bins,
        planets=planets_obj,
    )

    payload = {
        "wavegrid": wavegrid_ds,
        "template": template_ds,
        "spectra_train": spectra_train,
        "spectra_val": spectra_val,
        "activity_train": activity_train,  # pré-bruit/pré-planètes
        "activity_val": activity_val,
        "time_values_train": time_values_tot[:n_train],
        "time_values_val": time_values_tot[n_train : n_train + n_val],
        "v_true_train": v_true_train,
        "v_true_val": v_true_val,
        "metadata": metadata,
    }
    if spectra_ds_no_activity is not None:
        payload["spectra_no_activity_train"] = spectra_ds_no_activity[:n_train]
        if n_val > 0:
            payload["spectra_no_activity_val"] = spectra_ds_no_activity[
                n_train : n_train + n_val
            ]

    save_npz(output_filepath, payload)

    # ---- Cleanup
    del spectra_sel, spectra_ds, spectra_train, spectra_val, activity_ds
    if spectra_ds_no_activity is not None:
        del spectra_ds_no_activity
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"💾 Fichier de sortie créé: {output_filepath}")
    print(f"   - {n_sel} spectres - {n_train} train / {n_val} val")
    print(f"   - {n_bins} pixels spectraux")
    print(f"   - Gamme: {wavegrid_ds.min():.1f} - {wavegrid_ds.max():.1f} Å")
    print("🧹 Nettoyage mémoire terminé")


if __name__ == "__main__":
    create_soap_gpu_paper_dataset(
        spectra_filepath="/home/tliopis/Codes/exoplanets_llopis_mary_2025/data/soap_gpu_paper/spec_cube_tot_filtered.h5",
        tmp_filepath="/home/tliopis/Codes/exoplanets_llopis_mary_2025/data/soap_gpu_paper/spec_master.npz",
        output_dir="/home/tliopis/Codes/exoplanets_llopis_mary_2025/data/npz_datasets",
        output_filename="test",
        idx_train_start=100,
        idx_train_end=220,
        idx_val_start=1000,
        idx_val_end=1120,
        wavemin=5000.0,
        wavemax=5050.0,
        downscaling_factor=2,
        smooth_after_downscaling=True,
        smooth_kernel_size=3,
        add_photon_noise=False,
        snr_target=None,
        noise_seed=None,
        planets_amplitudes=[0.1],
        planets_periods=[60.0],
        planets_phases=[0.0],
        batch_size=100,
        use_rassine=True
    )
