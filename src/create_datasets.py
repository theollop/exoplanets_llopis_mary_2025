from __future__ import annotations

import gc
import os
import pickle
import re
import sys
import tempfile
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Sequence, Tuple

import pandas as pd
import h5py
import numpy as np
import torch
from scipy.ndimage import uniform_filter1d

from src.interpolate import shift_spectra_linear, shift_spectra_cubic
from src.rassine import normalize_batch_with_rassine, normalize_with_rassine
from src.utils import clear_gpu_memory
from src.batch_ccf_analysis import compute_ccf_proxies_from_arrays
# ============================================================
# ---------------------- Config types ------------------------
# ============================================================


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
    use_realistic_harps_noise: bool = False
    harps_material_path: Optional[str] = None
    snr_scaling: float = 1.0


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
    n_spectra: int,
    wavemin: float,
    wavemax: float,
    prep: PreprocessParams,
    noise: NoiseParams,
    planets: Optional[PlanetParams],
) -> str:
    bits = [f"ns{n_spectra}", f"{int(wavemin)}-{int(wavemax)}"]
    if prep.downscaling_factor and prep.downscaling_factor > 1:
        bits.append(f"dx{prep.downscaling_factor}")
    if prep.smooth_after_downscaling:
        bits.append(f"sm{prep.smooth_kernel_size}")
    if noise.use_realistic_harps_noise:
        bits.append(f"harps-noise-{_fmt_num(noise.snr_scaling)}")
    elif noise.add_photon_noise:
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


def build_mask(wavegrid: np.ndarray, wavemin: float, wavemax: float) -> np.ndarray:
    if wavemin is None:
        wavemin = float(wavegrid.min())
    if wavemax is None:
        wavemax = float(wavegrid.max())
    return (wavegrid >= wavemin) & (wavegrid <= wavemax)


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


def load_activity_proxies_npz(path_npz: str) -> np.ndarray:
    """
    Charge un fichier .npz de résultats CCF et construit un tableau (N,3)
    contenant [depths, fwhms, spans] en float32.
    """
    npz = np.load(path_npz)
    # Adapte les noms de clés si besoin (depths, fwhms, spans sont attendus)
    fwhm = npz["fwhms"].astype(np.float32)  # (N,)
    depth = npz["depths"].astype(np.float32)  # (N,)
    bis = npz["spans"].astype(np.float32)  # (N,)
    proxies = np.stack([depth, fwhm, bis], axis=1)  # (N,3)

    return proxies


def robust_zscore_train_only(x_train: np.ndarray, eps=1e-6):
    """Compute med and mad on training set for robust z-score.

    Returns med, mad arrays with same last-dim as features.
    """
    med = np.median(x_train, axis=0)
    mad = np.median(np.abs(x_train - med), axis=0)
    mad = np.maximum(mad, eps)
    return med, mad


def apply_robust_zscore(x: np.ndarray, med: np.ndarray, mad: np.ndarray):
    return (x - med) / mad


def _add_photon_noise(
    spectrum, snr_target=None, default_snr=300.0, min_flux=1e-12, max_w=1e12
):
    spec = np.asarray(spectrum, dtype=float)
    spec = np.clip(spec, min_flux, None)

    mu = (
        float(np.median(spec)) if np.isfinite(np.median(spec)) else float(np.mean(spec))
    )
    if mu <= 0:
        return spec.copy(), np.zeros_like(spec)

    S = float(snr_target) if (snr_target is not None) else float(default_snr)
    S = max(S, 1.0)
    k = (S * S) / mu

    # Bruit Poisson
    lam = k * spec
    counts = np.random.poisson(lam)
    noisy = counts / k

    # Poids pour L_fid = 1/variance = k / flux (flux sans bruit)
    w_pix = k / spec
    w_pix = np.clip(w_pix, 0.0, max_w)

    return noisy, w_pix


def add_photon_noise_batch(
    X: np.ndarray, snr_target: Optional[float], seed: Optional[int] = None
):
    if seed is not None:
        np.random.seed(int(seed))
    N, P = X.shape
    noisy_X = np.empty_like(X, dtype=float)
    weights_X = np.empty_like(X, dtype=float)
    for i in range(N):
        noisy_X[i], weights_X[i] = _add_photon_noise(X[i], snr_target)
    return noisy_X, weights_X


def load_harps_snr_profile(
    material_pkl_path: str = "/home/tliopis/Codes/exoplanets_llopis_mary_2025/data/rv_datachallenge/Sun_B57001_E61001_planet-FallChallenge1/HARPN/STAR1136_HPN_Analyse_material.p",
    target_wavegrid: Optional[np.ndarray] = None,
    target_wavemin: float = 5000,
    target_wavemax: float = 5050,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Charge le profil SNR réaliste HARPS du RV Data Challenge et l'interpole
    sur la grille spectrale cible.

    Returns:
        snr_profile: Profil SNR interpolé sur target_wavegrid
        reference_flux: Flux de référence interpolé
        ratio_factor: Facteur de ratio SNR appliqué
    """
    import pickle
    from scipy import interpolate

    # Charger les données HARPS
    with open(material_pkl_path, "rb") as f:
        material = pickle.load(f)

    # Extraire les données nécessaires
    harps_wave = material["wave"].to_numpy()
    harps_snr = material["master_snr_curve"]

    if "stellar_template" in material:
        harps_flux = material["stellar_template"]
    elif "reference_spectrum" in material:
        harps_flux = material["reference_spectrum"]
    else:
        raise ValueError("Aucun template/référence trouvé dans les données HARPS")

    # Facteur de ratio
    ratio_factor = 1.0
    if "ratio_factor_snr" in material and material["ratio_factor_snr"] is not None:
        rf = material["ratio_factor_snr"]
        if hasattr(rf, "iloc"):
            ratio_factor = float(rf.iloc[0])
        elif hasattr(rf, "__len__") and len(rf) == 1:
            ratio_factor = float(rf[0])
        else:
            ratio_factor = float(rf)

    # Appliquer le facteur de ratio
    effective_snr = harps_snr * ratio_factor

    # Créer la grille cible si non fournie
    if target_wavegrid is None:
        mask_harps = (harps_wave >= target_wavemin) & (harps_wave <= target_wavemax)
        target_wavegrid = harps_wave[mask_harps]
        target_snr = effective_snr[mask_harps]
        target_flux = harps_flux[mask_harps]
    else:
        # Interpoler sur la grille cible
        interp_snr = interpolate.interp1d(
            harps_wave,
            effective_snr,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        interp_flux = interpolate.interp1d(
            harps_wave,
            harps_flux,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )

        target_snr = interp_snr(target_wavegrid)
        target_flux = interp_flux(target_wavegrid)

    return target_snr, target_flux, ratio_factor


def add_realistic_harps_noise_batch(
    X: np.ndarray,
    wavegrid: np.ndarray,
    snr_scaling: float = 1.0,
    material_pkl_path: Optional[str] = None,
    seed: Optional[int] = None,
    min_flux: float = 1e-12,
    max_weight: float = 1e6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ajoute un bruit réaliste basé sur le profil SNR de HARPS du RV Data Challenge.

    Parameters:
        X: Spectres à bruiter (N, P)
        wavegrid: Grille de longueurs d'onde (P,)
        snr_scaling: Facteur d'échelle pour ajuster le niveau de SNR global
        material_pkl_path: Chemin vers le fichier material HARPS
        seed: Graine aléatoire

    Returns:
        noisy_X: Spectres bruités
        weights_X: Poids correspondants (inverse variance)
    """
    if seed is not None:
        np.random.seed(int(seed))

    # Charger le profil SNR HARPS par défaut si non spécifié
    if material_pkl_path is None:
        material_pkl_path = "/home/tliopis/Codes/exoplanets_llopis_mary_2025/data/rv_datachallenge/Sun_B57001_E61001_planet-FallChallenge1/HARPN/STAR1136_HPN_Analyse_material.p"

    # Charger et interpoler le profil SNR HARPS
    try:
        snr_profile, reference_flux, ratio_factor = load_harps_snr_profile(
            material_pkl_path,
            target_wavegrid=wavegrid,
            target_wavemin=wavegrid.min(),
            target_wavemax=wavegrid.max(),
        )

        # CORRECTION: Normaliser le profil SNR et le mettre à l'échelle cible
        # Le profil HARPS donne des variations relatives, pas des valeurs absolues
        target_snr_mean = 200.0  # SNR cible réaliste
        snr_profile_normalized = (
            snr_profile / snr_profile.mean()
        )  # Normaliser autour de 1
        effective_snr = snr_profile_normalized * target_snr_mean * snr_scaling

        print("📊 Profil SNR HARPS chargé et normalisé")
        print(
            f"   SNR HARPS original: [{snr_profile.min():.1f}, {snr_profile.max():.1f}], mean={snr_profile.mean():.1f}"
        )
        print(
            f"   SNR final: [{effective_snr.min():.1f}, {effective_snr.max():.1f}], mean={effective_snr.mean():.1f}"
        )

    except Exception as e:
        print(f"⚠️ Erreur chargement profil HARPS: {e}")
        print("   Utilisation du SNR uniforme par défaut")
        effective_snr = np.full_like(
            wavegrid, 200.0 * snr_scaling
        )  # SNR réaliste par défaut

    N, P = X.shape
    noisy_X = np.empty_like(X, dtype=float)
    weights_X = np.empty_like(X, dtype=float)

    for i in range(N):
        spectrum = np.asarray(X[i], dtype=float)
        spectrum = np.clip(spectrum, min_flux, None)

        # Pour chaque pixel, calculer le bruit basé sur le SNR local
        noisy_spectrum = np.empty_like(spectrum)
        weights_spectrum = np.empty_like(spectrum)

        for j in range(P):
            flux_j = spectrum[j]
            snr_j = effective_snr[j]

            # Variance = flux / SNR²
            variance_j = flux_j / (snr_j * snr_j)
            sigma_j = np.sqrt(variance_j)

            # Ajouter le bruit gaussien
            noisy_spectrum[j] = flux_j + np.random.normal(0, sigma_j)

            # Poids = 1 / variance = SNR² / flux
            weights_spectrum[j] = (snr_j * snr_j) / flux_j if flux_j > min_flux else 0.0

        # Limiter les poids extrêmes
        weights_spectrum = np.clip(weights_spectrum, 0.0, max_weight)

        noisy_X[i] = noisy_spectrum
        weights_X[i] = weights_spectrum

    return noisy_X, weights_X


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
    gpu_optimized: bool = False,
    replicate_input: bool = False,
    verbose: bool = False,
    device: Any = "cpu",
    interpolate_method: Literal["linear", "cubic"] = "linear",
) -> torch.Tensor:
    """
    Inject velocities into spectra.
    - gpu_optimized=True: offload des sorties sur CPU et nettoyage du cache GPU pour limiter l'utilisation mémoire.
    - replicate_input=True: si `spectra` est 1D (ex: un template), on le réplique à la volée par batch au lieu de l'expanser en entier.
    """
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if batch_size is None:
        spectra = spectra.to(dev)
        wave_b = wavegrid.unsqueeze(0).expand(spectra.shape[0], -1)
        if interpolate_method == "cubic":
            injected = shift_spectra_cubic(
                spectra=spectra,
                wavegrid=wave_b,
                velocities=velocities,
            )
        else:
            injected = shift_spectra_linear(
                spectra=spectra,
                wavegrid=wave_b,
                velocities=velocities,
            )
        if gpu_optimized and torch.cuda.is_available():
            out_cpu = injected.detach().cpu()
            del injected, spectra, wave_b
            torch.cuda.empty_cache()
            return out_cpu
        return injected

    out_chunks = []
    total = int(velocities.shape[0] if replicate_input else spectra.shape[0])
    for i in range(0, total, batch_size):
        if verbose:
            print(f"Processing batch {i // batch_size + 1}")
        end = min(i + batch_size, total)
        if replicate_input:
            spectra_b = spectra.unsqueeze(0).expand(end - i, -1)
        else:
            spectra_b = spectra[i:end]
        wave_b = wavegrid.unsqueeze(0).expand(end - i, -1)
        v_b = velocities[i:end]
        if interpolate_method == "cubic":
            injected = shift_spectra_cubic(
                spectra=spectra_b.to(dev),
                wavegrid=wave_b.to(dev),
                velocities=v_b.to(dev),
            )
        else:
            injected = shift_spectra_linear(
                spectra=spectra_b.to(dev),
                wavegrid=wave_b.to(dev),
                velocities=v_b.to(dev),
            )
        if gpu_optimized and torch.cuda.is_available():
            out_chunks.append(injected.detach().cpu())
            del injected, spectra_b, wave_b, v_b
            torch.cuda.empty_cache()
        else:
            out_chunks.append(injected)
    if gpu_optimized and torch.cuda.is_available():
        if not replicate_input:
            del spectra
        del wavegrid, velocities
        torch.cuda.empty_cache()
    return torch.cat(out_chunks, dim=0)


# ============================================================
# ---------------------- Sauvegarde --------------------------
# ============================================================


def build_metadata(
    n_file_total: int,
    n_spectra: int,
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
        "n_spectra": int(n_spectra),
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
    template_filepath: str,
    wavegrid_filepath: str,
    time_values_filepath: str,
    output_dir: str,
    output_filename: Optional[str] = None,
    idx_start: int = 0,
    idx_end: int = 100,
    wavemin: float = 5000,
    wavemax: float = 5050,
    downscaling_factor: int = 2,
    add_photon_noise: bool = False,
    snr_target: Optional[float] = None,
    noise_seed: Optional[int] = None,
    use_realistic_harps_noise: bool = False,
    harps_material_path: Optional[str] = None,
    snr_scaling: float = 1.0,
    planets_amplitudes: Optional[Sequence[float]] = None,
    planets_periods: Optional[Sequence[float]] = None,
    planets_phases: Optional[Sequence[float]] = None,
    batch_size: int = 100,
    smooth_after_downscaling: bool = False,
    smooth_kernel_size: int = 3,
    use_rassine=False,
    storage_dtype=np.float64,
    compute_ccf_proxies: bool = False,
    new_wavegrid_filepath: str = None,
    interpolate_method="linear",
):
    print("🔄 Création du dataset SOAP GPU Paper...")

    # ---- Load template & build mask
    template = np.load(template_filepath)
    wavegrid = np.load(wavegrid_filepath)
    time_values = np.load(time_values_filepath)
    if wavemin is None:
        wavemin = wavegrid.min()
    if wavemax is None:
        wavemax = wavegrid.max()
    mask = build_mask(wavegrid, wavemin, wavemax)
    template_masked = template[mask]
    wavegrid_masked = wavegrid[mask]
    if new_wavegrid_filepath is not None:
        new_wavegrid = np.load(new_wavegrid_filepath)
        new_wavegrid_mask = build_mask(new_wavegrid, wavemin, wavemax)
        new_wavegrid_masked = new_wavegrid[new_wavegrid_mask]

    # ---- Load spectra selection (+ time)
    with h5py.File(spectra_filepath, "r") as f:
        n_file_total = f["spec_sel"].shape[0]
        if idx_start < 0 or idx_end > n_file_total:
            raise ValueError(
                f"Index range [{idx_start}:{idx_end}] out of bounds for file with {n_file_total} spectra."
            )
        spectra = f["spec_sel"][idx_start:idx_end, :][:, mask]
        time_values = time_values[idx_start:idx_end]

    n_spectra = spectra.shape[0]
    print(f"Données chargées: fichier={n_file_total} | sélection={n_spectra} spectra")
    print(f"Gamme spectrale: {wavemin:.1f} - {wavemax:.1f} Å")

    if use_rassine:
        # ---- Normalisation avec Rassine ----*
        print("🐍 Normalisation des spectres avec Rassine...")
        for i in range(0, n_spectra, batch_size):
            print(f"  Traitement du lot {i // batch_size + 1}...")
            spectra[i : i + batch_size] = normalize_batch_with_rassine(
                wavegrid_masked, spectra[i : i + batch_size]
            )
        template_masked = normalize_with_rassine(wavegrid_masked, template_masked)

    # ---- Downscaling
    Npix = wavegrid_masked.size
    # If a new_wavegrid was supplied, interpolate template and spectra on it
    if new_wavegrid_filepath is not None:
        try:
            from scipy.interpolate import interp1d

            # Utiliser la même méthode d'interpolation pour template et spectres
            # pour éviter les inconsistances aux bords
            
            # interpolate template
            try:
                # Utiliser np.interp qui est plus conservateur aux bords
                template_masked = np.interp(
                    new_wavegrid_masked, wavegrid_masked, template_masked
                )
            except Exception as e:
                print(f"⚠️  Erreur interpolation template: {e}")
                # En cas d'échec, garder la grille originale
                new_wavegrid_masked = wavegrid_masked
                print("   Utilisation de la grille originale")

            # interpolate each spectrum onto new grid
            spect_interp = np.empty(
                (spectra.shape[0], new_wavegrid_masked.size), dtype=spectra.dtype
            )
            for i in range(spectra.shape[0]):
                try:
                    spect_interp[i] = np.interp(
                        new_wavegrid_masked, wavegrid_masked, spectra[i]
                    )
                except Exception:
                    # if interpolation fails for a spectrum, fill with template values
                    print(f"⚠️  Erreur interpolation spectre {i}, utilisation du template")
                    spect_interp[i] = template_masked.copy()

            spectra = spect_interp
            # replace the masked wavegrid with the new one for downstream ops
            wavegrid_masked = new_wavegrid_masked
            
            # Vérification de cohérence après interpolation
            activity_check = np.mean(np.abs(spectra - template_masked.reshape(1, -1)), axis=0)
            edge_pixels = 10
            edge_activity = np.mean([
                activity_check[:edge_pixels].mean(), 
                activity_check[-edge_pixels:].mean()
            ])
            center_activity = activity_check[edge_pixels:-edge_pixels].mean()
            
            if edge_activity > 3 * center_activity:
                print(f"⚠️  Activité élevée aux bords après interpolation: "
                      f"bords={edge_activity:.6f}, centre={center_activity:.6f}")
                print("   Cela peut indiquer un problème d'interpolation")
            
            print(
                f"Interpolated spectra and template on new grid (n_pix={wavegrid_masked.size})"
            )
        except Exception as e:
            print(f"⚠️  Interpolation sur new_wavegrid_masked a échoué: {e}")
    if downscaling_factor > 1:
        wavegrid_ds = downscale_mean_1d(wavegrid_masked, downscaling_factor)
        template_ds = downscale_mean_1d(template_masked, downscaling_factor)
        spectra_ds, n_bins = downscale_mean_2d(spectra, downscaling_factor)
        print(f"📐 Downscaling: {Npix} → {n_bins} (factor {downscaling_factor})")
    else :
        wavegrid_ds = wavegrid_masked
        template_ds = template_masked
        spectra_ds = spectra
        n_bins = Npix

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
    noise = NoiseParams(
        add_photon_noise,
        snr_target,
        noise_seed,
        use_realistic_harps_noise,
        harps_material_path,
        snr_scaling,
    )
    weights_fid = None

    if use_realistic_harps_noise:
        print("🔊 Bruit réaliste HARPS basé sur le RV Data Challenge...")
        spectra_ds, weights_fid = add_realistic_harps_noise_batch(
            spectra_ds,
            wavegrid_ds,
            snr_scaling=snr_scaling,
            material_pkl_path=harps_material_path,
            seed=noise_seed,
        )
        # Optionnel: normaliser les poids pour éviter le déséquilibre
        if weights_fid is not None:
            weights_mean = weights_fid.mean()
            # Option 1: Normalisation
            # weights_fid = weights_fid / weights_mean
            # Option 2: Écrêtage (recommandé)
            weights_fid = np.clip(weights_fid, weights_mean * 0.1, weights_mean * 10)
            print(
                f"   Poids écrêtés: range [{weights_fid.min():.3f}, {weights_fid.max():.3f}]"
            )
    elif noise.add_photon_noise:
        print("🔊 Bruit photonique...")
        spectra_ds, weights_fid = add_photon_noise_batch(
            spectra_ds, noise.snr_target, noise.seed
        )

    # ---- Activity after noise (bruitée) ----
    # Calculer l'activité après bruitage : spectra_bruités - template_bruité
    activity_noised = None
    if noise.add_photon_noise or use_realistic_harps_noise:
        # Pour l'activité bruitée, on utilise les spectres bruités moins le template (non bruité)
        # car le template reste la référence stellaire
        activity_noised = spectra_ds - template_ds.reshape(1, -1)
        print(
            f"📊 Activité bruitée calculée: range [{activity_noised.min():.6f}, {activity_noised.max():.6f}]"
        )

    # ---- Planets injection (optional)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v_true_tot = np.zeros_like(time_values, dtype=float)
    spectra_ds_no_activity = None

    if (
        planets_amplitudes is not None
        and planets_periods is not None
        and planets_phases is not None
        and len(planets_amplitudes)
        and len(planets_periods)
        and len(planets_phases)
    ):
        print(
            f"🌌 Injection du signal planétaire... méthode d'interpolation : {interpolate_method}"
        )
        planets = PlanetParams(planets_amplitudes, planets_periods, planets_phases)
        v_np = compute_velocities(time_values, planets)
        v_true_tot = v_np.astype(float, copy=False)

        # tensors (use float32 to halve memory)
        dtype = torch.float32
        dev = device
        spectra_t = torch.tensor(spectra_ds, device="cpu", dtype=dtype)
        wave_t = torch.tensor(wavegrid_ds, device=dev, dtype=dtype)
        v_t = torch.tensor(v_np, device=dev, dtype=dtype)

        # inject on dataset with activity+noise (batch stream to GPU, offload to CPU)
        spectra_inj = inject_with_velocities(
            spectra_t,
            wave_t,
            v_t,
            batch_size=batch_size,
            gpu_optimized=True,
            replicate_input=False,
            verbose=False,
            device=dev,
            interpolate_method=interpolate_method,
        )
        spectra_ds = spectra_inj.detach().cpu().numpy()

        # inject template only (no activity) using replicate_input to avoid pre-expansion
        tmpl_t = torch.tensor(template_ds, device="cpu", dtype=dtype)
        spectra_noact_inj = inject_with_velocities(
            tmpl_t,
            wave_t,
            v_t,
            batch_size=batch_size,
            gpu_optimized=True,
            replicate_input=True,
            verbose=False,
            device=dev,
            interpolate_method=interpolate_method,
        )
        spectra_ds_no_activity = spectra_noact_inj.detach().cpu().numpy()

    # ---- Train/val splits for save
    spectra_out = spectra_ds[:n_spectra]
    activity_out = activity_ds[:n_spectra]
    activity_noised_out = (
        activity_noised[:n_spectra] if activity_noised is not None else None
    )
    v_true_out = v_true_tot[:n_spectra]

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
            output_dir, n_spectra, wavemin, wavemax, prep, noise, planets_obj
        )
    else:
        os.makedirs(output_dir, exist_ok=True)
        output_filepath = os.path.join(output_dir, output_filename)

    # ---- Save
    metadata = build_metadata(
        n_file_total,
        n_spectra,
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
        "wavegrid": wavegrid_ds.astype(storage_dtype, copy=False),
        "template": template_ds.astype(storage_dtype, copy=False),
        "spectra": spectra_out.astype(storage_dtype, copy=False),
        "activity": activity_out.astype(storage_dtype, copy=False),
        "time_values": time_values.astype(storage_dtype, copy=False),
        "v_true": v_true_out.astype(storage_dtype, copy=False),
        "metadata": metadata,
    }

    # Ajouter l'activité bruitée si calculée
    if activity_noised_out is not None:
        payload["activity_noised"] = activity_noised_out.astype(
            storage_dtype, copy=False
        )
    # Optionnel: calculer et joindre les proxies CCF sur le dataset final
    if compute_ccf_proxies:
        print("Calcul des proxies CCF...")
        try:
            proxies_ccf = compute_ccf_proxies_from_arrays(
                spectra=spectra_out,
                wavegrid=wavegrid_ds,
                batch_size=batch_size or 128,
                v_grid_range=(-20000, 20000),
                v_grid_step=250,
                window_size_velocity=820.0,
                mask_type="G2",
                normalize=True,
                return_raw_ccfs=True,
                verbose=True,
            )
            depths = proxies_ccf["depths"].astype(np.float32)
            fwhms = proxies_ccf["fwhms"].astype(np.float32)
            spans = proxies_ccf["spans"].astype(np.float32)
            proxies = np.stack([depths, fwhms, spans], axis=1)
            med, mad = robust_zscore_train_only(proxies)
            proxies_norm = apply_robust_zscore(proxies, med, mad)

            payload["activity_proxies"] = proxies.astype(np.float32, copy=False)
            payload["activity_proxies_norm"] = proxies_norm.astype(
                np.float32, copy=False
            )
            payload["activity_proxies_med"] = med.astype(np.float32, copy=False)
            payload["activity_proxies_mad"] = mad.astype(np.float32, copy=False)
            payload["metadata"]["activity_proxies_included"] = True
            payload["metadata"]["activity_proxies_mask_type"] = "G2"
            payload["metadata"]["activity_proxies_v_grid_range"] = (
                -20000,
                20000,
            )
            payload["metadata"]["activity_proxies_v_grid_step"] = 250
            payload["metadata"]["activity_proxies_window_size_velocity"] = 820.0
            payload["ccfs"] = proxies_ccf["raw_ccfs"].astype(np.float32, copy=False)
        except Exception as e:
            print(f"⚠️  Échec du calcul des proxies CCF: {e}")
    if spectra_ds_no_activity is not None:
        payload["spectra_no_activity"] = spectra_ds_no_activity[:n_spectra].astype(
            storage_dtype, copy=False
        )
    if weights_fid is not None:
        payload["weights_fid"] = weights_fid[:n_spectra].astype(
            storage_dtype, copy=False
        )
    save_npz(output_filepath, payload)

    # ---- Cleanup
    del spectra, spectra_ds, spectra_out, activity_ds
    if spectra_ds_no_activity is not None:
        del spectra_ds_no_activity
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"💾 Fichier de sortie créé: {output_filepath}")
    print(f"   - {n_spectra} spectres")
    print(f"   - {n_bins} pixels spectraux")
    print(f"   - Gamme: {wavegrid_ds.min():.1f} - {wavegrid_ds.max():.1f} Å")
    print("🧹 Nettoyage mémoire terminé")


def create_rvdatachallenge_dataset(
    flux_path="/home/tliopis/Codes/exoplanets_llopis_mary_2025/data/rv_datachallenge/Sun_B57001_E61001_planet-FallChallenge1/HARPN/STAR1136_HPN_flux_YVA.npy",
    summary_csv_path="/home/tliopis/Codes/exoplanets_llopis_mary_2025/data/rv_datachallenge/Sun_B57001_E61001_planet-FallChallenge1/HARPN/STAR1136_HPN_Analyse_summary.csv",
    material_pkl_path="/home/tliopis/Codes/exoplanets_llopis_mary_2025/data/rv_datachallenge/Sun_B57001_E61001_planet-FallChallenge1/HARPN/STAR1136_HPN_Analyse_material.p",
    output_dir: Optional[str] = None,
    output_filename: Optional[str] = None,
    idx_start: Optional[int] = None,
    idx_end: Optional[int] = None,
    wavemin: Optional[float] = 5000,
    wavemax: Optional[float] = 5050,
    downscaling_factor: int = 2,
    smooth_after_downscaling: bool = False,
    smooth_kernel_size: int = 3,
    planets_amplitudes: Optional[Sequence[float]] = None,
    planets_periods: Optional[Sequence[float]] = None,
    planets_phases: Optional[Sequence[float]] = None,
    batch_size: int = 100,
    storage_dtype=np.float64,
    ccf_npz_path: Optional[str] = None,
):
    """
    Construis un payload comparable à `create_soap_gpu_paper_dataset` à partir des
    données RV Data Challenge. Aucun traitement de normalisation ni ajout de bruit
    n'est appliqué (comme demandé). Les clés pour lesquelles il n'y a pas de
    vérité terrain (ex: `activity`) ne sont pas ajoutées.

    Si `output_dir` est fourni, le fichier .npz est sauvegardé comme dans
    `create_soap_gpu_paper_dataset`.
    """

    # ---- Load raw files
    flux_all = np.load(flux_path)  # (N_tot, P)
    df_summary = pd.read_csv(summary_csv_path)
    if "jdb" in df_summary.columns:
        times_all = df_summary["jdb"].values
    else:
        raise ValueError("Colonne 'jdb' non trouvée dans le CSV résumé.")

    with open(material_pkl_path, "rb") as f:
        material = pickle.load(f)

    if "wave" in material:
        wave = np.asarray(material["wave"])
    else:
        raise ValueError("'wave' absent du fichier material.p")

    if "stellar_template" in material:
        reference_flux = np.asarray(material["stellar_template"])
    elif "reference_spectrum" in material:
        reference_flux = np.asarray(material["reference_spectrum"])
    else:
        raise ValueError("Aucun spectre de référence trouvé dans le pickle.")

    if "master_snr_curve" in material:
        snr_curve = np.asarray(material["master_snr_curve"])
    else:
        raise ValueError("'master_snr_curve' non trouvé dans le pickle.")

    if "ratio_factor_snr" in material and material["ratio_factor_snr"] is not None:
        val = material["ratio_factor_snr"]
        try:
            # pandas Series
            if hasattr(val, "values") and not np.isscalar(val):
                arr = np.asarray(val.values)
            else:
                arr = np.asarray(val)

            if arr.size == 1:
                factor = float(arr.ravel()[0])
                snr_curve = snr_curve * factor
            elif arr.size == snr_curve.size:
                snr_curve = snr_curve * arr
            else:
                # fallback: use first element and warn
                print(
                    "⚠️  ratio_factor_snr length != snr_curve length; using first element"
                )
                factor = float(arr.ravel()[0])
                snr_curve = snr_curve * factor
        except Exception as e:
            print(f"⚠️  Impossible d'appliquer ratio_factor_snr: {e}")

    # ---- Temporal selection
    N_tot = flux_all.shape[0]
    # If both None -> take full range
    if idx_start is None:
        idx_start_i = 0
    else:
        idx_start_i = int(idx_start)
    if idx_end is None:
        idx_end_i = N_tot
    else:
        idx_end_i = int(idx_end)

    # clamp bounds
    idx_start_i = max(0, idx_start_i)
    idx_end_i = min(N_tot, idx_end_i)
    if idx_end_i < idx_start_i:
        raise ValueError(f"idx_end ({idx_end_i}) must be >= idx_start ({idx_start_i})")

    spectra_sel = flux_all[idx_start_i:idx_end_i]
    time_values = times_all[idx_start_i:idx_end_i]

    n_spectra = spectra_sel.shape[0]

    # ---- Spectral mask and exclusion
    if wavemin is None:
        wavemin = float(wave.min())
    if wavemax is None:
        wavemax = float(wave.max())
    mask_wave = build_mask(wave, wavemin, wavemax)

    # Ne pas exclure de pixels supplémentaires: on utilise uniquement le masque
    # spectral défini par wavemin/wavemax (pas de suppression via pixels_rnr/mask_brute)
    n_pix = reference_flux.shape[0]
    mask = mask_wave.copy()

    # ---- Apply mask
    wave_masked = wave[mask]
    reference_flux_masked = reference_flux[mask]
    snr_masked = snr_curve[mask]
    spectra_masked = spectra_sel[:, mask]

    # ---- Downscaling
    if downscaling_factor is None or downscaling_factor <= 1:
        wavegrid_ds = wave_masked
        template_ds = reference_flux_masked
        spectra_ds = spectra_masked
        n_bins = wave_masked.size
    else:
        wavegrid_ds = downscale_mean_1d(wave_masked, downscaling_factor)
        template_ds = downscale_mean_1d(reference_flux_masked, downscaling_factor)
        spectra_ds, n_bins = downscale_mean_2d(spectra_masked, downscaling_factor)

    # ---- Optional smoothing
    if smooth_after_downscaling:
        template_ds = uniform_filter1d(
            template_ds, size=smooth_kernel_size, mode="reflect"
        )
        maybe_smooth_inplace(spectra_ds, size=smooth_kernel_size)

    # ---- Activity: not available reliably => omit
    activity_ds = None

    # ---- Build spectra_no_activity baseline from template (one per observation)
    # By default it's the template repeated; if planets provided we'll inject velocities below
    tmpl_batch = np.tile(template_ds.reshape(1, -1), (n_spectra, 1))
    spectra_ds_no_activity = tmpl_batch.copy()

    # ---- Sigma per pixel and weights
    sigma_pix = reference_flux_masked / np.clip(snr_masked, 1e-12, None)
    if downscaling_factor is None or downscaling_factor <= 1:
        sigma_ds = sigma_pix
    else:
        sigma_ds = downscale_mean_1d(sigma_pix, downscaling_factor)

    with np.errstate(divide="ignore"):
        weights_pix = 1.0 / (sigma_ds**2)
    weights_pix = np.nan_to_num(weights_pix, posinf=0.0, neginf=0.0)
    # replicate per spectrum to match create_soap shape when needed
    weights_fid = np.tile(weights_pix, (n_spectra, 1))

    # ---- Planets
    planets_obj = None
    v_true = np.zeros(n_spectra, dtype=float)
    spectra_ds_no_activity = None
    if (
        planets_amplitudes is not None
        and planets_periods is not None
        and planets_phases is not None
        and len(planets_amplitudes)
        and len(planets_periods)
        and len(planets_phases)
    ):
        planets_obj = PlanetParams(planets_amplitudes, planets_periods, planets_phases)
        v_np = compute_velocities(time_values, planets_obj)
        v_true = v_np.astype(float, copy=False)

        # inject on dataset: shift both the downscaled observed spectra and the template
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float64

        # prepare tensors
        spectra_t = torch.tensor(spectra_ds, device=device, dtype=dtype)
        wave_t = torch.tensor(wavegrid_ds, device=device, dtype=dtype)
        v_t = torch.tensor(v_np, device=device, dtype=dtype)

        # inject planets into observed downscaled spectra
        spectra_inj = inject_with_velocities(
            spectra_t, wave_t, v_t, batch_size=batch_size
        )
        spectra_ds = spectra_inj.detach().cpu().numpy()

        # inject planets into template to build spectra_no_activity
        tmpl_t = torch.tensor(template_ds, device=device, dtype=dtype)
        tmpl_batch = tmpl_t.unsqueeze(0).expand(n_spectra, -1).contiguous()
        spectra_noact_inj = inject_with_velocities(
            tmpl_batch, wave_t, v_t, batch_size=batch_size
        )
        spectra_ds_no_activity = spectra_noact_inj.detach().cpu().numpy()

    # ---- Metadata
    prep = PreprocessParams(
        wavemin,
        wavemax,
        downscaling_factor,
        smooth_after_downscaling,
        smooth_kernel_size,
    )
    noise = NoiseParams(add_photon_noise=False, snr_target=None, seed=None)
    metadata = build_metadata(
        n_file_total=N_tot,
        n_spectra=n_spectra,
        wavemin=wavemin,
        wavemax=wavemax,
        wavegrid_ds=wavegrid_ds,
        prep=prep,
        noise=noise,
        batch_size=batch_size,
        original_pixels=int(n_pix),
        downscaled_pixels=int(n_bins),
        planets=planets_obj,
    )

    # ---- Build payload similar to create_soap_gpu_paper_dataset
    payload = {
        "wavegrid": wavegrid_ds.astype(storage_dtype, copy=False),
        "template": template_ds.astype(storage_dtype, copy=False),
        "spectra": spectra_ds.astype(storage_dtype, copy=False),
        "time_values": time_values[:n_spectra].astype(storage_dtype, copy=False),
        "v_true": v_true.astype(storage_dtype, copy=False),
        "metadata": metadata,
    }
    # Optional: add activity proxies from CCF .npz if provided
    if ccf_npz_path is not None:
        try:
            proxies = load_activity_proxies_npz(ccf_npz_path)
            if proxies.shape[0] < n_spectra:
                raise ValueError(
                    f"CCF proxies length ({proxies.shape[0]}) < n_spectra ({n_spectra})"
                )
            proxies = proxies[:n_spectra]
            med, mad = robust_zscore_train_only(proxies)
            proxies_norm = apply_robust_zscore(proxies, med, mad)

            payload["activity_proxies"] = proxies.astype(np.float32, copy=False)
            payload["activity_proxies_norm"] = proxies_norm.astype(
                np.float32, copy=False
            )
            payload["activity_proxies_med"] = med.astype(np.float32, copy=False)
            payload["activity_proxies_mad"] = mad.astype(np.float32, copy=False)
            payload["metadata"]["activity_proxies_included"] = True
        except Exception as e:
            print(f"⚠️  Loading activity proxies failed: {e}")

    if activity_ds is not None:
        payload["activity"] = activity_ds.astype(storage_dtype, copy=False)
    if spectra_ds_no_activity is not None:
        payload["spectra_no_activity"] = spectra_ds_no_activity[:n_spectra].astype(
            storage_dtype, copy=False
        )
    if weights_fid is not None:
        payload["weights_fid"] = weights_fid[:n_spectra].astype(
            storage_dtype, copy=False
        )
    # add sigma per pixel
    payload["sigma"] = sigma_ds.astype(storage_dtype, copy=False)

    # Save if requested
    # Save if requested — same naming bits but prefix 'rvdatachallenge_'
    if output_dir is not None:
        planets_obj = (
            PlanetParams(planets_amplitudes, planets_periods, planets_phases)
            if (planets_amplitudes and planets_periods and planets_phases)
            else None
        )
        if not output_filename:
            bits = [f"ns{n_spectra}", f"{int(wavemin)}-{int(wavemax)}"]
            if prep.downscaling_factor and prep.downscaling_factor > 1:
                bits.append(f"dx{prep.downscaling_factor}")
            if prep.smooth_after_downscaling:
                bits.append(f"sm{prep.smooth_kernel_size}")
            if noise.add_photon_noise:
                bits.append(
                    "noise"
                    if noise.snr_target is None
                    else f"snr{_fmt_num(noise.snr_target)}"
                )
            if (
                planets_obj is not None
                and len(planets_obj.periods)
                and len(planets_obj.amplitudes)
                and len(planets_obj.phases)
            ):
                bits += [
                    f"P{_fmt_list(planets_obj.periods)}",
                    f"K{_fmt_list(planets_obj.amplitudes)}",
                    f"Phi{_fmt_list(planets_obj.phases)}",
                ]
            base = _slugify("rvdatachallenge_" + "_".join(bits), max_len=80)
            os.makedirs(output_dir, exist_ok=True)
            output_filepath = os.path.join(output_dir, f"{base}.npz")
        else:
            os.makedirs(output_dir, exist_ok=True)
            output_filepath = os.path.join(output_dir, output_filename)

        save_npz(output_filepath, payload)
        print(f"💾 RV Data Challenge payload saved: {output_filepath}")

    return payload


if __name__ == "__main__":
    clear_gpu_memory()

    # create_rvdatachallenge_dataset(
    #     flux_path="/home/tliopis/Codes/exoplanets_llopis_mary_2025/data/rv_datachallenge/Sun_B57001_E61001_planet-FallChallenge1/HARPN/STAR1136_HPN_flux_YVA.npy",
    #     summary_csv_path="/home/tliopis/Codes/exoplanets_llopis_mary_2025/data/rv_datachallenge/Sun_B57001_E61001_planet-FallChallenge1/HARPN/STAR1136_HPN_Analyse_summary.csv",
    #     material_pkl_path="/home/tliopis/Codes/exoplanets_llopis_mary_2025/data/rv_datachallenge/Sun_B57001_E61001_planet-FallChallenge1/HARPN/STAR1136_HPN_Analyse_material.p",
    #     idx_start=None,
    #     idx_end=None,
    #     wavemin=5000,
    #     wavemax=5050,
    #     downscaling_factor=1,
    #     smooth_after_downscaling=False,
    #     smooth_kernel_size=3,
    #     planets_amplitudes=None,
    #     planets_periods=None,
    #     planets_phases=None,
    #     output_dir="/home/tliopis/Codes/exoplanets_llopis_mary_2025/data/npz_datasets",
    # )

    # create_soap_gpu_paper_dataset(

    # )

    create_soap_gpu_paper_dataset(
        spectra_filepath="/home/tliopis/Codes/exoplanets_llopis_mary_2025/data/soap_gpu_paper/soap_equiv_for_harps.h5",
        template_filepath="/home/tliopis/Codes/exoplanets_llopis_mary_2025/data/soap_gpu_paper/template.npy",
        wavegrid_filepath="/home/tliopis/Codes/exoplanets_llopis_mary_2025/data/soap_gpu_paper/wavegrid.npy",
        new_wavegrid_filepath="/home/tliopis/Codes/exoplanets_llopis_mary_2025/data/soap_gpu_paper/new_wavegrid.npy",
        time_values_filepath="/home/tliopis/Codes/exoplanets_llopis_mary_2025/data/soap_gpu_paper/time_values_soap_equiv_for_harps.npy",
        output_dir="/home/tliopis/Codes/exoplanets_llopis_mary_2025/data/npz_datasets",
        idx_start=0,
        idx_end=1275,
        wavemin=5000,
        wavemax=5010,
        downscaling_factor=1,
        smooth_after_downscaling=False,
        smooth_kernel_size=1,
        add_photon_noise=True,
        snr_target=2000,
        noise_seed=42,
        # use_realistic_harps_noise=False,
        # harps_material_path="/home/tliopis/Codes/exoplanets_llopis_mary_2025/data/rv_datachallenge/Sun_B57000_E61000_planet-FallChallenge3/HARPN/STAR1134_HPN_Analyse_material.p",
        # snr_scaling=1,
        planets_amplitudes=[0.5],
        planets_periods=[100],
        planets_phases=[0],
        batch_size=100,
        use_rassine=False,
        storage_dtype=np.float32,
        compute_ccf_proxies=True,
        interpolate_method="linear",
    )

    # create_soap_gpu_paper_dataset(
    #     spectra_filepath="/home/tliopis/Codes/exoplanets_llopis_mary_2025/data/soap_gpu_paper/spec_cube_tot_filtered_normalized_float32.h5",
    #     template_filepath="/home/tliopis/Codes/exoplanets_llopis_mary_2025/data/soap_gpu_paper/template.npy",
    #     wavegrid_filepath="/home/tliopis/Codes/exoplanets_llopis_mary_2025/data/soap_gpu_paper/wavegrid.npy",
    #     # new_wavegrid_filepath="/home/tliopis/Codes/exoplanets_llopis_mary_2025/data/soap_gpu_paper/new_wavegrid.npy",
    #     time_values_filepath="/home/tliopis/Codes/exoplanets_llopis_mary_2025/data/soap_gpu_paper/time_values.npy",
    #     output_dir="/home/tliopis/Codes/exoplanets_llopis_mary_2025/data/npz_datasets",
    #     output_filename="plot_dataset_ccf_activity_planet.npz",
    #     idx_start=500,
    #     idx_end=800,
    #     wavemin=None,
    #     wavemax=None,
    #     downscaling_factor=1,
    #     smooth_after_downscaling=False,
    #     smooth_kernel_size=1,
    #     add_photon_noise=False,
    #     snr_target=300.0,
    #     noise_seed=42,
    #     planets_amplitudes=[4],
    #     planets_periods=[100],
    #     planets_phases=[0.0],
    #     batch_size=10,
    #     use_rassine=False,
    #     storage_dtype=np.float32,
    #     compute_ccf_proxies=False,
    #     interpolate_method="linear",
    # )

    # create_soap_gpu_paper_dataset(
    #     spectra_filepath="/home/tliopis/Codes/exoplanets_llopis_mary_2025/data/soap_gpu_paper/soap_equiv_for_harps.h5",
    #     template_filepath="/home/tliopis/Codes/exoplanets_llopis_mary_2025/data/soap_gpu_paper/template.npy",
    #     wavegrid_filepath="/home/tliopis/Codes/exoplanets_llopis_mary_2025/data/soap_gpu_paper/wavegrid.npy",
    #     new_wavegrid_filepath="/home/tliopis/Codes/exoplanets_llopis_mary_2025/data/soap_gpu_paper/new_wavegrid.npy",
    #     time_values_filepath="/home/tliopis/Codes/exoplanets_llopis_mary_2025/data/soap_gpu_paper/time_values_equiv_for_harps.npy",
    #     output_dir="/home/tliopis/Codes/exoplanets_llopis_mary_2025/data/npz_datasets",
    #     output_filename="plot_dataset_ccf_activity_planet_harps.npz",
    #     idx_start=500,
    #     idx_end=800,
    #     wavemin=None,
    #     wavemax=None,
    #     downscaling_factor=1,
    #     smooth_after_downscaling=False,
    #     smooth_kernel_size=1,
    #     add_photon_noise=False,
    #     snr_target=300.0,
    #     noise_seed=42,
    #     planets_amplitudes=[4],
    #     planets_periods=[100],
    #     planets_phases=[0.0],
    #     batch_size=10,
    #     use_rassine=False,
    #     storage_dtype=np.float32,
    #     compute_ccf_proxies=False,
    #     interpolate_method="linear",
    # )
