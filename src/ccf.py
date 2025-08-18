from matplotlib import pyplot as plt
import numpy as np
from numba import njit, prange
from scipy.sparse import coo_matrix, csr_matrix
from scipy.optimize import curve_fit
from typing import Optional, Dict, Any
from numpy.typing import NDArray

from src.utils import get_mask, get_free_memory, get_max_nv
import torch
import time
from src.interpolate import shift_spectra_linear

__all__ = [
    "compute_CCFs",
    "normalize_CCFs",
    "analyze_ccfs",
    "fit_gaussian_ccf",
    "compute_bisector_span",
    "build_CCF_masks_sparse",
    "get_full_ccf_analysis",
    "get_full_ccf_analysis_template",
]

# =============================================================
# API PUBLIQUE
# =============================================================


def get_full_ccf_analysis(
    spectra: NDArray[np.floating],
    wavegrid: NDArray[np.floating],
    v_grid: NDArray[np.floating],
    window_size_velocity: float,
    mask_type: str = "G2",
    verbose: bool = False,
    batch_size: Optional[int] = None,
    normalize: bool = True,
    return_raw_ccfs: bool = False,
) -> Dict[str, NDArray[np.floating]]:
    """
    Calcule les CCFs puis réalise l'analyse (RV, profondeur, FWHM, span).

    Args:
        spectra: Tableau 2D des spectres, de forme (n_spectra, n_wavelengths).
        wavegrid: Grille de longueurs d'onde (pas constant, triée).
        v_grid: Grille de vitesses radiales en m/s.
        window_size_velocity: Largeur (sigma en m/s) du profil gaussien autour de chaque raie.
        mask_type: Type de masque de raies (ex: "G2").
        verbose: Affiche des informations de progression.
        batch_size: Taille des lots pour calculer les CCFs (None = tout d'un coup).
        normalize: Normaliser les CCFs par leur amplitude max absolue.

    Returns:
        Dictionnaire contenant les mesures pour chaque spectre:
        {"rv", "depth", "fwhm", "span", "continuum", "amplitude", "popt"}.
    """
    CCFs = compute_CCFs(
        spectra=spectra,
        wavegrid=wavegrid,
        v_grid=v_grid,
        window_size_velocity=window_size_velocity,
        mask_type=mask_type,
        verbose=verbose,
        batch_size=batch_size,
        normalize=normalize,
    )
    if return_raw_ccfs:
        return analyze_ccfs(CCFs, v_grid), CCFs
    else:
        return analyze_ccfs(CCFs, v_grid)


def get_full_ccf_analysis_template(
    spectra: NDArray[np.floating],
    template: NDArray[np.floating],
    wavegrid: NDArray[np.floating],
    v_grid: NDArray[np.floating],
    dtype: torch.dtype = torch.float32,
    verbose: bool = False,
    batch_size_specs: Optional[int] = None,
    normalize: bool = True,
    return_raw_ccfs: bool = False,
) -> Dict[str, NDArray[np.floating]]:
    """
    Calcule les CCFs via un template décalé (méthode rapide) puis réalise l'analyse.

    Args:
        spectra: Tableau 2D des spectres, de forme (n_spectra, n_wavelengths), continuum ~ 1.
        template: Profil 1D (longueur = n_wavelengths) servant de template.
        wavegrid: Grille de longueurs d'onde (pas constant, triée).
        v_grid: Grille de vitesses radiales en m/s.
        dtype: Type torch utilisé pour les produits matriciels.
        verbose: Affiche des informations de progression.
        batch_size_specs: Taille des lots pour les spectres (None = auto).
        normalize: Normaliser les CCFs par leur amplitude max absolue.
        return_raw_ccfs: Si True, retourne aussi la matrice des CCFs.

    Returns:
        Dictionnaire contenant les mesures pour chaque spectre:
        {"rv", "depth", "fwhm", "span", "continuum", "amplitude", "popt"}.
        Si return_raw_ccfs=True, retourne un tuple (dict, CCFs).
    """
    CCFs = compute_CCFs_template(
        specs=spectra,
        template=template,
        wavegrid=wavegrid,
        v_grid=v_grid,
        dtype=dtype,
        verbose=verbose,
        batch_size_specs=batch_size_specs,
    )

    # Décalage à 0 par CCF, pour cohérence avec compute_CCFs
    CCFs -= np.min(CCFs, axis=1, keepdims=True)
    if normalize:
        CCFs = normalize_CCFs(CCFs)

    if return_raw_ccfs:
        return analyze_ccfs(CCFs, v_grid, template=True), CCFs
    else:
        return analyze_ccfs(CCFs, v_grid, template=True)


def compute_CCFs(
    spectra: NDArray[np.floating],
    wavegrid: NDArray[np.floating],
    v_grid: NDArray[np.floating],
    window_size_velocity: float,
    mask_type: str = "G2",
    verbose: bool = False,
    batch_size: Optional[int] = None,
    normalize: bool = True,
) -> NDArray[np.floating]:
    """
    Calcule les CCFs par produit matrice-vecteur avec un masque creux gaussien.

    Args:
        spectra: Tableau 2D des spectres, de forme (n_spectra, n_wavelengths).
        wavegrid: Grille de longueurs d'onde (pas constant, triée).
        v_grid: Grille de vitesses radiales en m/s.
        window_size_velocity: Largeur (sigma en m/s) du profil gaussien autour de chaque raie.
        mask_type: Type de masque de raies (ex: "G2").
        verbose: Affiche des informations de progression.
        batch_size: Taille des lots pour calculer les CCFs (None = tout d'un coup).
        normalize: Normaliser les CCFs par leur amplitude max absolue.

    Returns:
        Tableau 2D des CCFs, de forme (n_spectra, len(v_grid)).
    """
    # Charger le masque (positions et poids des raies)
    mask = get_mask(mask_type)
    line_pos = mask[:, 0]
    line_weights = mask[:, 1]

    if verbose:
        print("Construction des masques CCF gaussiens (sparse)...")
    CCF_masks = build_CCF_masks_sparse(
        line_pos, line_weights, v_grid, wavegrid, window_size_velocity
    )

    n_spectra = spectra.shape[0]

    # Chemin sans batch
    if batch_size is None:
        CCFs = (CCF_masks.dot((spectra - 1.0).T)).T
        # Décalage à 0 par CCF
        CCFs -= np.min(CCFs, axis=1, keepdims=True)
        if normalize:
            CCFs = normalize_CCFs(CCFs)
        return CCFs

    # Traitement par lots
    CCFs_list = []
    n_batches = (n_spectra + batch_size - 1) // batch_size
    for batch_idx, start in enumerate(range(0, n_spectra, batch_size)):
        end = min(start + batch_size, n_spectra)
        batch_specs = spectra[start:end]
        batch_CCFs = (CCF_masks.dot((batch_specs - 1.0).T)).T
        batch_CCFs -= np.min(batch_CCFs, axis=1, keepdims=True)
        CCFs_list.append(batch_CCFs)
        if verbose:
            print(
                f"Batch {batch_idx + 1}/{n_batches} traité ({end}/{n_spectra} spectres)"
            )

    CCFs = np.vstack(CCFs_list)
    if normalize:
        CCFs = normalize_CCFs(CCFs)
    return CCFs


def normalize_CCFs(CCFs: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Normalise chaque CCF par son amplitude maximale absolue.

    Args:
        CCFs: Tableau 2D des CCFs (n_spectra, len(v_grid)).

    Returns:
        Tableau des CCFs normalisés, même forme que l'entrée.
    """
    denom = np.max(np.abs(CCFs), axis=1, keepdims=True)
    # Évite la division par zéro
    denom[denom == 0] = 1.0
    return CCFs / denom


# =============================================================
# ANALYSE DES CCFs (Fit gaussien, bisector, agrégation)
# =============================================================


def gaussian(
    x: NDArray[np.floating], c: float, k: float, x0: float, fwhm: float
) -> NDArray[np.floating]:
    """Profil gaussien paramétré par c, k, x0 et FWHM."""
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return c + k * np.exp(-((x - x0) ** 2) / (2 * sigma**2))


def fit_gaussian_ccf(
    v_grid: NDArray[np.floating], ccf: NDArray[np.floating]
) -> Dict[str, Any]:
    """
    Ajuste une gaussienne sur une CCF et calcule les paramètres dérivés.

    Args:
        v_grid: Grille de vitesses en m/s.
        ccf: Profil de CCF correspondant.

    Returns:
        Dictionnaire: {"c", "k", "x0", "fwhm", "rv", "depth", "span", "continuum", "amplitude", "popt"}.
        - En cas d'échec du fit, les champs sont remplis avec NaN et popt est un tableau NaN de taille 4.
    """
    # Estimations initiales simples
    c_init = float((ccf[0] + ccf[-1]) / 2)
    k_init = 0.0
    x0_init = float(v_grid[len(v_grid) // 2])
    fwhm_init = float((v_grid[-1] - v_grid[0]) / 5)

    # Cherche le point d'écart maximal à c_init (fonctionne pour absorption/émission)
    for i in range(len(ccf)):
        if abs(ccf[i] - c_init) > abs(k_init):
            k_init = float(ccf[i] - c_init)
            x0_init = float(v_grid[i])

    try:
        popt, _ = curve_fit(
            gaussian, v_grid, ccf, p0=[c_init, k_init, x0_init, fwhm_init], maxfev=10000
        )
        c, k, x0, fwhm = map(float, popt)
        span = compute_bisector_span(v_grid, ccf, c, k, x0, fwhm)
        return {
            "rv": x0,
            "depth": abs(k),
            "fwhm": fwhm,
            "span": span,
            "continuum": c,
            "amplitude": k,
            "popt": np.array([c, k, x0, fwhm], dtype=float),
            "c": c,
            "k": k,
            "x0": x0,
        }
    except Exception:
        nan4 = np.array([np.nan, np.nan, np.nan, np.nan], dtype=float)
        return {
            "rv": np.nan,
            "depth": np.nan,
            "fwhm": np.nan,
            "span": np.nan,
            "continuum": np.nan,
            "amplitude": np.nan,
            "popt": nan4,
            "c": np.nan,
            "k": np.nan,
            "x0": np.nan,
        }


def compute_bisector_span(
    v_grid: NDArray[np.floating],
    ccf: NDArray[np.floating],
    c: float,
    k: float,
    x0: float,
    fwhm: float,
) -> float:
    """
    Calcule le bisector span d'une CCF (inspiré de BIS_FIT2.cpp).

    Le span est défini comme la différence moyenne de la RV du bisector
    entre les niveaux de profondeur [0.1, 0.4] et [0.6, 0.9].

    Args:
        v_grid: Grille de vitesses (m/s).
        ccf: Profil CCF.
        c, k, x0, fwhm: Paramètres de la gaussienne ajustée.

    Returns:
        Valeur du bisector span (m/s) ou NaN si non calculable.
    """
    # Sécurité sur les paramètres
    if not np.isfinite([c, k, x0, fwhm]).all() or fwhm <= 0 or abs(k) < 1e-12:
        return float("nan")

    n = len(v_grid)
    nstep = 100
    margin = 5
    len_depth = nstep - 2 * margin + 1

    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    if sigma <= 0:
        return float("nan")

    # CCF normalisée (profondeur 0..1 environ)
    # norm_CCF ~ (ccf - c)/k remis en signe pour suivre l'implémentation d'origine
    norm_CCF = -c / k * (1 - ccf / c) if c != 0 else (ccf - c) / (k if k != 0 else 1)
    depth = np.array([(i + margin) / nstep for i in range(len_depth)], dtype=float)

    # Coefficients d'interpolation parabolique
    p0 = np.zeros(n - 1)
    p1 = np.zeros(n - 1)
    p2 = np.zeros(n - 1)

    for i in range(n - 1):
        vr = 0.5 * (v_grid[i] + v_grid[i + 1])
        dx = vr - x0
        exp_arg = np.exp(-(dx * dx) / (2 * sigma * sigma))
        dCCFdRV = -(dx) / (sigma * sigma) * exp_arg
        d2CCFdRV2 = ((dx * dx) / (sigma * sigma) - 1) / (sigma * sigma) * exp_arg

        if abs(dCCFdRV) > 1e-10:
            d2RVdCCF2 = -d2CCFdRV2 / (dCCFdRV**3)
            p2[i] = 0.5 * d2RVdCCF2

            dnorm = norm_CCF[i + 1] - norm_CCF[i]
            if abs(dnorm) > 1e-10:
                p1[i] = (
                    v_grid[i + 1]
                    - v_grid[i]
                    - p2[i] * (norm_CCF[i + 1] ** 2 - norm_CCF[i] ** 2)
                ) / dnorm
                p0[i] = v_grid[i] - p1[i] * norm_CCF[i] - p2[i] * (norm_CCF[i] ** 2)

    # Indice du maximum de la CCF normalisée
    ind_max = int(np.argmax(norm_CCF))

    # Calcul du bisector par niveaux de profondeur
    bis = np.zeros(len_depth)
    for ii in range(len_depth):
        i_b = ind_max
        i_r = ind_max

        # Recherche des indices gauche/droite où la CCF atteint la profondeur
        while i_b > 1 and norm_CCF[i_b] > depth[ii]:
            i_b -= 1
        while i_r < n - 2 and norm_CCF[i_r + 1] > depth[ii]:
            i_r += 1

        if i_b < len(p0) and i_r < len(p0):
            bis[ii] = (
                p0[i_b]
                + p0[i_r]
                + (p1[i_b] + p1[i_r]) * depth[ii]
                + (p2[i_b] + p2[i_r]) * (depth[ii] ** 2)
            ) * 0.5
        else:
            bis[ii] = np.nan

    # Span = RV_top - RV_bottom
    rv_top_indices = (depth >= 0.1) & (depth <= 0.4)
    rv_bottom_indices = (depth >= 0.6) & (depth <= 0.9)

    if np.any(rv_top_indices) and np.any(rv_bottom_indices):
        rv_top = np.nanmean(bis[rv_top_indices])
        rv_bottom = np.nanmean(bis[rv_bottom_indices])
        return float(rv_top - rv_bottom)
    return float("nan")


def analyze_ccfs(
    CCFs: NDArray[np.floating],
    v_grid: NDArray[np.floating],
    template: bool = False,
) -> Dict[str, NDArray[np.floating]]:
    """
    Analyse un ensemble de CCFs pour extraire RV, profondeur, FWHM et bisector span.

    Args:
        CCFs: Tableau 2D des CCFs (n_spectra, len(v_grid)).
        v_grid: Grille des vitesses en m/s.

    Returns:
        Dictionnaire des mesures, tableau 1D par clé (sauf "popt" en (n_spectra, 4)).
    """
    n_spectra = CCFs.shape[0]

    # Use plain np.ndarray typing for runtime simplicity and explicit float dtype
    results: Dict[str, np.ndarray] = {
        "rv": np.full(n_spectra, np.nan, dtype=np.float64),
        "depth": np.full(n_spectra, np.nan, dtype=np.float64),
        "fwhm": np.full(n_spectra, np.nan, dtype=np.float64),
        "span": np.full(n_spectra, np.nan, dtype=np.float64),
        "continuum": np.full(n_spectra, np.nan, dtype=np.float64),
        "amplitude": np.full(n_spectra, np.nan, dtype=np.float64),
        # Compatibility keys often expected by notebooks / older code
        "amp": np.full(n_spectra, np.nan, dtype=np.float64),
        "mu": np.full(n_spectra, np.nan, dtype=np.float64),
        "sigma": np.full(n_spectra, np.nan, dtype=np.float64),
        # popt stores the fitted parameters [c, k, x0, fwhm] per spectrum
        "popt": np.full((n_spectra, 4), np.nan, dtype=np.float64),
    }

    for i in range(n_spectra):
        ccf_i = CCFs[i]
        if template:
            # Si la CCF est majoritairement une "cuvette", on la renverse pour fitter une gaussienne vers le haut
            if np.isfinite(ccf_i).any():
                c_init = float((ccf_i[0] + ccf_i[-1]) / 2)
                up_span = float(np.nanmax(ccf_i) - c_init)
                down_span = float(c_init - np.nanmin(ccf_i))
                if down_span > up_span:
                    # Renversement autour de c_init: c' = 2*c_init - c
                    ccf_i = 2 * c_init - ccf_i
        params = fit_gaussian_ccf(v_grid, ccf_i)
        results["rv"][i] = params["rv"]
        results["depth"][i] = params["depth"]
        results["fwhm"][i] = params["fwhm"]
        results["span"][i] = params["span"]
        results["continuum"][i] = params.get("continuum", np.nan)
        results["amplitude"][i] = params.get("amplitude", np.nan)
        # compatibility: shorter names used in some notebooks
        results["amp"][i] = params.get("amplitude", np.nan)
        results["mu"][i] = params.get("x0", params.get("rv", np.nan))
        # convert fwhm -> sigma (std dev) for convenience
        fwhm_val = params.get("fwhm", np.nan)
        try:
            sigma_val = (
                float(fwhm_val) / (2 * np.sqrt(2 * np.log(2)))
                if np.isfinite(fwhm_val)
                else np.nan
            )
        except Exception:
            sigma_val = np.nan
        results["sigma"][i] = sigma_val
        results["popt"][i, :] = params.get(
            "popt", np.array([np.nan, np.nan, np.nan, np.nan], dtype=float)
        )

    return results


# =============================================================
# CONSTRUCTION DES MASQUES CCF (sparse) + UTILITAIRES NUMBA
# =============================================================


@njit(inline="always")
def _gamma_numba(v: float) -> np.float64:
    """
    Calcule le facteur de Lorentz pour une vitesse v (m/s).
    """
    C_LIGHT = 299_792_458  # m/s
    return np.sqrt((1 + v / C_LIGHT) / (1 - v / C_LIGHT))


@njit(inline="always")
def extend_wavegrid_numba(
    wavegrid: np.ndarray,
    v_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Étend une grille de longueurs d'onde à pas constant pour couvrir les décalages Doppler maximaux.

    Args:
        wavegrid: Tableau trié de longueurs d'onde à pas constant.
        v_grid: Tableau de vitesses radiales pour lesquelles préparer les décalages.

    Returns:
        wavegrid_extended: Grille complète incluant les extensions avant et après.
        wave_before: Portion avant la longueur d'onde minimale d'origine.
        wave_after: Portion après la longueur d'onde maximale d'origine.
    """
    SAFETY_MARGIN = 100.0
    step = wavegrid[1] - wavegrid[0]
    vmax = v_grid.max() + SAFETY_MARGIN
    vmin = v_grid.min() - SAFETY_MARGIN

    gamma_max = _gamma_numba(vmax)
    gamma_min = _gamma_numba(-vmin)

    wl_max = (wavegrid * gamma_max).max()
    wl_min = (wavegrid * gamma_min).min()

    wave_before = np.arange(wl_min, wavegrid.min(), step)
    wave_after = np.arange(wavegrid.max() + step, wl_max, step)

    total_len = len(wave_before) + len(wavegrid) + len(wave_after)
    extended = np.empty(total_len)
    extended[: len(wave_before)] = wave_before
    extended[len(wave_before) : len(wave_before) + len(wavegrid)] = wavegrid
    extended[len(wave_before) + len(wavegrid) :] = wave_after

    return extended, wave_before, wave_after


@njit(nopython=True, parallel=True)
def _count_per_velocity(
    line_pos,
    line_weights,
    v_grid,
    wavegrid_ext,
    window_size_velocity,
):
    """
    Compte le nombre d'entrées de la matrice creuse pour chaque vitesse avec un profil gaussien.
    Support considéré: ±4σ autour de chaque raie.
    """
    c = 299792458.0
    n_v = v_grid.shape[0]
    counts = np.zeros(n_v, np.int64)
    for i in prange(n_v):
        shift = ((1.0 + v_grid[i] / c) / (1.0 - v_grid[i] / c)) ** 0.5
        total = 0
        for j in range(line_pos.shape[0]):
            lam0 = line_pos[j]
            lam_c = lam0 * shift
            sigma = lam0 * (window_size_velocity / c)
            start = lam_c - 4.0 * sigma
            end = lam_c + 4.0 * sigma
            idx0 = int(np.searchsorted(wavegrid_ext, start))
            idx1 = int(np.searchsorted(wavegrid_ext, end))
            if idx0 < 0:
                idx0 = 0
            if idx1 >= wavegrid_ext.shape[0]:
                idx1 = wavegrid_ext.shape[0] - 1
            if idx1 >= idx0:
                total += idx1 - idx0 + 1
        counts[i] = total
    return counts


@njit(nopython=True, parallel=True)
def _fill_entries(
    line_pos,
    line_weights,
    v_grid,
    wavegrid_ext,
    window_size_velocity,
    offsets,
    rows,
    cols,
    vals,
):
    """
    Remplit les entrées de la matrice creuse (profil gaussien sur ±4σ).
    Chaque raie contribue selon exp(-0.5*((lambda-lam_c)/sigma)^2).
    """
    c = 299792458.0
    n_v = v_grid.shape[0]
    for i in prange(n_v):
        shift = ((1.0 + v_grid[i] / c) / (1.0 - v_grid[i] / c)) ** 0.5
        idx = offsets[i]
        for j in range(line_pos.shape[0]):
            lam0 = line_pos[j]
            weight = line_weights[j]
            lam_c = lam0 * shift
            sigma = lam0 * (window_size_velocity / c)
            start = lam_c - 4.0 * sigma
            end = lam_c + 4.0 * sigma
            idx0 = int(np.searchsorted(wavegrid_ext, start))
            idx1 = int(np.searchsorted(wavegrid_ext, end))
            if idx0 < 0:
                idx0 = 0
            if idx1 >= wavegrid_ext.shape[0]:
                idx1 = wavegrid_ext.shape[0] - 1
            for k in range(idx0, idx1 + 1):
                lam = wavegrid_ext[k]
                x = (lam - lam_c) / sigma
                val = weight * np.exp(-0.5 * x * x)
                rows[idx] = i
                cols[idx] = k
                vals[idx] = val
                idx += 1


def build_CCF_masks_sparse(
    line_pos: NDArray[np.floating],
    line_weights: NDArray[np.floating],
    v_grid: NDArray[np.floating],
    wavegrid: NDArray[np.floating],
    window_size_velocity: float,
) -> csr_matrix:
    """
    Construit une matrice creuse CSR de masques CCF gaussiens.

    Args:
        line_pos: Positions (longueurs d'onde) des raies.
        line_weights: Poids des raies.
        v_grid: Grille RV (m/s).
        wavegrid: Grille de longueurs d'onde au pas constant.
        window_size_velocity: Sigma du profil gaussien en m/s.

    Returns:
        Matrice CSR de dimension (len(v_grid), len(wavegrid)).
    """
    # Extension doppler de la grille de longueurs d'onde
    wavegrid_ext, wave_before, wave_after = extend_wavegrid_numba(wavegrid, v_grid)
    n_v = v_grid.size

    # Compter les entrées par vitesse
    counts = _count_per_velocity(
        line_pos, line_weights, v_grid, wavegrid_ext, window_size_velocity
    )

    # Offsets cumulés
    offsets = np.empty(n_v, np.int64)
    total = 0
    for i in range(n_v):
        offsets[i] = total
        total += counts[i]

    # Triplets (i, j, val)
    rows = np.empty(total, dtype=np.int64)
    cols = np.empty(total, dtype=np.int64)
    vals = np.empty(total, dtype=np.float64)

    # Remplir les entrées
    _fill_entries(
        line_pos,
        line_weights,
        v_grid,
        wavegrid_ext,
        window_size_velocity,
        offsets,
        rows,
        cols,
        vals,
    )

    # Assembler et recadrer sur la grille d'origine
    coo = coo_matrix((vals, (rows, cols)), shape=(n_v, wavegrid_ext.size))
    CCF = coo.tocsr()
    s = len(wave_before)
    e = s + len(wavegrid)
    return CCF[:, s:e]


def compute_CCFs_template(
    specs: np.ndarray,
    template: np.ndarray,
    wavegrid: np.ndarray,
    v_grid: np.ndarray,
    dtype: torch.dtype = torch.float32,
    verbose: bool = False,
    batch_size_specs: int = None,
) -> np.ndarray:
    """
    Version optimisée : pré-calcule les templates décalés une seule fois,
    puis traite les spectres par batches avec calculs vectorisés.

    OPTIMISATIONS CLÉS :
    1. Pré-calcul des templates décalés (évite la recomputation)
    2. Traitement par batches de spectres
    3. Utilisation de torch.mm pour les produits matriciels
    4. Gestion optimale de la mémoire GPU
    """
    # Sélection du device et libération de la mémoire GPU si dispo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    n_specs = specs.shape[0]
    L = wavegrid.shape[0]
    n_v = len(v_grid)

    # Gère le cas d'un template par spectre ([N, L]) vs un seul template ([L])
    def _ndim(arr):
        if isinstance(arr, np.ndarray):
            return arr.ndim
        if isinstance(arr, torch.Tensor):
            return arr.dim()
        return 1

    template_ndim = _ndim(template)
    is_batched_template = template_ndim == 2
    if is_batched_template:
        t_shape = (
            template.shape
            if isinstance(template, np.ndarray)
            else tuple(template.size())
        )
        assert t_shape[0] == n_specs, (
            f"Template batched doit avoir N identique à spectra: {t_shape[0]} != {n_specs}"
        )
        assert t_shape[1] == L, (
            f"Template batched doit avoir la même longueur que wavegrid: {t_shape[1]} != {L}"
        )

    # Calcul de la mémoire GPU utilisable
    safety_factor = 0.85
    if torch.cuda.is_available():
        free_mem = get_free_memory() * safety_factor
    else:
        # Sur CPU, on ne se base pas sur la VRAM. On traitera toutes les vitesses d'un coup par défaut.
        free_mem = float("inf")

    if verbose:
        print(f"Mémoire GPU libre : {free_mem / 1e9:.2f} GB")
        print(f"Traitement de {n_specs} spectres avec {n_v} vitesses")

    # Détermination de la taille de batch pour les vitesses
    nv_max = (
        get_max_nv(L, free_mem, dtype) if torch.cuda.is_available() else len(v_grid)
    )
    if verbose:
        print(f"Taille max de vitesses par batch : {nv_max}")

    # Détermination de la taille de batch pour les spectres
    if batch_size_specs is None:
        # Estimation basée sur la mémoire disponible
        if dtype == torch.float16:
            bytes_per_element = 2
        elif dtype == torch.float32:
            bytes_per_element = 4
        else:
            bytes_per_element = 8
        memory_per_spec = L * bytes_per_element
        memory_per_vbatch = nv_max * L * bytes_per_element  # pour les templates décalés
        if torch.cuda.is_available():
            remaining_memory = max(0.0, free_mem - memory_per_vbatch)
            batch_size_specs = max(
                1, int(remaining_memory // memory_per_spec // 2)
            )  # facteur 2 de sécurité
        else:
            # Sur CPU, batch raisonnable par défaut si non fourni
            batch_size_specs = min(256, specs.shape[0])

    batch_size_specs = min(batch_size_specs, n_specs)

    if verbose:
        print(f"Taille de batch pour les spectres : {batch_size_specs}")

    # Découpage en batches pour les vitesses
    v_batches = [v_grid[i : i + nv_max] for i in range(0, len(v_grid), nv_max)]

    # Découpage en batches pour les spectres
    spec_batches = [
        specs[i : i + batch_size_specs] for i in range(0, n_specs, batch_size_specs)
    ]

    CCFs = np.zeros(
        (n_specs, n_v), dtype=np.float32 if dtype == torch.float32 else np.float64
    )

    total_start = time.time()

    # Boucle sur les batches de vitesses
    for v_idx, v_batch in enumerate(v_batches):
        if verbose:
            print(
                f"Traitement du batch de vitesses {v_idx + 1}/{len(v_batches)} ({len(v_batch)} vitesses)"
            )

        # Préparation des constantes du batch de vitesses
        v_start_time = time.time()
        Bv = len(v_batch)
        wave_t = (
            torch.as_tensor(wavegrid, dtype=dtype).to(device)
            if not isinstance(wavegrid, torch.Tensor)
            else wavegrid.to(dtype=dtype, device=device)
        )  # (L,)
        v_batch_t = (
            torch.as_tensor(v_batch, dtype=dtype).to(device)
            if not isinstance(v_batch, torch.Tensor)
            else v_batch.to(dtype=dtype, device=device)
        )  # (Bv,)

        # Si template unique (1D), pré-calcul des templates décalés pour tout ce batch de vitesses
        if not is_batched_template:
            template_t = (
                torch.as_tensor(template, dtype=dtype).to(device)
                if not isinstance(template, torch.Tensor)
                else template.to(dtype=dtype, device=device)
            )  # (L,)
            wave_batch = wave_t.unsqueeze(0).expand(Bv, -1).contiguous()  # (Bv, L)
            template_batch = (
                template_t.unsqueeze(0).expand(Bv, -1).contiguous()
            )  # (Bv, L)
            shifted_templates = shift_spectra_linear(
                spectra=template_batch,
                wavegrid=wave_batch,
                velocities=v_batch_t,
                extrapolate="constant",
                return_mask=False,
            )  # (Bv, L)
            templates_gpu = shifted_templates
            template_time = time.time() - v_start_time
            if verbose:
                print(f"  Temps de calcul des templates décalés : {template_time:.3f}s")

        # Boucle sur les batches de spectres
        for s_idx, spec_batch in enumerate(spec_batches):
            if verbose and len(spec_batches) > 1:
                print(
                    f"    Batch de spectres {s_idx + 1}/{len(spec_batches)} ({len(spec_batch)} spectres)"
                )

            # Transfert des spectres vers GPU
            if isinstance(spec_batch, torch.Tensor):
                specs_gpu = spec_batch.to(dtype=dtype, device=device)
            else:
                specs_gpu = torch.as_tensor(spec_batch, dtype=dtype).to(device)
            # (n_batch, L)
            specs_centered = specs_gpu - 1.0  # cohérent avec compute_CCFs

            if is_batched_template:
                # Sélection des templates correspondant à ce batch de spectres
                s_start = s_idx * batch_size_specs
                s_end = s_start + len(spec_batch)
                temp_slice = template[s_start:s_end]
                temp_batch = (
                    temp_slice.to(dtype=dtype, device=device)
                    if isinstance(temp_slice, torch.Tensor)
                    else torch.as_tensor(temp_slice, dtype=dtype).to(device)
                )  # (S, L)

                # Méthode mémoire-sûre: on décale par vitesse (colonne par colonne)
                S = temp_batch.shape[0]
                wave_batch_S = wave_t.unsqueeze(0).expand(S, -1).contiguous()  # (S, L)
                cols = []
                for j in range(Bv):
                    vS = v_batch_t[j].repeat(S)
                    shifted_S = shift_spectra_linear(
                        spectra=temp_batch,
                        wavegrid=wave_batch_S,
                        velocities=vS,
                        extrapolate="constant",
                        return_mask=False,
                    )  # (S, L)
                    col = torch.sum(shifted_S * specs_centered, dim=1)  # (S,)
                    cols.append(col)
                    del shifted_S, vS, col
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                ccf_batch = torch.stack(cols, dim=1)  # (S, Bv)
                del temp_batch, wave_batch_S, cols
            else:
                # Calcul vectorisé de la CCF: (nv_batch, L) @ (n_batch, L).T = (nv_batch, n_batch)
                ccf_batch = torch.mm(
                    templates_gpu, specs_centered.T
                )  # (nv_batch, n_batch)

            # Stockage des résultats
            v_start = v_idx * nv_max
            v_end = v_start + len(v_batch)
            s_start = s_idx * batch_size_specs
            s_end = s_start + len(spec_batch)

            # Pour template batched, ccf_batch est (S, Bv) -> assigner directement.
            if is_batched_template:
                CCFs[s_start:s_end, v_start:v_end] = ccf_batch.cpu().numpy()
            else:
                # Pour template unique, ccf_batch est (nv_batch, n_batch) -> transpose
                CCFs[s_start:s_end, v_start:v_end] = ccf_batch.T.cpu().numpy()

            # Nettoyage
            del specs_gpu, specs_centered, ccf_batch

        # Nettoyage des templates décalés
        if not is_batched_template:
            del shifted_templates, templates_gpu, template_batch, wave_batch, template_t
        del v_batch_t, wave_t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_time = time.time() - total_start
    if verbose:
        print(
            f"Temps total optimisé : {total_time:.2f}s pour {n_specs} spectres et {n_v} vitesses"
        )
        speedup_estimate = (n_specs * n_v * L) / (total_time * 1e9)  # GFLOPS estimate
        print(f"Performance estimée : {speedup_estimate:.1f} GFLOPS")

    return CCFs


if __name__ == "__main__":
    data = np.load(
        "data/npz_datasets/dataset_1000specs_5000_5010_Kp1e-1_P100.npz",
        allow_pickle=True,
    )
    spectra = data["spectra"]
    wavegrid = data["wavegrid"]

    res = get_full_ccf_analysis(
        spectra=spectra,
        wavegrid=wavegrid,
        v_grid=np.linspace(-20000, 20000, 100),
        window_size_velocity=820.0,
        mask_type="G2",
        verbose=False,
        batch_size=100,
        normalize=True,
    )

    plt.figure(figsize=(10, 6))
    plt.plot(res["rv"], "o", markersize=2, label="Points RV vs Depth")
    plt.xlabel("Vitesse Radiale (m/s)")
    plt.ylabel("Profondeur (absorption)")
    plt.title("Analyse des CCFs - RV vs Profondeur")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig("ccf_analysis_rv_vs_depth.png", dpi=300)
    plt.show()
    print("Analyse CCF terminée et graphique enregistré.")
