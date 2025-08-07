import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
from scipy.sparse import coo_matrix, csr_matrix
import torch
from src.utils import get_mask
from scipy.optimize import curve_fit


@njit(inline="always")
def _gamma_numba(v: float) -> np.float64:
    """
    Calcule le facteur de Lorentz pour une vitesse donnée.

    Args:
        v: Vitesse en m/s.

    Returns:
        Facteur de Lorentz en np.float64.
    """
    C_LIGHT = 299_792_458  # Vitesse de la lumière en m/s
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
        wave_before: Portion de la grille avant la longueur d'onde minimale d'origine.
        wave_after: Portion de la grille après la longueur d'onde maximale d'origine.
    """
    SAFETY_MARGIN = 100
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
    wavegrid_step,
    begin_wave,
):
    """
    Compte le nombre d'entrées de la matrice creuse pour chaque vitesse en profil gaussien.
    On prend un support ±4σ autour de chaque raie.
    """
    c = 299792458.0
    n_v = v_grid.shape[0]
    counts = np.zeros(n_v, np.int64)
    for i in prange(n_v):
        shift = 1.0 + v_grid[i] / c  # approximation non-relativiste
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
    wavegrid_step,
    begin_wave,
    offsets,
    rows,
    cols,
    vals,
):
    """
    Remplit les entrées de la matrice creuse en profil gaussien.
    Chaque raie contribue selon exp(-0.5*((lambda-lam_c)/sigma)^2).
    """
    c = 299792458.0
    n_v = v_grid.shape[0]
    for i in prange(n_v):
        shift = 1.0 + v_grid[i] / c
        idx_base = offsets[i]
        idx = idx_base
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
    line_pos: np.ndarray,
    line_weights: np.ndarray,
    v_grid: np.ndarray,
    wavegrid: np.ndarray,
    window_size_velocity: float,
) -> csr_matrix:
    """
    Construit une matrice creuse CSR de masques CCF gaussiens.
    Args:
      - line_pos, line_weights: positions et poids des raies
      - v_grid: grille RV (m/s)
      - wavegrid: grille de longueurs d'onde (constant step)
      - window_size_velocity: sigma du profil gaussien en m/s
    Returns:
      CSR matrix de dimension (len(v_grid), len(wavegrid))
    """
    # extension doppler
    wavegrid_ext, wave_before, wave_after = extend_wavegrid_numba(wavegrid, v_grid)
    step = wavegrid[1] - wavegrid[0]
    begin = wavegrid_ext[0]
    n_v = v_grid.size

    # compter les entrées
    counts = _count_per_velocity(
        line_pos, line_weights, v_grid, wavegrid_ext, window_size_velocity, step, begin
    )

    # offsets
    offsets = np.empty(n_v, np.int64)
    total = 0
    for i in range(n_v):
        offsets[i] = total
        total += counts[i]

    # allouer triplets
    rows = np.empty(total, dtype=np.int64)
    cols = np.empty(total, dtype=np.int64)
    vals = np.empty(total, dtype=np.float64)

    # remplir
    _fill_entries(
        line_pos,
        line_weights,
        v_grid,
        wavegrid_ext,
        window_size_velocity,
        step,
        begin,
        offsets,
        rows,
        cols,
        vals,
    )

    # assembler et recadrer
    coo = coo_matrix((vals, (rows, cols)), shape=(n_v, wavegrid_ext.size))
    CCF = coo.tocsr()
    s = len(wave_before)
    e = s + len(wavegrid)
    return CCF[:, s:e]


def compute_CCFs(
    spectra: np.ndarray,
    v_grid: np.ndarray,
    wavegrid: np.ndarray,
    window_size_velocity: float,
    mask_type: str = "G2",
    verbose: bool = False,
    batch_size: int = None,
    normalize: bool = True,
):
    # charger masque
    mask = get_mask(mask_type)
    line_pos = mask[:, 0]
    line_weights = mask[:, 1]
    if verbose:
        print("Construction CCF gaussian sparse...")
    CCF_masks = build_CCF_masks_sparse(
        line_pos, line_weights, v_grid, wavegrid, window_size_velocity
    )

    n_spectra = spectra.shape[0]

    # Si pas de batch_size défini, traiter tout d'un coup
    if batch_size is None:
        CCFs = (CCF_masks.dot((spectra - 1.0).T)).T
        CCFs -= np.min(CCFs, axis=1, keepdims=True)
        return CCFs

    # Traitement par batch
    CCFs_list = []
    n_batches = (n_spectra + batch_size - 1) // batch_size
    for batch_idx, i in enumerate(range(0, n_spectra, batch_size)):
        end_idx = min(i + batch_size, n_spectra)
        batch_specs = spectra[i:end_idx]
        batch_CCFs = (CCF_masks.dot((batch_specs - 1.0).T)).T
        batch_CCFs -= np.min(batch_CCFs, axis=1, keepdims=True)
        CCFs_list.append(batch_CCFs)

        if verbose:
            print(
                f"Batch {batch_idx + 1}/{n_batches} traité ({end_idx}/{n_spectra} spectres)"
            )
    CCFs = np.vstack(CCFs_list)
    if normalize:
        CCFs = normalize_CCFs(CCFs)
    return CCFs


def normalize_CCFs(CCFs: np.ndarray) -> np.ndarray:
    """
    Normalise les CCFs en divisant par le max / min de chaque CCF.

    Args:
        CCFs (np.ndarray): Tableau des CCFs à normaliser.

    Returns:
        np.ndarray: Tableau des CCFs normalisés.
    """

    return CCFs / np.max(np.abs(CCFs), axis=1, keepdims=True)


def gaussian(x, c, k, x0, fwhm):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return c + k * np.exp(-((x - x0) ** 2) / (2 * sigma**2))


def fitian_ccf(v_grid: np.ndarray, ccf: np.ndarray) -> dict:
    """
    Ajuste une gaussienne sur une CCF et calcule les paramètres.

    Args:
        v_grid: Grille de vitesses en m/s
        ccf: Profil CCF

    Returns:
        dict: Paramètres ajustés {c, k, x0, fwhm, rv, depth, span}
    """

    # Estimation initiale des paramètres
    c_init = (ccf[0] + ccf[-1]) / 2
    k_init = 0
    x0_init = v_grid[len(v_grid) // 2]
    fwhm_init = (v_grid[-1] - v_grid[0]) / 5

    # Trouver le minimum (pour une CCF en absorption)
    for i in range(len(ccf)):
        if abs(ccf[i] - c_init) > abs(k_init):
            k_init = ccf[i] - c_init
            x0_init = v_grid[i]

    try:
        popt, _ = curve_fit(
            gaussian, v_grid, ccf, p0=[c_init, k_init, x0_init, fwhm_init], maxfev=10000
        )
        c, k, x0, fwhm = popt

        # Calcul du bissector span
        span = calculate_bissector_span(v_grid, ccf, c, k, x0, fwhm)

        return {
            "rv": x0,
            "depth": abs(k),
            "fwhm": fwhm,
            "span": span,
            "continuum": c,
            "amplitude": k,
            "popt": popt,
        }
    except Exception:
        return {
            "rv": np.nan,
            "depth": np.nan,
            "fwhm": np.nan,
            "span": np.nan,
            "continuum": np.nan,
            "amplitude": np.nan,
        }


def calculate_bissector_span(
    v_grid: np.ndarray, ccf: np.ndarray, c: float, k: float, x0: float, fwhm: float
) -> float:
    """
    Calcule le bissector span d'une CCF selon l'algorithme BIS_FIT2.cpp

    Args:
        v_grid: Grille de vitesses
        ccf: Profil CCF
        c, k, x0, fwhm: Paramètres gaussiens ajustés

    Returns:
        float: Bissector span
    """
    n = len(v_grid)
    nstep = 100
    margin = 5
    len_depth = nstep - 2 * margin + 1

    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

    # CCF normalisée
    norm_CCF = -c / k * (1 - ccf / c)
    depth = np.array([(i + margin) / nstep for i in range(len_depth)])

    # Calcul des coefficients de l'interpolation parabolique
    p0 = np.zeros(n - 1)
    p1 = np.zeros(n - 1)
    p2 = np.zeros(n - 1)

    for i in range(n - 1):
        vr = (v_grid[i] + v_grid[i + 1]) / 2
        dCCFdRV = -(vr - x0) / sigma**2 * np.exp(-((vr - x0) ** 2) / (2 * sigma**2))
        d2CCFdRV2 = (
            ((vr - x0) ** 2 / sigma**2 - 1)
            / sigma**2
            * np.exp(-((vr - x0) ** 2) / (2 * sigma**2))
        )

        if abs(dCCFdRV) > 1e-10:
            d2RVdCCF2 = -d2CCFdRV2 / dCCFdRV**3
            p2[i] = d2RVdCCF2 / 2

            if abs(norm_CCF[i + 1] - norm_CCF[i]) > 1e-10:
                p1[i] = (
                    v_grid[i + 1]
                    - v_grid[i]
                    - p2[i] * (norm_CCF[i + 1] ** 2 - norm_CCF[i] ** 2)
                ) / (norm_CCF[i + 1] - norm_CCF[i])
                p0[i] = v_grid[i] - p1[i] * norm_CCF[i] - p2[i] * norm_CCF[i] ** 2

    # Trouver le maximum de la CCF normalisée
    ind_max = np.argmax(norm_CCF)

    # Calcul du bissector
    bis = np.zeros(len_depth)
    for i in range(len_depth):
        i_b = ind_max
        i_r = ind_max

        # Recherche des indices gauche et droite
        while i_b > 1 and norm_CCF[i_b] > depth[i]:
            i_b -= 1
        while i_r < n - 2 and norm_CCF[i_r + 1] > depth[i]:
            i_r += 1

        if i_b < len(p0) and i_r < len(p0):
            bis[i] = (
                p0[i_b]
                + p0[i_r]
                + (p1[i_b] + p1[i_r]) * depth[i]
                + (p2[i_b] + p2[i_r]) * depth[i] ** 2
            ) / 2

    # Calcul du span (différence entre haut et bas du bissector)
    rv_top_indices = (depth >= 0.1) & (depth <= 0.4)
    rv_bottom_indices = (depth >= 0.6) & (depth <= 0.9)

    if np.any(rv_top_indices) and np.any(rv_bottom_indices):
        rv_top = np.mean(bis[rv_top_indices])
        rv_bottom = np.mean(bis[rv_bottom_indices])
        span = rv_top - rv_bottom
    else:
        span = np.nan

    return span


def analyze_CCFs(CCFs: np.ndarray, v_grid: np.ndarray) -> dict:
    """
    Analyse un ensemble de CCFs pour extraire RV, profondeur, FWHM et bissector span.

    Args:
        CCFs: Tableau 2D des CCFs (n_spectra, n_velocities)
        v_grid: Grille de vitesses en m/s

    Returns:
        dict: Dictionnaire avec les mesures pour chaque spectre
    """
    n_spectra = CCFs.shape[0]

    results = {
        "rv": np.zeros(n_spectra),
        "depth": np.zeros(n_spectra),
        "fwhm": np.zeros(n_spectra),
        "span": np.zeros(n_spectra),
        "continuum": np.zeros(n_spectra),
        "amplitude": np.zeros(n_spectra),
        "popt": np.zeros((n_spectra, 4)),
    }

    for i in range(n_spectra):
        params = fitian_ccf(v_grid, CCFs[i])
        for key in results:
            results[key][i] = params[key]

    return results


# Point d'entrée du script: appel de la fonction compute_ccf et tracé
if __name__ == "__main__":
    from src.plots_aestra import plot_ccf_analysis

    torch.cuda.empty_cache()  # Nettoyage de la mémoire GPU

    # Hyperparamètres
    verbose: bool = True
    v_grid_max: int = 20000
    v_grid_step: int = 100
    v_grid: np.ndarray = np.arange(-v_grid_max, v_grid_max, v_grid_step)
    window_size_velocity: int = 820  # Taille de la fenêtre en espace de vitesse en m/s

    # Charger les données
    dset = np.load(
        "data/soap_gpu_paper/dataset_1000specs_5000_6000_Kp10_P100_auto_safe.npz"
    )
    spectra = dset["cube"]
    print(f"Spectres min/max: {spectra.min():.3f} / {spectra.max():.3f}")
    print(f"Spectres moyenne: {spectra.mean():.3f}")

    wavegrid = dset["wavegrid"]

    # Calculer les CCFs
    CCFs = compute_CCFs(
        spectra=spectra,
        wavegrid=wavegrid,
        v_grid=v_grid,
        mask_type="G2",
        window_size_velocity=410,
        verbose=verbose,
        batch_size=32,
        normalize=True,
    )

    # Analyser les CCFs
    res = analyze_CCFs(CCFs, v_grid)

    # Visualisation détaillée de la première CCF avec analyse complète
    spectrum_idx = 0
    analysis_first = {
        "rv": res["rv"][spectrum_idx],
        "depth": res["depth"][spectrum_idx],
        "fwhm": res["fwhm"][spectrum_idx],
        "span": res["span"][spectrum_idx],
        "continuum": res["continuum"][spectrum_idx],
        "amplitude": res["amplitude"][spectrum_idx],
        "popt": res["popt"][spectrum_idx],
    }

    plot_ccf_analysis(
        v_grid=v_grid,
        ccf=CCFs[spectrum_idx],
        analysis_results=analysis_first,
        spectrum_idx=spectrum_idx,
        show_plot=True,
    )

    # Plot de la série temporelle des RV
    plt.figure(figsize=(18, 6))
    plt.plot(res["rv"], "ko-", markersize=3, label="RV mesurées")
    plt.grid(True, alpha=0.3)
    plt.xlabel("Index du spectre")
    plt.ylabel("Vitesse radiale (m/s)")
    plt.title("Série temporelle des vitesses radiales extraites des CCFs")
    plt.legend()
    plt.show()

    # Statistiques globales
    print("\n=== Statistiques des mesures CCF ===")
    print(f"Nombre de spectres analysés: {len(res['rv'])}")
    print(f"RV moyenne: {np.nanmean(res['rv']):.2f} ± {np.nanstd(res['rv']):.2f} m/s")
    print(
        f"FWHM moyenne: {np.nanmean(res['fwhm']):.2f} ± {np.nanstd(res['fwhm']):.2f} m/s"
    )
    print(
        f"Bissector span moyen: {np.nanmean(res['span']):.2f} ± {np.nanstd(res['span']):.2f} m/s"
    )
    print(
        f"Profondeur moyenne: {np.nanmean(res['depth']):.4f} ± {np.nanstd(res['depth']):.4f}"
    )
