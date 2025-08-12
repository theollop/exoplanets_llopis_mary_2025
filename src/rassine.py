from __future__ import annotations
from __future__ import annotations

"""
RASSINE-based normalization utilities.

This module exposes a single public function:
  - normalize_with_rassine(wave, flux, config=None) -> np.ndarray

It uses the local RASSINE implementation found under ../Rassine_public
to estimate the stellar continuum (without any GUI) and returns the
normalized spectrum (flux / continuum).

If RASSINE cannot be imported or fails at runtime, it falls back to a
simple median normalization (flux / median(flux)).

Optionally, a batch helper is provided:
  - normalize_batch_with_rassine(wave, fluxes, config=None) -> np.ndarray

Constraints honored:
  - No GUI or plots are produced.
  - No temporary files are created.
  - Inputs are validated and documented.
"""

from typing import Dict, Optional

import os
import sys
import types
import importlib
import numpy as np


def _fallback_normalize(flux: np.ndarray) -> np.ndarray:
    """Return a safe median normalization as a fallback.

    If the median is non-positive or not finite, use the mean; if still
    invalid, return the input unchanged to avoid crashing.
    """
    f = np.asarray(flux, dtype=float)
    med = np.nanmedian(f)
    if not np.isfinite(med) or med <= 0:
        mean = np.nanmean(f)
        if not np.isfinite(mean) or mean == 0:
            return f.copy()
        return f / mean
    return f / med


def _default_config() -> Dict[str, object]:
    """Lightweight, conservative defaults compatible with RASSINE internals.

    Notes
    -----
    We avoid any interactive/auto-tuning paths and use a simple, robust
    setup that relies only on functions from Rassine_functions.py.
    """
    return {
        "axes_stretching": 7.0,  # fixed numeric stretch to avoid auto paths
        "vicinity_local_max": 5,
        "smoothing_box": 3,
        "smoothing_kernel": "gaussian",
        "denoising_dist": 3,
        "outputs_interpolation_save": "linear",
        "outputs_denoising_save": "undenoised",
        "float_precision": "float64",
    }


def _import_rassine_functions():
    """Import Rassine_functions from local Rassine_public safely.

    Strategy:
    1) Try a regular import.
    2) If it fails (often due to forced Qt5Agg backend), load the file
       and execute a sanitized version with matplotlib-related imports
       removed to avoid GUI/backend requirements.
    """
    rassine_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Rassine_public"))
    if rassine_dir not in sys.path:
        sys.path.insert(0, rassine_dir)

    try:
        return importlib.import_module("Rassine_functions")
    except Exception:
        # Fallback: sanitize source and exec into a module
        src_path = os.path.join(rassine_dir, "Rassine_functions.py")
        with open(src_path, "r", encoding="utf-8") as f:
            src = f.read()

        # Drop GUI/backend lines to avoid Qt dependency
        filtered_lines = []
        for line in src.splitlines():
            l = line.strip()
            if l.startswith("import matplotlib"):
                continue
            if "matplotlib.use(" in l:
                continue
            if l.startswith("from matplotlib"):
                continue
            filtered_lines.append(line)
        sanitized_src = "\n".join(filtered_lines)

        module = types.ModuleType("rassine_functions_sanitized")
        module.__file__ = src_path
        # Provide a minimal matplotlib shim if anything slips through
        module.matplotlib = types.SimpleNamespace(use=lambda *a, **k: None)
        exec(compile(sanitized_src, src_path, "exec"), module.__dict__)
        return module


def normalize_with_rassine(
    wave: np.ndarray,
    flux: np.ndarray,
    config: Optional[Dict[str, object]] = None,
) -> np.ndarray:
    """Normalize a spectrum using local RASSINE internals (no GUI, no files).

    Parameters
    ----------
    wave : np.ndarray
        1D wavelength array (same length as flux).
    flux : np.ndarray
        1D flux array.
    config : dict | None
        Optional RASSINE-like configuration. Unspecified keys fall back to
        conservative defaults defined in this module.

    Returns
    -------
    np.ndarray
        Normalized spectrum (same shape as `wave`). On failure, a median
        normalization is returned.
    """
    if not isinstance(wave, np.ndarray) or not isinstance(flux, np.ndarray):
        raise TypeError("wave and flux must be numpy.ndarray")
    if wave.ndim != 1 or flux.ndim != 1:
        raise ValueError("wave and flux must be 1D arrays")
    if wave.size != flux.size:
        raise ValueError("wave and flux must have the same length")
    if wave.size == 0:
        return np.array([], dtype=float)

    cfg = _default_config()
    if config:
        cfg.update(dict(config))

    try:
        ras = _import_rassine_functions()
    except Exception:
        return _fallback_normalize(flux)

    try:
        grid = np.asarray(wave, dtype=float).copy()
        spectrei = np.asarray(flux, dtype=float).copy()

        # Replace non-finite entries by neighbor average
        if np.any(~np.isfinite(grid)):
            bad = np.where(~np.isfinite(grid))[0]
            for j in bad:
                if 0 < j < len(grid) - 1:
                    grid[j] = 0.5 * (grid[j - 1] + grid[j + 1])
        if np.any(~np.isfinite(spectrei)):
            bad = np.where(~np.isfinite(spectrei))[0]
            for j in bad:
                if 0 < j < len(spectrei) - 1:
                    spectrei[j] = 0.5 * (spectrei[j - 1] + spectrei[j + 1])

        # Sort and remember original order
        order = np.argsort(grid)
        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(order.size)
        grid = grid[order]
        spectrei = spectrei[order]

        # Basic cleanup similar to RASSINE
        spectrei[spectrei < 0] = 0.0
        spectrei = ras.empty_ccd_gap(grid, spectrei, left=None, right=None)

        # Normalize axes scale
        len_x = float(grid.max() - grid.min())
        len_y = float(np.max(spectrei) - np.min(spectrei))
        if len_x <= 0 or not np.isfinite(len_x) or not np.isfinite(len_y) or len_y == 0:
            return _fallback_normalize(spectrei)[inv_order]
        spectre = spectrei / (len_y / len_x)

        # Optional smoothing
        box = int(cfg.get("smoothing_box", 0) or 0)
        kernel = str(cfg.get("smoothing_kernel", "gaussian"))
        if box > 0:
            spectre = ras.smooth(spectre, box_pts=box, shape=kernel)

        # Stretching (fixed numeric)
        try:
            stretch_val = float(cfg.get("axes_stretching", 7.0))
        except Exception:
            stretch_val = 7.0
        if stretch_val <= 0:
            stretch_val = 7.0
        spectre = spectre / stretch_val

        # Anchor detection
        vicinity = int(cfg.get("vicinity_local_max", 5))
        idx, anchor_flux = ras.local_max(spectre, vicinity)
        idx = idx.astype(int)
        if idx.size == 0:
            return _fallback_normalize(spectrei)[inv_order]
        wave_anchors = grid[idx]
        flux_anchors = anchor_flux.copy()

        # Ensure edges
        if flux_anchors[0] < spectre[0]:
            wave_anchors = np.insert(wave_anchors, 0, grid[0])
            flux_anchors = np.insert(flux_anchors, 0, spectre[0])
            idx = np.insert(idx, 0, 0)
        if flux_anchors[-1] < spectre[-1]:
            wave_anchors = np.append(wave_anchors, grid[-1])
            flux_anchors = np.append(flux_anchors, spectre[-1])
            idx = np.append(idx, len(spectre) - 1)

        # Remove obvious cosmic anchors using rolling IQR
        try:
            import pandas as pd  # ensured by project requirements

            med = np.ravel(pd.DataFrame(flux_anchors).rolling(10, center=True).quantile(0.50))
            iq = (
                np.ravel(pd.DataFrame(flux_anchors).rolling(10, center=True).quantile(0.75))
                - med
            )
            iq[np.isnan(iq)] = np.nanmax(spectre)
            med[np.isnan(med)] = np.nanmax(spectre)
            cosmic = flux_anchors > med + 20.0 * iq
            if np.any(cosmic):
                mask = ~cosmic
                wave_anchors = wave_anchors[mask]
                flux_anchors = flux_anchors[mask]
                idx = idx[mask]
        except Exception:
            pass

        if wave_anchors.size < 2:
            return _fallback_normalize(spectrei)[inv_order]

        # Denoise anchors with local averaging
        dn = int(cfg.get("denoising_dist", 3))
        if dn > 0:
            flux_anchor_denoised = []
            for k in idx:
                a = max(0, int(k) - dn)
                b = min(len(spectre), int(k) + dn + 1)
                flux_anchor_denoised.append(float(np.mean(spectre[a:b])))
            flux_anchor_denoised = np.array(flux_anchor_denoised, dtype=float)
        else:
            flux_anchor_denoised = flux_anchors.copy()

        # Continuum via RASSINE helper
        interp_choice = str(cfg.get("outputs_interpolation_save", "linear"))
        denoise_choice = str(cfg.get("outputs_denoising_save", "undenoised"))

        c1, c3, c1d, c3d = ras.make_continuum(
            wave_anchors,
            flux_anchors,
            flux_anchor_denoised,
            grid,
            spectrei,
            continuum_to_produce=[interp_choice, denoise_choice],
        )

        if denoise_choice == "denoised":
            continuum = c1d if interp_choice == "linear" else c3d
        elif denoise_choice == "undenoised":
            continuum = c1 if interp_choice == "linear" else c3
        else:
            continuum = c1d if np.any(c1d) else c1

        continuum = np.asarray(continuum, dtype=float)
        if not np.all(np.isfinite(continuum)) or np.all(continuum == 0):
            return _fallback_normalize(spectrei)[inv_order]
        continuum = np.where(continuum <= 0, np.nanmedian(continuum[continuum > 0]), continuum)

        normalized = spectrei / continuum
        normalized = np.clip(normalized, 0.05, 5.0)
        return normalized[inv_order]

    except Exception:
        return _fallback_normalize(flux)


def normalize_batch_with_rassine(
    wave: np.ndarray,
    fluxes: np.ndarray,
    config: Optional[Dict[str, object]] = None,
) -> np.ndarray:
    """Normalize a batch of spectra with RASSINE, with graceful fallbacks.

    Parameters
    ----------
    wave : np.ndarray
        1D wavelength grid shared by all spectra.
    fluxes : np.ndarray
        2D array of shape (N, P) with N spectra of length P.
    config : dict | None
        Optional configuration, forwarded to normalize_with_rassine.

    Returns
    -------
    np.ndarray
        Array of shape (N, P) with normalized spectra.
    """
    if not isinstance(fluxes, np.ndarray) or fluxes.ndim != 2:
        raise ValueError("fluxes must be a 2D numpy array of shape (N, P)")
    N, P = fluxes.shape
    if wave.ndim != 1 or wave.size != P:
        raise ValueError("wave must be 1D and have the same length as fluxes' second dimension")

    out = np.empty_like(fluxes, dtype=float)

    try:
        from concurrent.futures import ThreadPoolExecutor

        workers = min(8, max(1, (os.cpu_count() or 2)))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(
                ex.map(lambda row: normalize_with_rassine(wave, row, config=config), fluxes)
            )
        for i, r in enumerate(results):
            out[i] = r
    except Exception:
        for i in range(N):
            out[i] = normalize_with_rassine(wave, fluxes[i], config=config)

    return out


__all__ = [
    "normalize_with_rassine",
    "normalize_batch_with_rassine",
]
