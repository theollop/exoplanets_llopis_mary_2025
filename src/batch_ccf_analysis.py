import h5py
import numpy as np
import os
import gc
from src.ccf import get_full_ccf_analysis
from src.dataset import _normalize_spectrum_with_rassine
from src.utils import clear_gpu_memory


def process_spectra_batch_ccf(
    spectra_filepath="data/soap_gpu_paper/spec_cube_tot.h5",
    spec_filepath="data/soap_gpu_paper/spec_master.npz",
    output_dir="data/ccf_results",
    batch_size=100,
    v_grid_range=(-20000, 20000),
    v_grid_points=200,
    window_size_velocity=820.0,
    mask_type="G2",
    use_rassine=True,
    wavemin=5000,
    wavemax=5050,
    specs_to_remove=[246, 249, 1196, 1453, 2176],
):
    # Chargement du template et wavegrid
    spec_master = np.load(spec_filepath)
    wavegrid_full = spec_master["wavelength"]

    # Masque de longueurs d'onde
    wave_mask = (wavegrid_full >= wavemin) & (wavegrid_full <= wavemax)
    wavegrid = wavegrid_full[wave_mask]

    # Grille de vitesses
    v_grid = np.linspace(v_grid_range[0], v_grid_range[1], v_grid_points)

    # Configuration Rassine basique
    rassine_config = {"column_wave": "wave", "column_flux": "flux"}

    os.makedirs(output_dir, exist_ok=True)

    # Listes pour stocker tous les résultats
    all_rvs = []
    all_depths = []
    all_fwhms = []
    all_spans = []
    all_continuum = []
    all_amplitude = []
    all_ccfs = []

    with h5py.File(spectra_filepath, "r") as f:
        dset = f["spec_cube"]
        n_total_spectra = dset.shape[0]
        n_pixels = wave_mask.sum()

        # Créer un masque pour les spectres à conserver
        valid_indices = np.ones(n_total_spectra, dtype=bool)
        if specs_to_remove:
            specs_to_remove = np.array(specs_to_remove)
            specs_to_remove = specs_to_remove[specs_to_remove < n_total_spectra]
            valid_indices[specs_to_remove] = False

        # Obtenir les indices des spectres valides
        valid_spec_indices = np.where(valid_indices)[0]
        n_valid_spectra = len(valid_spec_indices)

        print(f"Traitement de {n_valid_spectra} spectres par lots de {batch_size}")
        if specs_to_remove is not None and len(specs_to_remove) > 0:
            print(f"Spectres exclus: {specs_to_remove}")
        print(f"Domaine spectral: {wavemin}-{wavemax} Å ({n_pixels} pixels)")

        for batch_start in range(0, n_valid_spectra, batch_size):
            batch_end = min(batch_start + batch_size, n_valid_spectra)
            current_batch_size = batch_end - batch_start

            print(
                f"Lot {batch_start // batch_size + 1}: spectres {batch_start}-{batch_end - 1}"
            )

            # Indices des spectres pour ce batch
            batch_indices = valid_spec_indices[batch_start:batch_end]

            # Chargement du lot avec crop spectral
            batch_spectra = dset[batch_indices][:, wave_mask]

            # Normalisation avec Rassine si demandée
            if use_rassine:
                normalized_spectra = np.zeros_like(batch_spectra)
                for i in range(current_batch_size):
                    normalized_spectra[i] = _normalize_spectrum_with_rassine(
                        wavegrid, batch_spectra[i], rassine_config
                    )
            else:
                # Normalisation simple par la médiane
                normalized_spectra = batch_spectra / np.median(
                    batch_spectra, axis=1, keepdims=True
                )

            # Calcul des CCFs et analyse
            ccf_results, raw_ccfs = get_full_ccf_analysis(
                spectra=normalized_spectra,
                wavegrid=wavegrid,
                v_grid=v_grid,
                window_size_velocity=window_size_velocity,
                mask_type=mask_type,
                verbose=False,
                normalize=True,
                return_raw_ccfs=True,
            )

            # Stockage des résultats
            all_rvs.extend(ccf_results["rv"])
            all_depths.extend(ccf_results["depth"])
            all_fwhms.extend(ccf_results["fwhm"])
            all_spans.extend(ccf_results["span"])
            all_continuum.extend(ccf_results["continuum"])
            all_amplitude.extend(ccf_results["amplitude"])
            all_ccfs.append(raw_ccfs)

            # Nettoyage mémoire
            del batch_spectra, normalized_spectra, raw_ccfs
            clear_gpu_memory()
            gc.collect()

    # Conversion en arrays numpy
    results = {
        "rvs": np.array(all_rvs),
        "depths": np.array(all_depths),
        "fwhms": np.array(all_fwhms),
        "spans": np.array(all_spans),
        "continuum": np.array(all_continuum),
        "amplitude": np.array(all_amplitude),
        "raw_ccfs": np.vstack(all_ccfs),
        "v_grid": v_grid,
        "wavegrid": wavegrid,
        "metadata": {
            "n_spectra": n_valid_spectra,
            "n_pixels": n_pixels,
            "wavemin": wavemin,
            "wavemax": wavemax,
            "batch_size": batch_size,
            "window_size_velocity": window_size_velocity,
            "mask_type": mask_type,
            "use_rassine": use_rassine,
            "specs_to_remove": specs_to_remove,
        },
    }

    # Sauvegarde
    output_filepath = os.path.join(output_dir, "ccf_analysis_results.npz")
    np.savez_compressed(output_filepath, **results)

    print(f"Résultats sauvegardés dans {output_filepath}")
    print(f"- {n_valid_spectra} spectres analysés")
    print(f"- RV moyenne: {np.nanmean(all_rvs):.2f} m/s")
    print(f"- Profondeur moyenne: {np.nanmean(all_depths):.4f}")

    # Nettoyage final
    clear_gpu_memory()
    gc.collect()

    return results


if __name__ == "__main__":
    process_spectra_batch_ccf()
