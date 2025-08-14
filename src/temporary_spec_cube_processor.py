import h5py
import numpy as np
from src.rassine import normalize_batch_with_rassine
import json
import os

# ----------------------- PARAMS -----------------------
input_path = "data/soap_gpu_paper/spec_cube_tot.h5"
output_path = "data/soap_gpu_paper/spec_cube_tot_filtered_normalized_float64.h5"
indices_to_remove = [246, 249, 1196, 1453, 2176]
wave = np.load("data/soap_gpu_paper/spec_master.npz")["wavelength"]
indices_to_remove = np.asarray(indices_to_remove, dtype=int)
chunk_size = 100


rassine_config = {
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

# -------------------- MAIN --------------------
with h5py.File(input_path, "r") as f_in:
    dset_in = f_in["spec_cube"]
    n_specs, n_pixels = dset_in.shape
    if wave.shape[0] != n_pixels:
        raise ValueError(
            f"Incohérence: len(wave)={wave.shape[0]} != n_pixels={n_pixels}"
        )

    # Indices à garder
    mask_keep = np.ones(n_specs, dtype=bool)
    valid = (indices_to_remove >= 0) & (indices_to_remove < n_specs)
    mask_keep[indices_to_remove[valid]] = False
    indices_keep = np.nonzero(mask_keep)[0]
    n_keep = len(indices_keep)

    print(
        f"Total spectres: {n_specs} | conservés: {n_keep} | retirés: {np.sum(~mask_keep)}"
    )

    # --- Création ou reprise ---
    mode = "a" if os.path.exists(output_path) else "w"
    with h5py.File(output_path, mode) as f_out:
        # Créer datasets s'ils n'existent pas
        if "spec_cube" not in f_out:
            dset_out = f_out.create_dataset(
                "spec_cube",
                shape=(n_keep, n_pixels),
                dtype=np.float64,
                chunks=(min(chunk_size, n_keep), n_pixels),
                compression="gzip",
                shuffle=True,
                fillvalue=np.nan,  # important pour reprise
            )
            f_out.create_dataset("wave", data=wave)
            f_out.create_dataset("indices_keep", data=indices_keep, dtype=np.int64)
            # Attributs
            dset_out.attrs["normalized"] = "rassine-lite"
            dset_out.attrs["source"] = os.path.basename(input_path)
            dset_out.attrs["removed_indices"] = np.array(indices_to_remove, dtype=int)
            dset_out.attrs["rassine_config_json"] = json.dumps(rassine_config)
            f_out.flush()
        else:
            dset_out = f_out["spec_cube"]
            # Vérifs de cohérence
            if dset_out.shape != (n_keep, n_pixels):
                raise ValueError(
                    f"Sortie existante incompatible: {dset_out.shape} vs attendu {(n_keep, n_pixels)}"
                )
            if "indices_keep" in f_out:
                old_keep = f_out["indices_keep"][:]
                if not np.array_equal(old_keep, indices_keep):
                    raise ValueError(
                        "L'ordre/ensemble d'indices_keep a changé. Abandon pour éviter incohérences."
                    )

        # Reprise: trouver la première ligne encore en NaN
        # resume_at = _first_nan_row(dset_out)
        # if resume_at >= n_keep:
        #     print(f"✅ Déjà terminé : {output_path}")
        # else:
        #     print(f"↻ Reprise au rang {resume_at}/{n_keep}")
        resume_at = 0

        # Boucle de traitement à partir de resume_at
        for start_out in range(resume_at, n_keep, chunk_size):
            end_out = min(start_out + chunk_size, n_keep)
            # Indices sources correspondants
            block_idx_in = indices_keep[start_out:end_out]
            # Lecture en bloc des flux
            block_flux = dset_in[block_idx_in, :]  # (B, n_pixels)

            Y = normalize_batch_with_rassine(wave, block_flux, config=rassine_config)
            dset_out[start_out:end_out, :] = Y

            f_out.flush()  # flush à chaque bloc pour limiter la perte en cas de crash

            done = end_out
            print(f"[{done}/{n_keep}] {(done / n_keep) * 100:.1f}%")

print(f"✅ Terminé : {output_path}")
