import os
import pickle
import re  # <-- NEW: local import pour le slug
import sys
import tempfile
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.interpolate import augment_spectra_uniform, shift_spectra_linear
from src.utils import get_free_memory

##############################################################################
##############################################################################
#                           *Dataset et gestion des données*                 #
##############################################################################
##############################################################################


def check_system_resources():
    """
    Vérifie les ressources système disponibles et retourne des recommandations.

    Returns:
        dict: Dictionnaire avec les informations système et recommandations
    """
    import psutil

    # Mémoire système
    memory = psutil.virtual_memory()
    available_memory_gb = memory.available / (1024**3)
    total_memory_gb = memory.total / (1024**3)

    # Mémoire GPU
    gpu_available = torch.cuda.is_available()
    gpu_memory_gb = 0
    if gpu_available:
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    # CPU
    cpu_count = psutil.cpu_count()

    # Recommandations
    recommendations = {
        "batch_size": 100,
        "max_gpu_memory_gb": 4,
        "force_cpu": False,
        "use_chunked_loading": False,
    }

    # Ajustements basés sur les ressources
    if available_memory_gb < 4:
        recommendations["batch_size"] = 25
        recommendations["use_chunked_loading"] = True
        print("⚠️  Mémoire système faible, réduction des paramètres recommandée")
    elif available_memory_gb < 8:
        recommendations["batch_size"] = 50
        print("ℹ️  Mémoire système modérée, paramètres conservateurs recommandés")

    if not gpu_available:
        recommendations["force_cpu"] = True
        print("💻 GPU non disponible, utilisation CPU uniquement")
    elif gpu_memory_gb < 2:
        recommendations["max_gpu_memory_gb"] = 1
        recommendations["batch_size"] = min(recommendations["batch_size"], 25)
        print("⚠️  Mémoire GPU faible, limitation des opérations GPU")

    return {
        "system": {
            "total_memory_gb": total_memory_gb,
            "available_memory_gb": available_memory_gb,
            "gpu_available": gpu_available,
            "gpu_memory_gb": gpu_memory_gb,
            "cpu_count": cpu_count,
        },
        "recommendations": recommendations,
    }


# * -- Classe principale standardisée: charge UNIQUEMENT un NPZ depuis data/npz_datasets --
def _to_tensor(x, dtype):
    return torch.tensor(x).to(dtype=dtype).contiguous() if x is not None else None

class SpectrumDataset(Dataset):
    """
    Dataset pour charger des spectres depuis un .npz 'nouveau format'.

    Clés attendues:
      - wavegrid (P,), template (P,)
      - spectra_train (Ntr,P), spectra_val (Nv,P)
      - time_values_train (Ntr,), time_values_val (Nv,)
      - Optionnel: activity_train/val, spectra_no_activity_train/val, v_true_train/val
      - metadata (dict) incluant: n_spectra_train, n_spectra_val, n_pixels, wavemin, wavemax,
        planets_periods, planets_amplitudes, planets_phases (pour fallback v_true)
    """

    def __init__(
        self,
        dataset_filepath: str,
        split: str = "train",          # "train" | "val" | "all"
        data_dtype: torch.dtype = torch.float32,
        cuda: bool = True,
    ):
        if not dataset_filepath.endswith(".npz"):
            raise ValueError("dataset_filepath doit pointer vers un fichier .npz")
        if not os.path.exists(dataset_filepath):
            raise FileNotFoundError(f"NPZ dataset not found: {dataset_filepath}")

        self.dataset_filepath = dataset_filepath
        self.split = split
        self.data_dtype = data_dtype

        self._init_from_npz(dataset_filepath, data_dtype)

        if cuda and torch.cuda.is_available():
            self.move_to_cuda()

    # --------- lecture / assemblage ----------
    def _init_from_npz(self, npz_path: str, data_dtype: torch.dtype):
        ds = np.load(npz_path, allow_pickle=True)

        # Fixes
        if "metadata" not in ds.files:
            raise KeyError("Clé 'metadata' manquante dans le npz.")
        self.metadata = dict(ds["metadata"].item())

        # Invariants (communs)
        if "wavegrid" not in ds.files or "template" not in ds.files:
            raise KeyError("Clés 'wavegrid' et 'template' requises.")
        wavegrid_np = ds["wavegrid"]
        template_np = ds["template"]

        # --- Sélection split ---
        def pick(name):
            # name sans suffixe, ex: "spectra", "time_values", "activity", "spectra_no_activity", "v_true"
            if self.split == "train":
                key = f"{name}_train"
                return ds[key] if key in ds.files else None
            elif self.split == "val":
                key = f"{name}_val"
                return ds[key] if key in ds.files else None
            elif self.split == "all":
                ktr, kval = f"{name}_train", f"{name}_val"
                a = ds[ktr] if ktr in ds.files else None
                b = ds[kval] if kval in ds.files else None
                if a is None and b is None:
                    return None
                if a is None:  return b
                if b is None:  return a
                # Concat sur l'axe 0 si 2D, sinon 1D
                axis = 0 if a.ndim > 1 else 0
                return np.concatenate([a, b], axis=axis)
            else:
                raise ValueError("split doit être 'train', 'val' ou 'all'.")

        spectra_np = pick("spectra")
        if spectra_np is None:
            raise KeyError(f"Clés '{'spectra_train/val'}' manquantes pour split={self.split}.")

        time_values_np = pick("time_values")
        if time_values_np is None:
            raise KeyError(f"Clés '{'time_values_train/val'}' manquantes pour split={self.split}.")

        activity_np = pick("activity")
        spectra_no_activity_np = pick("spectra_no_activity")
        v_true_np = pick("v_true")

        # Fallback v_true si absent -> sinus de metadata
        if v_true_np is None:
            P = self.metadata.get("planets_periods", []) or []
            K = self.metadata.get("planets_amplitudes", []) or []
            PHI = self.metadata.get("planets_phases", []) or []
            if len(P) == len(K) == len(PHI) and len(P) > 0:
                t = time_values_np.astype(np.float64)
                v = np.zeros_like(t, dtype=np.float64)
                for k, p, phi in zip(K, P, PHI):
                    v += k * np.sin(2 * np.pi * t / p + phi)
                v_true_np = v.astype(np.float32)
            else:
                # sinon, vecteur nul
                v_true_np = np.zeros_like(time_values_np, dtype=np.float32)

        # Conversion -> torch
        self.spectra = _to_tensor(spectra_np, data_dtype)
        self.wavegrid = _to_tensor(wavegrid_np, data_dtype)
        self.template = _to_tensor(template_np, data_dtype)
        self.planet_periods = self.metadata.get("planets_periods", [])
        self.planet_amplitudes = self.metadata.get("planets_amplitudes", [])
        self.planet_phases = self.metadata.get("planets_phases", [])
        self.time_values = _to_tensor(time_values_np, data_dtype)
        self.activity = _to_tensor(activity_np, data_dtype)
        self.spectra_no_activity = _to_tensor(spectra_no_activity_np, data_dtype)
        self.v_true = _to_tensor(v_true_np, data_dtype)

        # Tailles / bornes
        self.n_spectra = self.spectra.shape[0]
        self.n_pixels = self.spectra.shape[1]
        self.wavemin = float(self.metadata.get("wavemin", float(self.wavegrid.min().item())))
        self.wavemax = float(self.metadata.get("wavemax", float(self.wavegrid.max().item())))

        # Sanity checks rapides
        assert self.time_values.shape[0] == self.n_spectra, "time_values et spectra désalignés"
        assert self.v_true.shape[0] == self.n_spectra, "v_true et spectra désalignés"
        if self.activity is not None:
            assert self.activity.shape == self.spectra.shape, "activity et spectra doivent avoir la même forme"
        if self.spectra_no_activity is not None:
            assert self.spectra_no_activity.shape == self.spectra.shape, "spectra_no_activity et spectra doivent avoir la même forme"

    # --------- API Dataset ----------
    def __len__(self):
        return self.n_spectra

    def __getitem__(self, idx):
        # On conserve ton comportement minimal (retourne le spectre).
        # Si tu veux plus d’info, tu peux changer ici pour retourner un dict.
        return self.spectra[idx]

    # --------- utilitaires ----------
    def _estimate_memory_usage(self):
        def mb(t):
            return 0 if t is None else t.numel() * t.element_size() / (1024 * 1024)
        return mb(self.spectra) + mb(self.wavegrid) + mb(self.template) + mb(self.time_values)

    def move_to_cuda(self):
        if torch.cuda.is_available():
            for name in ["spectra","wavegrid","template","time_values","activity","spectra_no_activity","v_true"]:
                t = getattr(self, name, None)
                if t is not None:
                    setattr(self, name, t.cuda())

    def convert_dtype(self, new_dtype: torch.dtype):
        def cast(name):
            t = getattr(self, name, None)
            if t is not None:
                setattr(self, name, t.to(dtype=new_dtype))
        old = self._estimate_memory_usage()
        for k in ["spectra","wavegrid","template","time_values","activity","spectra_no_activity","v_true"]:
            cast(k)
        self.data_dtype = new_dtype
        new = self._estimate_memory_usage()
        print(f"Conversion dtype: {old:.2f} -> {new:.2f} MB (-{(old-new)/old*100:.1f}%)")
        return self

    def __repr__(self):
        def shape_dtype(t):
            return f"{tuple(t.shape)} | {t.dtype}" if t is not None else "None"
        return (
            f"\n======== SpectrumDataset ({self.split}) ========\n"
            f"n_spectra={self.n_spectra}, n_pixels={self.n_pixels}\n"
            f"spectra={shape_dtype(self.spectra)}\n"
            f"spectra_no_activity={shape_dtype(self.spectra_no_activity)}\n"
            f"activity={shape_dtype(self.activity)}\n"
            f"wavegrid={shape_dtype(self.wavegrid)}\n"
            f"template={shape_dtype(self.template)}\n"
            f"time_values={shape_dtype(self.time_values)}\n"
            f"v_true={shape_dtype(self.v_true)}\n"
            f"[{self.wavemin:.3f}, {self.wavemax:.3f}]  dtype={self.data_dtype}\n"
            f"Memory ~{self._estimate_memory_usage():.2f} MB\n"
            f"===============================================\n"
        )

    def to_dict(self):
        return {
            "dataset_filepath": self.dataset_filepath,
            "split": self.split,
            "data_dtype": self.data_dtype,
            "cuda": self.spectra.is_cuda,
        }

# * -- Fonction de collate pour le DataLoader (simplifie la vie) --
def generate_collate_fn(
    dataset,
    M=1,
    vmin=-3,
    vmax=3,
    interpolate="linear",
    extrapolate="linear",
    out_dtype=torch.float32,
):
    """
    Génère une fonction de collate pour le DataLoader.
    Cette fonction récupère directement les spectres du dataset et les augmente en utilisant
    la fonction augment_spectra_uniform.

    Args:
        dataset (SpectrumDataset): Le dataset à utiliser.
        M (int): Le nombre de spectres y_aug_j générés pour chaque spectre observé (par défaut 1).
        vmin (float): La vitesse minimale pour l'augmentation des spectres.
        vmax (float): La vitesse maximale pour l'augmentation des spectres.
        interpolate (str): Méthode d'interpolation à utiliser.
        extrapolate (str): Méthode d'extrapolation à utiliser.
        out_dtype (torch.dtype): Le type de données de sortie des spectres augmentés.
    """

    def collate_fn(batch):
        # batch : liste de y_obs (Tensor [n_pixel]) de taille B
        batch_yobs = torch.stack(batch, dim=0)  # Tensor [B, n_pixel]

        batch_yobs = (
            batch_yobs.unsqueeze(1).repeat(1, M, 1).view(-1, dataset.n_pixels)
        )  # Tensor [M*B, n_pixel]

        batch_wavegrid = (
            dataset.wavegrid.unsqueeze(0)
            .repeat(batch_yobs.shape[0], 1)
            .to(batch_yobs.device)
            .contiguous()
        )  # Tensor [M*B, n_pixel]

        # Tensor [M*B, n_pixel]
        batch_yaug, batch_voffset = augment_spectra_uniform(
            batch_yobs,
            batch_wavegrid,
            vmin=vmin,
            vmax=vmax,
            interpolate=interpolate,
            extrapolate=extrapolate,
            out_dtype=out_dtype,
        )

        return (batch_yobs, batch_yaug, batch_voffset, batch_wavegrid)

    return collate_fn


if __name__ == "__main__":
    spec_dset = SpectrumDataset(
        dataset_filepath="/home/tliopis/Codes/exoplanets_llopis_mary_2025/data/npz_datasets/soapgpu_nst120_nsv120_5000-5050_dx2_sm3_p60_k0p1_phi0.npz"
    )
