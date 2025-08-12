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
class SpectrumDataset(Dataset):
    """
    Dataset pour charger des spectres depuis un fichier .npz standardisé.

    Le fichier doit contenir au minimum les clés suivantes:
      - spectra (np.ndarray [N, P])
      - wavegrid (np.ndarray [P])
      - time_values (np.ndarray [N])
      - metadata (dict) avec: n_spectra, n_pixels, wavemin, wavemax, data_dtype,
        planets_periods, planets_amplitudes, planets_phases
    """

    def __init__(
        self,
        dataset_filepath: str = "",
        data_dtype: torch.dtype = torch.float32,
        cuda: bool = True,
    ):
        print("Initialisation du SpectrumDataset...")

        if not dataset_filepath or not dataset_filepath.endswith(".npz"):
            raise ValueError(
                "dataset_filepath doit être un chemin complet vers un fichier .npz"
            )
        if not os.path.exists(dataset_filepath):
            raise FileNotFoundError(f"NPZ dataset not found: {dataset_filepath}")

        # Stocker le chemin du fichier
        self.dataset_filepath = dataset_filepath

        # Chargement
        self._init_from_npz(self.dataset_filepath, data_dtype)

        print(self)
        print("Déplacement des données vers le GPU si disponible...")

        if cuda and torch.cuda.is_available():
            print("CUDA est activé, les données seront déplacées vers le GPU.")
            print(f"CUDA disponible: {get_free_memory() / 1e9:.3f} GB")
            self.move_to_cuda()
            print(f"CUDA disponible: {get_free_memory() / 1e9:.3f} GB")

        print("Dataset initialisé avec succès.")

    def _init_from_npz(self, npz_path: str, data_dtype: torch.dtype):
        """
        Initialise le dataset depuis un fichier NPZ standardisé.
        Attend les clés: spectra, wavegrid, time_values et metadata.
        """
        ds = np.load(npz_path, allow_pickle=True)

        # Validation des clés requises
        required_keys = {"spectra", "wavegrid", "time_values", "metadata"}
        missing = required_keys - set(ds.files)
        if missing:
            raise KeyError(
                f"Clés manquantes dans {npz_path}: {sorted(missing)} (requis: {sorted(required_keys)})"
            )

        # Données principales
        spectra_np = ds["spectra"]
        spectra_no_activity_np = (
            ds["spectra_no_activity"] if "spectra_no_activity" in ds else None
        )
        wavegrid_np = ds["wavegrid"]
        time_values_np = ds["time_values"]
        template_np = ds["template"] if "template" in ds else None
        activity_np = ds["activity"] if "activity" in ds else None

        # Conversion vers torch
        self.spectra = torch.tensor(spectra_np).to(dtype=data_dtype).contiguous()
        self.spectra_no_activity = (
            torch.tensor(spectra_no_activity_np).to(dtype=data_dtype).contiguous()
            if spectra_no_activity_np is not None
            else None
        )
        self.wavegrid = torch.tensor(wavegrid_np).to(dtype=data_dtype).contiguous()
        self.template = (
            torch.tensor(template_np).to(dtype=data_dtype).contiguous()
            if template_np is not None
            else self.spectra.mean(dim=0)
        )
        self.time_values = (
            torch.tensor(time_values_np).to(dtype=data_dtype).contiguous()
        )
        self.activity = (
            torch.tensor(activity_np).to(dtype=data_dtype).contiguous()
            if activity_np is not None
            else None
        )
        # Métadonnées unifiées
        metadata = ds["metadata"].item()
        self.metadata = metadata
        self.n_spectra = int(metadata.get("n_spectra", self.spectra.shape[0]))
        self.n_pixels = int(metadata.get("n_pixels", self.spectra.shape[1]))
        self.data_dtype = data_dtype
        self.wavemin = float(metadata.get("wavemin", float(self.wavegrid.min())))
        self.wavemax = float(metadata.get("wavemax", float(self.wavegrid.max())))
        # Quelques alias utiles
        self.planets_periods = metadata.get("planets_periods", None)
        self.planets_amplitudes = metadata.get("planets_amplitudes", None)
        self.planets_phases = metadata.get("planets_phases", None)

        self.v_true = torch.zeros_like(self.time_values)
        for Kp, P, Phi in zip(
            self.planets_amplitudes, self.planets_periods, self.planets_phases
        ):
            v = Kp * torch.sin(2 * np.pi * self.time_values / P + Phi)
            self.v_true += v

        # Normaliser cohérence
        assert self.n_spectra == self.spectra.shape[0]
        assert self.n_pixels == self.spectra.shape[1]

    def __len__(self):
        return self.spectra.shape[0]

    def __getitem__(self, idx):
        """
        Retourne un spectre, sa grille de longueurs d'onde, le template stellaire et la date julienne.
        """
        return self.spectra[idx]

    def __repr__(self):
        dtype_info = ""
        if self.data_dtype == torch.float16:
            dtype_info = " (HALF PRECISION - économie mémoire ~50%)"
        elif self.data_dtype == torch.float32:
            dtype_info = " (SINGLE PRECISION - standard)"
        elif self.data_dtype == torch.float64:
            dtype_info = " (DOUBLE PRÉCISION - haute précision)"

        return (
            f"\n======== SpectrumDataset ========\n"
            f"n_spectra={self.spectra.shape[0]}, n_pixels={self.spectra.shape[1]}\n"
            f"spectra_shape={self.spectra.shape} | {self.spectra.dtype}{dtype_info}\n"
            f"wavegrid_shape={self.wavegrid.shape} | {self.wavegrid.dtype}\n"
            f"template_shape={self.template.shape} | {self.template.dtype}\n"
            f"time_values_shape={self.time_values.shape} | {self.time_values.dtype})\n"
            f"Memory footprint: ~{self._estimate_memory_usage():.2f} MB\n"
            f"======== End of SpectrumDataset ========\n"
        )

    def _estimate_memory_usage(self):
        """
        Estime l'utilisation mémoire du dataset en MB.
        """

        def tensor_memory_mb(tensor):
            return tensor.numel() * tensor.element_size() / (1024 * 1024)

        total_memory = 0
        total_memory += tensor_memory_mb(self.spectra)
        total_memory += tensor_memory_mb(self.wavegrid)
        total_memory += tensor_memory_mb(self.template)
        total_memory += tensor_memory_mb(self.time_values)

        return total_memory

    def move_to_cuda(self):
        """
        Déplace les données du dataset vers le GPU si disponible.
        """
        if torch.cuda.is_available():
            self.spectra = self.spectra.cuda()
            self.wavegrid = self.wavegrid.cuda()
            self.template = self.template.cuda()
            self.v_true = self.v_true.cuda()
        else:
            print("CUDA n'est pas disponible, les données restent sur le CPU.")

    def convert_dtype(self, new_dtype):
        """
        Convertit le dataset vers un nouveau type de données.

        Args:
            new_dtype (torch.dtype): Nouveau type de données (ex: torch.float16, torch.float32)

        Returns:
            SpectrumDataset: Nouveau dataset avec le type de données converti
        """
        print(f"Conversion du dataset de {self.data_dtype} vers {new_dtype}...")

        old_memory = self._estimate_memory_usage()

        # Conversion des tenseurs
        self.spectra = self.spectra.to(dtype=new_dtype)
        self.wavegrid = self.wavegrid.to(dtype=new_dtype)
        self.template = self.template.to(dtype=new_dtype)
        self.time_values = self.time_values.to(dtype=new_dtype)
        self.data_dtype = new_dtype

        new_memory = self._estimate_memory_usage()
        memory_savings = ((old_memory - new_memory) / old_memory) * 100

        print("Conversion terminée:")
        print(f"  Mémoire avant: {old_memory:.2f} MB")
        print(f"  Mémoire après: {new_memory:.2f} MB")
        print(f"  Économie: {memory_savings:.1f}%")

        return self

    def to_dict(self):
        """
        Retourne un dict minimal pour recharger ce dataset.
        """
        return {
            "dataset_filepath": self.dataset_filepath,
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
    pass
