import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from src.utils import get_free_memory
from src.interpolate import augment_spectra_uniform
import h5py
import os
import sys
import tempfile
import pickle
from src.interpolate import shift_spectra_linear

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
        wavegrid_np = ds["wavegrid"]
        time_values_np = ds["time_values"]
        template_np = ds["template"] if "template" in ds else None

        # Conversion vers torch
        self.spectra = torch.tensor(spectra_np).to(dtype=data_dtype).contiguous()
        self.wavegrid = torch.tensor(wavegrid_np).to(dtype=data_dtype).contiguous()
        self.template = (
            torch.tensor(template_np).to(dtype=data_dtype).contiguous()
            if template_np is not None
            else self.spectra.mean(dim=0)
        )
        self.time_values = (
            torch.tensor(time_values_np).to(dtype=data_dtype).contiguous()
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

    def init_rvdatachallenge_dataset(
        self,
        star_name="STAR1136",
        data_root_dir="data",
        n_spectra=None,
        wavemin=None,
        wavemax=None,
        data_dtype=torch.float32,
    ):
        """
        Retourne le dataset du RV Data Challenge pour l'étoile spécifiée.
        """
        if star_name == "STAR1136":
            dataset_dirpath = f"{data_root_dir}/rv_datachallenge/Sun_B57001_E61001_planet-FallChallenge1"
        elif star_name == "STAR1138":
            dataset_dirpath = f"{data_root_dir}/rv_datachallenge/Sun_B57002_E61002_planet-FallChallenge2"
        elif star_name == "STAR1134":
            dataset_dirpath = f"{data_root_dir}/rv_datachallenge/Sun_B57000_E61000_planet-FallChallenge3"
        else:
            raise ValueError(
                f"Nom d'étoile inconnu: {star_name}. Utilisez 'STAR1136', 'STAR1138' ou 'STAR1134'."
            )

        dataset_filepath = f"{dataset_dirpath}/HARPN/{star_name}_HPN_flux_YVA.npy"

        analyse_material = np.load(
            f"{data_root_dir}/rv_datachallenge/Sun_B57001_E61001_planet-FallChallenge1/HARPN/{star_name}_HPN_Analyse_material.p",
            allow_pickle=True,
        )

        wavegrid = analyse_material["wave"].to_numpy()
        template = analyse_material["stellar_template"].to_numpy()

        if wavemin is None:
            wavemin = wavegrid.min()
        if wavemax is None:
            wavemax = wavegrid.max()

        wave_mask = (wavegrid >= wavemin) & (wavegrid <= wavemax)
        wavegrid = wavegrid[wave_mask]
        template = template[wave_mask]

        analyse_summary = pd.read_csv(
            f"{data_root_dir}/rv_datachallenge/Sun_B57001_E61001_planet-FallChallenge1/HARPN/{star_name}_HPN_Analyse_summary.csv"
        )

        if n_spectra is None:
            n_spectra = analyse_summary.shape[0]
        data = np.load(dataset_filepath)
        data = data[:n_spectra, wave_mask]

        self.spectra = torch.tensor(data).to(dtype=data_dtype).contiguous()
        self.wavegrid = torch.tensor(wavegrid).to(dtype=data_dtype).contiguous()
        self.template = torch.tensor(template).to(dtype=data_dtype).contiguous()
        self.time_values = (
            torch.tensor(analyse_summary["jdb"][:n_spectra])
            .to(dtype=data_dtype)
            .contiguous()
        )
        self.n_spectra = n_spectra
        self.n_pixels = wavegrid.shape[0]
        self.data_dtype = data_dtype
        self.wavemin = wavemin
        self.wavemax = wavemax

        transit_information = pd.read_csv(
            f"{dataset_dirpath}/{star_name}_HPN_Transit_information.csv"
        )
        self.known_periods = transit_information["p"].to_numpy()

    def init_soap_gpu_paper_dataset(
        self,
        dataset_filename="dataset_1000specs_5000_5050_Kp1e-1_P100.npz",
        data_dtype=torch.float32,
    ):
        """
        DEPRECATED: Les chargements doivent passer par le NPZ standardisé via __init__.
        """
        raise NotImplementedError(
            "Utiliser SpectrumDataset(data_root_dir, dataset_filename) pour charger le NPZ."
        )


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


def inject_dataset(
    spectra: torch.Tensor,
    wavegrid: torch.Tensor,
    time_values: np.ndarray,
    planets_amplitudes: list[float],
    planets_periods: list[float],
    planets_phases: list[float],
    batch_size=None,
):
    """
    Injects artificial planetary signals into the dataset by shifting spectra according to RV variations.

    Args:
        spectra: np.ndarray containing the spectra
        planets_amplitudes: List of semi-planets_amplitudes (Kp) in m/s for each planet
        planets_periods: List of planets_periods (P) in days for each planet
        batch_size: If None, process all spectra at once. Otherwise, process in batches.

    Returns:
        torch.Tensor: Modified spectra with injected planetary signals
    """
    # Calculate RV velocities for all time points
    velocities = np.zeros(len(time_values))
    time_values = time_values

    for Kp, P, phase in zip(planets_amplitudes, planets_periods, planets_phases):
        velocities += Kp * np.sin(2 * np.pi * time_values / P + phase)

    # Convert to tensor and ensure same device as dataset
    velocities = torch.tensor(velocities, dtype=spectra.dtype, device=spectra.device)

    if batch_size is None:
        # Process all spectra at once
        print(f"Processing all {len(spectra)} spectra at once...")
        injected_spectra = shift_spectra_linear(
            spectra=spectra,
            wavegrid=wavegrid.unsqueeze(0).expand(len(spectra), -1).contiguous(),
            velocities=velocities,
        )
    else:
        # Process in batches
        print(f"Processing {len(spectra)} spectra in batches of {batch_size}...")
        injected_spectra_list = []

        for i in range(0, len(spectra), batch_size):
            end_idx = min(i + batch_size, len(spectra))
            batch_spectra = spectra[i:end_idx]
            batch_velocities = velocities[i:end_idx]
            batch_wavegrid = wavegrid.unsqueeze(0).expand(end_idx - i, -1).contiguous()

            batch_injected = shift_spectra_linear(
                spectra=batch_spectra,
                wavegrid=batch_wavegrid,
                velocities=batch_velocities,
            )

            injected_spectra_list.append(batch_injected)

        injected_spectra = torch.cat(injected_spectra_list, dim=0)

    return injected_spectra


def create_soap_gpu_paper_dataset(
    spectra_filepath,
    spec_filepath,
    output_filepath,
    n_spectra,
    wavemin,
    wavemax,
    downscaling_factor,
    use_rassine=True,
    rassine_config=None,
    add_photon_noise=False,
    snr_target=None,
    noise_seed=None,
    specs_to_remove=[246, 249, 1196, 1453, 2176],
    planets_amplitudes=None,
    planets_periods=None,
    planets_phases=None,
    batch_size=100,  # Nouveau paramètre pour traitement par batches
):
    """
    Charge le template et les spectres, masque par longueur d'onde,
    normalise avec Rassine (optionnel), downscale en moyennant,
    ajoute du bruit photonique réaliste (optionnel),
    et sauve le résultat dans un .npz.

    Parameters
    ----------
    spectra_filepath : str
        Chemin vers le fichier HDF5 contenant le dataset 'spec_cube' avec les spectres.
    spec_filepath : str
        Chemin vers le fichier .npz contenant 'spec' et 'wavelength'.
    output_filepath : str
        Chemin du .npz de sortie qui sera créé.
    n_spectra : int
        Nombre de spectres à extraire du dataset.
    wavemin, wavemax : float
        Bornes de longueur d'onde en Angstrom.
    downscaling_factor : int
        Facteur de binning (nombre de pixels à moyenner).
    use_rassine : bool, optional
        Si True, utilise Rassine pour normaliser les spectres (défaut: True).
    rassine_config : dict, optional
        Configuration personnalisée pour Rassine. Si None, utilise la config par défaut.
    add_photon_noise : bool, optional
        Si True, ajoute du bruit photonique réaliste aux spectres (défaut: False).
    snr_target : float, optional
        SNR cible pour le bruit photonique. Si None, utilise le niveau de signal existant.
    noise_seed : int, optional
        Graine pour la génération aléatoire du bruit (pour reproductibilité).
    batch_size : int, optional
        Taille des batches pour le traitement (défaut: 100).
    """

    print("🔄 Création du dataset SOAP GPU Paper...")

    # 1) Chargement
    spec_data = np.load(spec_filepath)
    template = spec_data["spec"]
    wavegrid = spec_data["wavelength"]

    if wavemin is None:
        wavemin = wavegrid.min()
    if wavemax is None:
        wavemax = wavegrid.max()

    # 2) Masquage par longueur d'onde
    mask = (wavegrid >= wavemin) & (wavegrid <= wavemax)
    template_masked = template[mask]
    wavegrid_masked = wavegrid[mask]

    # 3) Récupérer le nombre total de spectres dans le dataset
    with h5py.File(spectra_filepath, "r") as f:
        n_spectra_tot = f["spec_cube"].shape[0]

    if n_spectra is None:
        n_spectra = n_spectra_tot

    time_values = np.arange(n_spectra)

    # 3) Chargement des spectres par chunks pour éviter les problèmes de mémoire
    print("📊 Chargement des spectres de données...")

    # Estimer la taille de données à charger
    with h5py.File(spectra_filepath, "r") as f:
        spectra_masked = f["spec_cube"][:n_spectra, mask]

        if specs_to_remove:
            print(f"⚠️ Suppression des spectres {specs_to_remove} du template")
            specs_to_remove = np.array(specs_to_remove)
            specs_to_remove = specs_to_remove[specs_to_remove < n_spectra]
            spectra_masked = np.delete(spectra_masked, specs_to_remove, axis=0)
            time_values = np.delete(time_values, specs_to_remove)
            # Update n_spectra to reflect the actual number of spectra after removal
            n_spectra = spectra_masked.shape[0]

    print(f"Données chargées: {n_spectra} spectres, {wavegrid_masked.size} pixels")
    print(f"Gamme spectrale: {wavemin:.1f} - {wavemax:.1f} Å")

    # 4) Normalisation avec Rassine (optionnel)
    if use_rassine:
        print("\n🔄 Normalisation avec Rassine...")

        # Configuration par défaut conservative pour Rassine
        default_rassine_config = {
            "axes_stretching": "auto_0.3",  # Plus conservative (0.3 au lieu de 0.5)
            "vicinity_local_max": 5,  # Fenêtre plus petite pour préserver détails
            "smoothing_box": 3,  # Moins de lissage (3 au lieu de 6)
            "smoothing_kernel": "gaussian",
            "fwhm_ccf": "auto",
            "CCF_mask": "master",
            "RV_sys": 0,
            "mask_telluric": [[6275, 6330], [6470, 6577], [6866, 8000]],
            "mask_broadline": [[3960, 3980], [6560, 6562], [10034, 10064]],
            "min_radius": "auto",
            "max_radius": "auto",
            "model_penality_radius": "poly_0.5",  # Pénalité plus faible (0.5 au lieu de 1.0)
            "denoising_dist": 3,  # Distance plus petite (3 au lieu de 5)
            "number_of_cut": 2,  # Moins de coupes itératives (2 au lieu de 3)
            "number_of_cut_outliers": 1,
            "interpol": "linear",  # Interpolation linéaire plus conservative
            "feedback": False,
            "only_print_end": True,
            "plot_end": False,
            "save_last_plot": False,
            "outputs_interpolation_save": "linear",  # Sortie linéaire
            "outputs_denoising_save": "undenoised",
            "light_file": True,
            "speedup": 0.5,  # Plus lent mais plus précis (0.5 au lieu de 1.0)
            "float_precision": "float64",
            "column_wave": "wave",
            "column_flux": "flux",
            "synthetic_spectrum": False,
            "anchor_file": "",
        }

        # Mise à jour avec la config utilisateur si fournie
        if rassine_config is not None:
            default_rassine_config.update(rassine_config)

        # Ajouter le chemin de Rassine au PYTHONPATH
        rassine_path = os.path.join(os.path.dirname(__file__), "..", "Rassine_public")
        rassine_path = os.path.abspath(rassine_path)
        if rassine_path not in sys.path:
            sys.path.insert(0, rassine_path)

        try:
            # Normalisation du template
            print("Normalisation du template...")
            template_normalized = _normalize_spectrum_with_rassine(
                wavegrid_masked, template_masked, default_rassine_config
            )

            # Normalisation de chaque spectre du dataset par batches
            print(
                f"Normalisation de {n_spectra} spectres par batches de {batch_size}..."
            )
            spectra_normalized = np.zeros_like(spectra_masked)

            for i in range(0, n_spectra, batch_size):
                end_idx = min(i + batch_size, n_spectra)
                print(f"  Batch {i // batch_size + 1}: spectres {i + 1}-{end_idx}")

                # Traiter chaque spectre du batch
                for j in range(i, end_idx):
                    spectra_normalized[j] = _normalize_spectrum_with_rassine(
                        wavegrid_masked, spectra_masked[j], default_rassine_config
                    )

                # Nettoyer le cache mémoire périodiquement
                if i % (batch_size * 5) == 0 and i > 0:
                    import gc

                    gc.collect()

            print("✅ Normalisation Rassine terminée")

        except ImportError as e:
            print(f"❌ Erreur d'import Rassine: {e}")
            print("Continuera sans normalisation...")
            template_normalized = template_masked
            spectra_normalized = spectra_masked
        except Exception as e:
            print(f"❌ Erreur lors de la normalisation Rassine: {e}")
            print("Continuera sans normalisation...")
            template_normalized = template_masked
            spectra_normalized = spectra_masked
    else:
        print("Pas de normalisation demandée")
        template_normalized = template_masked / np.max(template_masked)
        spectra_normalized = spectra_masked

    # 5) Calcul du nombre de bins complets pour le downscaling
    Npix = wavegrid_masked.size
    n_bins = Npix // downscaling_factor
    trim = n_bins * downscaling_factor

    print(
        f"\n📐 Downscaling: {Npix} pixels → {n_bins} bins (facteur {downscaling_factor})"
    )

    # 6) Trim et reshape + moyenne
    wavegrid_trim = wavegrid_masked[:trim].reshape(n_bins, downscaling_factor)
    template_trim = template_normalized[:trim].reshape(n_bins, downscaling_factor)
    spectra_trim = spectra_normalized[:, :trim].reshape(
        n_spectra, n_bins, downscaling_factor
    )

    wavegrid_ds = wavegrid_trim.mean(axis=1)
    template_ds = template_trim.mean(axis=1)
    spectra_ds = spectra_trim.mean(axis=2)

    # 8) Ajout de bruit photonique réaliste (optionnel)
    if add_photon_noise:
        print("\n🔊 Ajout de bruit photonique réaliste...")

        # Configurer la graine aléatoire pour reproductibilité
        if noise_seed is not None:
            np.random.seed(noise_seed)
            print(f"   Graine aléatoire: {noise_seed}")

        # Ajouter du bruit au template
        template_ds = _add_photon_noise(template_ds, snr_target)

        # Ajouter du bruit à chaque spectre du dataset
        print(f"   Ajout de bruit à {n_spectra} spectres...")
        for i in range(n_spectra):
            if i % 500 == 0 and i > 0:
                print(f"     Spectre {i}/{n_spectra}")
            spectra_ds[i] = _add_photon_noise(spectra_ds[i], snr_target)

        print("✅ Bruit photonique ajouté")

    # 10) Création du dataset sans activité avec gestion mémoire améliorée
    spectra_ds_no_activity = None
    if (
        planets_amplitudes is not None
        and planets_periods is not None
        and planets_phases is not None
    ):
        print("\n🌌 Injection du signal planétaire dans le dataset")
        spectra_ds = inject_dataset(
            spectra=torch.tensor(spectra_ds, device="cuda"),
            wavegrid=torch.tensor(wavegrid_ds, device="cuda"),
            time_values=time_values,
            planets_amplitudes=planets_amplitudes,
            planets_periods=planets_periods,
            planets_phases=planets_phases,
            batch_size=batch_size,  # Utiliser le même batch_size
        )
        spectra_ds = spectra_ds.detach().cpu().numpy()
        print("✅ Dataset injecté")

        print("\n🌌 Création du dataset sans activité...")

        # Utiliser la même logique de batch que pour l'injection
        spectra_ds_no_activity = inject_dataset(
            spectra=torch.tensor(template_ds, device="cuda")
            .unsqueeze(0)
            .repeat(n_spectra, 1),
            wavegrid=torch.tensor(wavegrid_ds, device="cuda"),
            time_values=time_values,
            planets_amplitudes=planets_amplitudes,
            planets_periods=planets_periods,
            planets_phases=planets_phases,
            batch_size=batch_size,  # Utiliser le même batch_size
        )

        spectra_ds_no_activity = spectra_ds_no_activity.detach().cpu().numpy()
        print("✅ Dataset sans activité créé")

    # 11) Sauvegarde avec nettoyage mémoire
    print("\n💾 Sauvegarde des données...")
    # Métadonnées standardisées
    metadata = {
        "n_spectra": int(n_spectra),
        "n_pixels": int(len(wavegrid_ds)),
        "wavemin": float(wavemin),
        "wavemax": float(wavemax),
        "data_dtype": str(np.array(template_ds).dtype),
        "planets_periods": planets_periods,
        "planets_amplitudes": planets_amplitudes,
        "planets_phases": planets_phases,
        "downscaling_factor": downscaling_factor,
        "use_rassine": use_rassine,
        "add_photon_noise": add_photon_noise,
        "snr_target": snr_target,
        "noise_seed": noise_seed,
        "original_pixels": int(Npix),
        "downscaled_pixels": int(n_bins),
        "batch_size": int(batch_size),
    }

    save_data = {
        "wavegrid": wavegrid_ds,
        "template": template_ds,
        "spectra": spectra_ds,
        "time_values": time_values[:n_spectra],
        "metadata": metadata,
    }

    # Ajouter les spectres sans activité seulement s'ils existent
    if spectra_ds_no_activity is not None:
        save_data["spectra_no_activity"] = spectra_ds_no_activity

    np.savez_compressed(output_filepath, **save_data)

    # Nettoyage final de la mémoire
    del spectra_masked, spectra_normalized, spectra_ds, template_normalized
    if spectra_ds_no_activity is not None:
        del spectra_ds_no_activity

    import gc

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"💾 Fichier de sortie créé: {output_filepath}")
    print(f"   - {n_spectra} spectres")
    print(f"   - {n_bins} pixels spectraux")
    print(f"   - Gamme: {wavegrid_ds.min():.1f} - {wavegrid_ds.max():.1f} Å")
    print("🧹 Nettoyage mémoire terminé")


def create_rvdatachallenge_dataset(
    star_name="STAR1136",
    output_filepath="data/npz_datasets/test_rv_datachallenge.npz",
    n_spectra=None,
    wavemin=None,
    wavemax=None,
    data_dtype=torch.float32,
):
    """
    Crée un dataset (fichier npz) standardisé pour le RV Data Challenge: data/npz_datasets/*.npz

    Le fichier contiendra: spectra, wavegrid, time_values, metadata (+ template).
    """
    if star_name == "STAR1136":
        dataset_dirpath = (
            "data/rv_datachallenge/Sun_B57001_E61001_planet-FallChallenge1"
        )
    elif star_name == "STAR1138":
        dataset_dirpath = (
            "data/rv_datachallenge/Sun_B57002_E61002_planet-FallChallenge2"
        )
    elif star_name == "STAR1134":
        dataset_dirpath = (
            "data/rv_datachallenge/Sun_B57000_E61000_planet-FallChallenge3"
        )
    else:
        raise ValueError(
            f"Nom d'étoile inconnu: {star_name}. Utilisez 'STAR1136', 'STAR1138' ou 'STAR1134'."
        )

    dataset_filepath = f"{dataset_dirpath}/HARPN/{star_name}_HPN_flux_YVA.npy"

    analyse_material = np.load(
        f"data/rv_datachallenge/Sun_B57001_E61001_planet-FallChallenge1/HARPN/{star_name}_HPN_Analyse_material.p",
        allow_pickle=True,
    )

    wavegrid = analyse_material["wave"].to_numpy()
    template = analyse_material["stellar_template"].to_numpy()

    if wavemin is None:
        wavemin = wavegrid.min()
    if wavemax is None:
        wavemax = wavegrid.max()

    wave_mask = (wavegrid >= wavemin) & (wavegrid <= wavemax)
    wavegrid = wavegrid[wave_mask]
    template = template[wave_mask]

    analyse_summary = pd.read_csv(
        f"data/rv_datachallenge/Sun_B57001_E61001_planet-FallChallenge1/HARPN/{star_name}_HPN_Analyse_summary.csv"
    )

    if n_spectra is None:
        n_spectra = analyse_summary.shape[0]
    data = np.load(dataset_filepath)
    data = data[:n_spectra, :]
    data = data[:, wave_mask]

    # Types numpy standardisés
    if data_dtype == torch.float16:
        np_dtype = np.float16
    elif data_dtype == torch.float64:
        np_dtype = np.float64
    else:
        np_dtype = np.float32

    spectra_np = data.astype(np_dtype, copy=False)
    wavegrid_np = wavegrid.astype(np_dtype, copy=False)
    template_np = template.astype(np_dtype, copy=False)
    time_values_np = analyse_summary["jdb"][:n_spectra].to_numpy().astype(np_dtype)

    n_pixels = wavegrid_np.shape[0]

    # Planets info indisponible ici
    planets_periods = None
    planets_amplitudes = None
    planets_phases = None

    metadata = {
        "n_spectra": int(n_spectra),
        "n_pixels": int(n_pixels),
        "wavemin": float(wavemin),
        "wavemax": float(wavemax),
        "data_dtype": str(np_dtype),
        "planets_periods": planets_periods,
        "planets_amplitudes": planets_amplitudes,
        "planets_phases": planets_phases,
        "source": "rv_datachallenge",
        "star_name": star_name,
    }

    save_data = {
        "spectra": spectra_np,
        "wavegrid": wavegrid_np,
        "template": template_np,
        "time_values": time_values_np,
        "metadata": metadata,
    }

    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    np.savez_compressed(output_filepath, **save_data)

    print(f"💾 Fichier de sortie créé: {output_filepath}")
    print(f"   - {n_spectra} spectres | {n_pixels} pixels")
    print(f"   - Gamme: {wavegrid_np.min():.1f} - {wavegrid_np.max():.1f} Å")


def _normalize_spectrum_with_rassine(wave, flux, config):
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


def _add_photon_noise(spectrum, snr_target=None):
    """
    Ajoute du bruit photonique réaliste à un spectre.
    Le bruit photonique suit une distribution de Poisson.

    Parameters
    ----------
    spectrum : np.ndarray
        Spectre auquel ajouter le bruit
    snr_target : float, optional
        SNR cible. Si None, utilise le niveau de signal existant pour déterminer le bruit.

    Returns
    -------
    np.ndarray
        Spectre avec bruit photonique ajouté
    """
    # Convertir les valeurs négatives en zéro (nécessaire pour Poisson)
    spectrum_positive = np.maximum(spectrum, 0)

    if snr_target is not None:
        # Calcul du niveau de signal pour atteindre le SNR cible
        # SNR = signal / sqrt(signal) = sqrt(signal)
        # Donc: signal = SNR²
        signal_level = snr_target**2

        # Normaliser le spectre pour avoir le bon niveau de signal moyen
        current_mean = np.mean(spectrum_positive)
        if current_mean > 0:
            scaling_factor = signal_level / current_mean
            spectrum_scaled = spectrum_positive * scaling_factor
        else:
            spectrum_scaled = spectrum_positive
    else:
        # Utiliser le niveau de signal existant
        spectrum_scaled = spectrum_positive

        # Assurer un niveau minimum pour éviter un bruit trop faible
        min_signal = 100  # Niveau minimum de photons
        spectrum_scaled = np.maximum(spectrum_scaled, min_signal)

    # Générer le bruit photonique (distribution de Poisson)
    # Le bruit de Poisson a une variance égale à la moyenne
    try:
        # Utiliser poisson pour générer des échantillons
        noisy_spectrum = np.random.poisson(spectrum_scaled).astype(float)
    except ValueError:
        # Si les valeurs sont trop grandes pour Poisson, utiliser une approximation gaussienne
        # Pour de grands nombres, Poisson(λ) ≈ Normal(λ, √λ)
        noise = np.random.normal(0, np.sqrt(spectrum_scaled))
        noisy_spectrum = spectrum_scaled + noise

    # Éviter les valeurs négatives
    noisy_spectrum = np.maximum(noisy_spectrum, 0)

    return noisy_spectrum


if __name__ == "__main__":
    # dataset = SpectrumDataset()

    # dataset_metadata = dataset.to_dict()
    # print("Metadata du dataset:", dataset_metadata)
    # new_dataset = SpectrumDataset(
    #     **dataset_metadata,  # Recharger le dataset avec les mêmes paramètres
    # )

    # Exemple d'appel optimisé pour éviter les crashes - Version SÉCURISÉE recommandée
    print("🚀 Démarrage de la création du dataset avec optimisation automatique...")

    create_soap_gpu_paper_dataset(
        spectra_filepath="data/soap_gpu_paper/spec_cube_tot.h5",
        spec_filepath="data/soap_gpu_paper/spec_master.npz",
        output_filepath="data/npz_datasets/dataset_1000specs_5000_5300_Kp1e-1_P100.npz",
        n_spectra=1000,
        wavemin=5000,
        wavemax=5300,
        downscaling_factor=2,
        use_rassine=True,
        rassine_config=None,
        add_photon_noise=False,
        planets_amplitudes=[0.1],
        planets_periods=[100],
        planets_phases=[0.0],
    )

    # Alternative avec paramètres manuels (utiliser seulement si vous connaissez votre système)
    # create_soap_gpu_paper_dataset(
    #     spectra_filepath="data/soap_gpu_paper/spec_cube_tot.h5",
    #     spec_filepath="data/soap_gpu_paper/spec_master.npz",
    #     output_filepath="data/soap_gpu_paper/dataset_1000specs_5000_6000_Kp10_P100_manual.npz",
    #     n_spectra=1000,
    #     wavemin=5000,
    #     wavemax=6000,
    #     downscaling_factor=2,
    #     use_rassine=True,
    #     rassine_config=None,
    #     add_photon_noise=False,
    #     planets_amplitudes=[10],
    #     planets_periods=[100],
    #     batch_size=50,  # Ajustez selon votre système
    #     max_gpu_memory_gb=4,  # Ajustez selon votre GPU
    #     force_cpu=False,  # True pour forcer CPU
    # )
