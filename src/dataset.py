import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Union
from src.interpolate import augment_spectra_uniform
from src.utils import get_mask
from src.ccf import build_CCF_masks_sparse

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
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
        return t.to(dtype=dtype, copy=False).contiguous()
    return torch.tensor(x, dtype=dtype).contiguous()


class SpectrumDataset(Dataset):
    """
    Dataset pour charger des spectres depuis un .npz (format harmonisé, un seul split).

        Clés attendues:
            - wavegrid (P,), template (P,)
            - spectra (N,P)
            - time_values (N,)
            - Optionnel: activity, spectra_no_activity, v_true
            - metadata (dict) incluant: n_spectra, n_pixels, wavemin, wavemax,
                planets_periods, planets_amplitudes, planets_phases (pour fallback v_true)
    """

    def __init__(
        self,
        dataset_filepath: str,
        split: str = "all",  # ignoré, conservé pour compatibilité API
        data_dtype: torch.dtype = torch.float32,
        cuda: bool = True,
        mask_weights_fid: Optional[Union[bool, str, np.ndarray, torch.Tensor]] = None,
    ):
        if not dataset_filepath.endswith(".npz"):
            raise ValueError("dataset_filepath doit pointer vers un fichier .npz")
        if not os.path.exists(dataset_filepath):
            raise FileNotFoundError(f"NPZ dataset not found: {dataset_filepath}")

        self.dataset_filepath = dataset_filepath
        self.split = split  # ignoré, conservé pour compatibilité
        self.data_dtype = data_dtype

        self._init_from_npz(dataset_filepath, data_dtype)

        # Optionnel: override des weights_fid par un masque basé sur les raies
        if mask_weights_fid is not None:
            # Only perform override when explicitly requested:
            # - bool True means use default mask ("G2")
            # - str indicates mask_type
            # - ndarray / torch.Tensor indicates explicit weights vector
            if isinstance(mask_weights_fid, (bool, np.bool_)):
                if mask_weights_fid:
                    self._override_weights_fid(mask_weights_fid)
                    self.metadata["mask_weights_fid"] = True
            elif isinstance(mask_weights_fid, str) or isinstance(
                mask_weights_fid, (np.ndarray, torch.Tensor)
            ):
                self._override_weights_fid(mask_weights_fid)
                self.metadata["mask_weights_fid"] = mask_weights_fid
            # Otherwise (e.g. False), do nothing

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
            # Accès direct à la clé (plus de split)
            return ds[name] if name in ds.files else None

        spectra_np = pick("spectra")
        if spectra_np is None:
            raise KeyError("Clé 'spectra' manquante dans le npz.")

        time_values_np = pick("time_values")
        if time_values_np is None:
            raise KeyError("Clé 'time_values' manquante dans le npz.")

        activity_np = pick("activity")
        spectra_no_activity_np = pick("spectra_no_activity")
        v_true_np = pick("v_true")
        weights_fid_np = pick("weights_fid")
        activity_proxies_norm_np = pick("activity_proxies_norm")

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
        self.activity_proxies_norm = _to_tensor(activity_proxies_norm_np, data_dtype)
        self.planet_periods = self.metadata.get("planets_periods", [])
        self.planet_amplitudes = self.metadata.get("planets_amplitudes", [])
        self.planet_phases = self.metadata.get("planets_phases", [])
        self.time_values = _to_tensor(time_values_np, data_dtype)
        self.activity = _to_tensor(activity_np, data_dtype)
        self.spectra_no_activity = _to_tensor(spectra_no_activity_np, data_dtype)
        self.v_true = _to_tensor(v_true_np, data_dtype)
        self.weights_fid = _to_tensor(weights_fid_np, data_dtype)

        # Tailles / bornes
        self.n_spectra = self.spectra.shape[0]
        self.n_pixels = self.spectra.shape[1]
        self.wavemin = float(
            self.metadata.get("wavemin", float(self.wavegrid.min().item()))
        )
        self.wavemax = float(
            self.metadata.get("wavemax", float(self.wavegrid.max().item()))
        )

        # Sanity checks rapides
        assert self.time_values.shape[0] == self.n_spectra, (
            "time_values et spectra désalignés"
        )
        assert self.v_true.shape[0] == self.n_spectra, "v_true et spectra désalignés"
        if self.activity is not None:
            assert self.activity.shape == self.spectra.shape, (
                "activity et spectra doivent avoir la même forme"
            )
        if self.spectra_no_activity is not None:
            assert self.spectra_no_activity.shape == self.spectra.shape, (
                "spectra_no_activity et spectra doivent avoir la même forme"
            )

    # --------- helpers masque de raies / weights_fid ----------
    def _build_binary_line_mask(
        self, mask_type: str = "G2", window_size_velocity: float = None
    ) -> torch.Tensor:
        """Construit un masque binaire 1D (longueur = n_pixels) qui met 1
        pour les pixels se trouvant dans une fenêtre autour de chaque raie.

        La fenêtre est définie en m/s (window_size_velocity) et convertie en
        étendue en longueur d'onde via sigma = lam0 * (window_size_velocity / c).
        On prend le support ±4*sigma (comme dans build_CCF_masks_sparse).

        Args:
            mask_type: nom du masque (ex: "G2").
            window_size_velocity: sigma en m/s; si None, on lit
                self.metadata['mask_window_velocity'] ou 410.0 par défaut.
        """
        # Prépare wavegrid côté CPU pour le matching numpy, garde dtype conforme
        wavegrid_t = self.wavegrid
        wavegrid_np = wavegrid_t.detach().cpu().numpy()
        P = wavegrid_np.shape[0]

        # window en m/s
        if window_size_velocity is None:
            window_size_velocity = float(
                self.metadata.get("mask_window_velocity", 820.0)
            )

        c = 299792458.0

        mask_arr = get_mask(mask_type)
        line_pos = mask_arr[:, 0].astype(wavegrid_np.dtype, copy=False)

        # Filtrer les raies dans les bornes de la grille (au moins partiellement)
        in_bounds = (line_pos + 0 >= wavegrid_np[0]) & (line_pos <= wavegrid_np[-1])
        line_pos = line_pos[in_bounds]
        if line_pos.size == 0:
            # Aucun recouvrement -> retourne tout à zéro
            zeros = torch.zeros(P, dtype=self.data_dtype, device=wavegrid_t.device)
            return zeros

        # Construire le masque en utilisant la même logique que build_CCF_masks_sparse
        # On récupère les positions et poids depuis le fichier de masque
        line_weights = mask_arr[:, 1]

        # build_CCF_masks_sparse attend des numpy arrays
        try:
            CCF_mask = build_CCF_masks_sparse(
                line_pos=line_pos,
                line_weights=line_weights,
                v_grid=np.array([0.0], dtype=float),
                wavegrid=wavegrid_np,
                window_size_velocity=float(window_size_velocity),
            )
        except Exception:
            print("Erreur lors de la construction du masque CCF")
            # En cas d'échec (sécurité), retomber sur le masque simple ±4σ
            weights_np = np.zeros(
                P, dtype=np.float32 if self.data_dtype == torch.float32 else np.float64
            )
            for lam0 in line_pos:
                sigma = lam0 * (window_size_velocity / c)
                start = lam0 - 4.0 * sigma
                end = lam0 + 4.0 * sigma
                i0 = int(np.searchsorted(wavegrid_np, start))
                i1 = int(np.searchsorted(wavegrid_np, end))
                if i0 < 0:
                    i0 = 0
                if i1 >= P:
                    i1 = P - 1
                if i1 >= i0:
                    weights_np[i0 : i1 + 1] = 1.0
            weights_t = torch.from_numpy(weights_np).to(
                dtype=self.data_dtype, device=wavegrid_t.device
            )
            return weights_t.contiguous()

        # CCF_mask est CSR shape (1, P). Convertir en tableau et rendre binaire (>0)
        mask_arr_1d = CCF_mask.toarray().ravel()
        bin_np = (mask_arr_1d > 0).astype(
            np.float32 if self.data_dtype == torch.float32 else np.float64
        )
        weights_t = torch.from_numpy(bin_np).to(
            dtype=self.data_dtype, device=wavegrid_t.device
        )
        return weights_t.contiguous()

    def _override_weights_fid(
        self, mask_weights_fid: Union[bool, str, np.ndarray, torch.Tensor]
    ) -> None:
        """Override `self.weights_fid`.

        Règles:
        - Si mask_weights_fid est un np.ndarray/torch.Tensor de longueur n_pixels:
          on l'utilise directement (cast vers dtype du dataset).
        - Si mask_weights_fid est True: on construit un masque binaire avec type "G2".
        - Si mask_weights_fid est une str: on l'utilise comme mask_type pour get_mask.
        """
        # Cas tableau explicite
        if isinstance(mask_weights_fid, (np.ndarray, torch.Tensor)):
            t = _to_tensor(mask_weights_fid, self.data_dtype)
            if t.dim() != 1 or t.shape[0] != self.n_pixels:
                raise ValueError(
                    "mask_weights_fid fourni doit être 1D et de même longueur que wavegrid"
                )
            # Conserver device de wavegrid
            self.weights_fid = t.to(device=self.wavegrid.device).contiguous()
            return

        # Cas bool/str -> construire masque binaire
        if mask_weights_fid is True:
            mask_type = "G2"
        elif isinstance(mask_weights_fid, str):
            mask_type = mask_weights_fid
        else:
            raise ValueError(
                "mask_weights_fid doit être un bool, une str (mask_type) ou un vecteur 1D"
            )

        self.weights_fid = self._build_binary_line_mask(mask_type=mask_type)

    # --------- API Dataset ----------
    def __len__(self):
        return self.n_spectra

    def __getitem__(self, idx):
        # On conserve ton comportement minimal (retourne le spectre).
        # Si tu veux plus d’info, tu peux changer ici pour retourner un dict.
        return self.spectra[idx], idx

    # --------- utilitaires ----------
    def _estimate_memory_usage(self):
        def mb(t):
            return 0 if t is None else t.numel() * t.element_size() / (1024 * 1024)

        return (
            mb(self.spectra)
            + mb(self.wavegrid)
            + mb(self.template)
            + mb(self.time_values)
            + mb(getattr(self, "weights_fid", None))
        )

    def move_to_cuda(self):
        if torch.cuda.is_available():
            for name in [
                "spectra",
                "wavegrid",
                "template",
                "time_values",
                "activity",
                "spectra_no_activity",
                "activity_proxies_norm",
                "v_true",
                "weights_fid",
            ]:
                t = getattr(self, name, None)
                if t is not None:
                    setattr(self, name, t.cuda())

    def convert_dtype(self, new_dtype: torch.dtype):
        def cast(name):
            t = getattr(self, name, None)
            if t is not None:
                setattr(self, name, t.to(dtype=new_dtype))

        old = self._estimate_memory_usage()
        for k in [
            "spectra",
            "wavegrid",
            "template",
            "time_values",
            "activity",
            "spectra_no_activity",
            "v_true",
            "weights_fid",
        ]:
            cast(k)
        self.data_dtype = new_dtype
        new = self._estimate_memory_usage()
        print(
            f"Conversion dtype: {old:.2f} -> {new:.2f} MB (-{(old - new) / old * 100:.1f}%)"
        )
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


def _take_opt(
    dataset, attr: str, enabled: bool, batch_indices: torch.Tensor, MB: int, device
) -> torch.Tensor:
    """Sélectionne les données optionnelles pour le batch en respectant les formes attendues.

    Règles:
    - Si tensor a une première dimension = n_spectra (par ex. [N, K] ou [N, P]), on indexe avec batch_indices -> [MB, ...]
    - Si tensor est 1D (par ex. [K] ou [P]), on le réplique sur le batch -> [MB, K]
    - Si tensor a shape [1, ...], on l'étend -> [MB, ...]
    - Sinon, si la première dimension vaut déjà MB, on renvoie tel quel.
    Retourne None si `enabled` est False ou si l'attribut est absent.
    """
    if not enabled:
        return None
    x = getattr(dataset, attr, None)
    if x is None:
        return None
    # Assure le bon device
    if x.device != device:
        x = x.to(device)

    # Cas per-sample: première dim = N (n_spectra)
    if x.dim() >= 1 and x.shape[0] == dataset.n_spectra:
        return x[batch_indices]

    # Cas 1D: répliquer pour chaque élément du batch
    if x.dim() == 1:
        return x.unsqueeze(0).expand(MB, -1).contiguous()

    # Cas [1, ...]: étendre sur MB
    if x.dim() >= 1 and x.shape[0] == 1:
        return x.expand(MB, *x.shape[1:]).contiguous()

    # Déjà à la bonne taille
    if x.dim() >= 1 and x.shape[0] == MB:
        return x

    # Forme non reconnue -> None pour éviter les plantages inattendus
    return None


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
        # batch : liste de (y_obs, idx) de taille B
        spectra_list, indices_list = zip(*batch)

        batch_yobs = torch.stack(spectra_list, dim=0)  # [B, n_pix]
        B, n_pix = batch_yobs.shape

        # Étendre à M * B
        if M > 1:
            batch_yobs = batch_yobs.unsqueeze(1).expand(B, M, n_pix).reshape(-1, n_pix)
        # sinon, on garde tel quel (B, n_pix)
        MB = batch_yobs.shape[0]

        # Indices alignés sur M*B et sur le bon device
        batch_indices = torch.as_tensor(
            indices_list, dtype=torch.long, device=batch_yobs.device
        )
        if M > 1:
            batch_indices = batch_indices.repeat_interleave(M)  # [M*B]

        # Wavegrid sur le bon device/dtype, sans dupliquer la mémoire
        batch_wavegrid = (
            dataset.wavegrid.to(batch_yobs.device, dtype=batch_yobs.dtype)
            .unsqueeze(0)
            .expand(MB, -1)
            .contiguous()
        )

        # Augment
        batch_yaug, batch_voffset = augment_spectra_uniform(
            batch_yobs,
            batch_wavegrid,
            vmin=vmin,
            vmax=vmax,
            interpolate=interpolate,
            extrapolate=extrapolate,
            out_dtype=out_dtype,
        )

        # Préparer indices batch et device pour la sélection des optionnels
        batch_device = batch_yobs.device
        # batch_indices est déjà [MB]

        # Optionnels (via helper _take_opt)
        batch_weights_fid = _take_opt(
            dataset,
            "weights_fid",
            True,
            batch_indices,
            MB,
            batch_device,
        )
        batch_yact_true = _take_opt(
            dataset,
            "activity",
            True,
            batch_indices,
            MB,
            batch_device,
        )
        batch_activity_proxies_norm = _take_opt(
            dataset,
            "activity_proxies_norm",
            dataset.metadata.get("activity_proxies_included", False),
            batch_indices,
            MB,
            batch_device,
        )

        return (
            batch_yobs,  # [M*B, n_pix]
            batch_yaug,  # [M*B, n_pix]
            batch_voffset,  # [M*B]
            batch_wavegrid,  # [M*B, n_pix]
            batch_weights_fid,  # [M*B, ...] ou None
            batch_indices,  # [M*B]
            batch_yact_true,  # [M*B, n_pix] ou None
            batch_activity_proxies_norm,  # [M*B, P] ou None
        )

    return collate_fn


if __name__ == "__main__":
    spec_dset = SpectrumDataset(
        dataset_filepath="/home/tliopis/Codes/exoplanets_llopis_mary_2025/data/npz_datasets/soapgpu_nst120_nsv120_5000-5050_dx2_sm3_p60_k0p1_phi0.npz"
    )
