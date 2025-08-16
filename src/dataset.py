import os
import numpy as np
import torch
from torch.utils.data import Dataset
from src.interpolate import (
    augment_spectra_uniform,
    shift_spectra_linear,
    shift_spectra_cubic,
)
from src.utils import get_mask

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
        # --- Lignes / fenêtres ---
        use_lines: bool = True,
        mask_type: str = "G2",
        window_half_width: int = 16,  # W = 2*half+1 pixels
    ):
        if not dataset_filepath.endswith(".npz"):
            raise ValueError("dataset_filepath doit pointer vers un fichier .npz")
        if not os.path.exists(dataset_filepath):
            raise FileNotFoundError(f"NPZ dataset not found: {dataset_filepath}")

        self.dataset_filepath = dataset_filepath
        self.split = split  # ignoré, conservé pour compatibilité
        self.data_dtype = data_dtype
        self.use_lines = use_lines
        self.mask_type = mask_type
        self.window_half_width = int(window_half_width)

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

        # Préparer la représentation par raies si demandé
        if self.use_lines:
            self._build_line_windows()
        else:
            # marqueurs vides si non utilisé
            self.spectra_lines = None
            self.template_lines = None
            self.wavegrid_lines = None
            self.line_positions = None
            self.line_positions_norm = None
            self.line_weights = None
            self.line_center_idx = None
            self.window_indices = None
            self.n_lines = 0
            self.window_size = 0

    # --------- API Dataset ----------
    def __len__(self):
        return self.n_spectra

    def __getitem__(self, idx):
        # Retourne par défaut les fenêtres centrées sur les raies si disponibles
        if getattr(self, "spectra_lines", None) is not None:
            return self.spectra_lines[idx], idx  # [M, W], idx
        # Fallback: spectre complet
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
            + mb(getattr(self, "spectra_lines", None))
            + mb(getattr(self, "wavegrid_lines", None))
            + mb(getattr(self, "template_lines", None))
            + mb(getattr(self, "line_positions", None))
            + mb(getattr(self, "line_positions_norm", None))
            + mb(getattr(self, "line_weights", None))
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
                # Lignes
                "spectra_lines",
                "template_lines",
                "wavegrid_lines",
                "line_positions",
                "line_positions_norm",
                "line_weights",
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
            # Lignes
            "spectra_lines",
            "template_lines",
            "wavegrid_lines",
            "line_positions",
            "line_positions_norm",
            "line_weights",
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
            f"spectra_lines={shape_dtype(getattr(self, 'spectra_lines', None))}\n"
            f"spectra_no_activity={shape_dtype(self.spectra_no_activity)}\n"
            f"activity={shape_dtype(self.activity)}\n"
            f"wavegrid={shape_dtype(self.wavegrid)}\n"
            f"wavegrid_lines={shape_dtype(getattr(self, 'wavegrid_lines', None))}\n"
            f"template={shape_dtype(self.template)}\n"
            f"template_lines={shape_dtype(getattr(self, 'template_lines', None))}\n"
            f"line_positions={shape_dtype(getattr(self, 'line_positions', None))}\n"
            f"line_positions_norm={shape_dtype(getattr(self, 'line_positions_norm', None))}\n"
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

    # --------- construction des fenêtres autour des raies ----------
    def _build_line_windows(self):
        """
        Construit les tenseurs:
          - spectra_lines: [N, M, W]
          - template_lines: [M, W]
          - wavegrid_lines: [M, W]
          - line_positions: [M]
          - line_weights: [M]

        où M = nb de raies du masque sélectionnées dans [wavemin, wavemax]
           W = 2*window_half_width + 1
        """
        # Charger le masque
        mask_np = get_mask(self.mask_type)  # shape [L, 2]
        if mask_np.ndim != 2 or mask_np.shape[1] < 2:
            raise ValueError("Le masque doit avoir deux colonnes: position, poids")

        # Sélection des raies dans l'intervalle
        mask_pos = mask_np[:, 0]
        mask_w = mask_np[:, 1]
        in_range = (mask_pos >= self.wavemin) & (mask_pos <= self.wavemax)
        pos_sel = mask_pos[in_range]
        w_sel = mask_w[in_range]

        if pos_sel.size == 0:
            raise ValueError(
                f"Aucune raie du masque {self.mask_type} dans [{self.wavemin}, {self.wavemax}]"
            )

        # Trouver l'index du pixel le plus proche pour chaque raie
        wg_np = self.wavegrid.detach().cpu().numpy()

        def nearest_index(arr: np.ndarray, x: float) -> int:
            i = np.searchsorted(arr, x)
            if i == 0:
                return 0
            if i >= arr.size:
                return arr.size - 1
            # choisir le plus proche entre i-1 et i
            return i - 1 if (x - arr[i - 1]) <= (arr[i] - x) else i

        centers = np.array([nearest_index(wg_np, x) for x in pos_sel], dtype=np.int64)

        # Garder seulement les raies dont la fenêtre tient entièrement dans le spectre
        H = int(self.window_half_width)
        W = 2 * H + 1
        valid = (centers - H >= 0) & (centers + H < self.n_pixels)
        centers = centers[valid]
        pos_sel = pos_sel[valid]
        w_sel = w_sel[valid]

        if centers.size == 0:
            raise ValueError(
                "Les fenêtres autour des raies sélectionnées dépassent les bords du spectre. Réduire window_half_width."
            )

        # Indices de fenêtres [M, W]
        offsets = np.arange(-H, H + 1, dtype=np.int64)[None, :]
        centers2d = centers[:, None]
        win_idx = centers2d + offsets  # [M, W], garanti in-bounds

        # Construire les tenseurs lignes
        # -> indices en torch.Long
        win_idx_t = torch.from_numpy(win_idx.astype(np.int64))

        # template_lines / wavegrid_lines: [M, W]
        self.template_lines = self.template[win_idx_t]
        self.wavegrid_lines = self.wavegrid[win_idx_t]

        # spectra_lines: [N, M, W] via take_along_dim
        # Préparer indices [N, M, W]
        idx_exp = win_idx_t.unsqueeze(0).expand(self.n_spectra, -1, -1)
        M_lines = idx_exp.shape[1]
        spectra_exp = self.spectra.unsqueeze(1).expand(-1, M_lines, -1)  # [N, M, P]
        self.spectra_lines = torch.take_along_dim(spectra_exp, idx_exp, dim=2)
        # self.spectra_lines: [N, M, W]

        # Mémoriser infos lignes
        self.line_positions = torch.as_tensor(pos_sel, dtype=self.data_dtype)
        # Normalisation simple 0..1 dans l'intervalle spectral global
        denom = (self.wavemax - self.wavemin) if (self.wavemax > self.wavemin) else 1.0
        pos_norm_np = (pos_sel - self.wavemin) / denom
        self.line_positions_norm = torch.as_tensor(pos_norm_np, dtype=self.data_dtype)
        self.line_weights = torch.as_tensor(w_sel, dtype=self.data_dtype)
        self.line_center_idx = torch.as_tensor(centers, dtype=torch.long)
        self.window_indices = win_idx_t  # [M, W] (long)
        self.n_lines = int(centers.size)
        self.window_size = int(W)


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
    M=1,  # nombre d'augmentations par échantillon
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

        # Détection mode lignes vs spectre complet
        lines_mode = (
            hasattr(dataset, "spectra_lines") and dataset.spectra_lines is not None
        )

        if lines_mode:
            # y_obs: [B, M_lines, W]
            batch_yobs = torch.stack(spectra_list, dim=0)
            B, M_lines, W = batch_yobs.shape

            # Étendre à M * B (M = nb d'augmentations par échantillon)
            if M > 1:
                batch_yobs = (
                    batch_yobs.unsqueeze(1)
                    .expand(B, M, M_lines, W)
                    .reshape(-1, M_lines, W)
                )
            MB = batch_yobs.shape[0]

            # Indices alignés [MB]
            batch_indices = torch.as_tensor(
                indices_list, dtype=torch.long, device=batch_yobs.device
            )
            if M > 1:
                batch_indices = batch_indices.repeat_interleave(M)

            # Wavegrid lignes: [MB, M_lines, W]
            base_wave = dataset.wavegrid_lines.to(
                batch_yobs.device, dtype=batch_yobs.dtype
            )
            batch_wavegrid = base_wave.unsqueeze(0).expand(MB, -1, -1).contiguous()

            # Échantillonner une vitesse par élément (même v pour toutes les raies d'un échantillon)
            batch_voffset = (
                torch.from_numpy(np.random.uniform(vmin, vmax, size=(MB, 1)))
                .to(batch_yobs.device)
                .double()
            )

            # Appliquer le shift par raie en aplatissant [MB*M_lines, W]
            y_flat = batch_yobs.reshape(MB * M_lines, W)
            w_flat = batch_wavegrid.reshape(MB * M_lines, W)
            v_flat = batch_voffset.repeat_interleave(M_lines, dim=0)

            if interpolate == "linear":
                yaug_flat = shift_spectra_linear(
                    spectra=y_flat,
                    wavegrid=w_flat,
                    velocities=v_flat,
                    extrapolate=extrapolate,
                    return_mask=False,
                )
            elif interpolate == "cubic":
                yaug_flat = shift_spectra_cubic(
                    spectra=y_flat,
                    wavegrid=w_flat,
                    velocities=v_flat,
                    return_mask=False,
                )
            else:
                raise ValueError("interpolate doit être 'linear' ou 'cubic'")

            # Remise en forme [MB, M_lines, W]
            batch_yaug = yaug_flat.reshape(MB, M_lines, W).to(out_dtype)
            batch_yobs = batch_yobs.to(out_dtype)
            batch_voffset = batch_voffset.squeeze(-1).to(out_dtype)
            batch_wavegrid = batch_wavegrid.to(out_dtype)

            # Poids des raies (du masque): [MB, M_lines]
            line_w = dataset.line_weights.to(batch_yobs.device, dtype=out_dtype)
            batch_line_weights = line_w.unsqueeze(0).expand(MB, -1).contiguous()

            # Proxies d'activité optionnels (si présents dans le dataset)
            batch_device = batch_yobs.device
            batch_activity_proxies_norm = _take_opt(
                dataset,
                "activity_proxies_norm",
                dataset.metadata.get("activity_proxies_included", False),
                batch_indices,
                MB,
                batch_device,
            )

            return (
                batch_yobs,  # [MB, M_lines, W]
                batch_yaug,  # [MB, M_lines, W]
                batch_voffset,  # [MB]
                batch_wavegrid,  # [MB, M_lines, W]
                batch_line_weights,  # [MB, M_lines]
                batch_indices,  # [MB]
                None,  # compat placeholder (activity)
                batch_activity_proxies_norm,  # [MB, P] ou None
            )
        else:
            # Mode spectre complet (fallback)
            batch_yobs = torch.stack(spectra_list, dim=0)  # [B, n_pix]
            B, n_pix = batch_yobs.shape

            if M > 1:
                batch_yobs = (
                    batch_yobs.unsqueeze(1).expand(B, M, n_pix).reshape(-1, n_pix)
                )
            MB = batch_yobs.shape[0]

            batch_indices = torch.as_tensor(
                indices_list, dtype=torch.long, device=batch_yobs.device
            )
            if M > 1:
                batch_indices = batch_indices.repeat_interleave(M)  # [M*B]

            batch_wavegrid = (
                dataset.wavegrid.to(batch_yobs.device, dtype=batch_yobs.dtype)
                .unsqueeze(0)
                .expand(MB, -1)
                .contiguous()
            )

            batch_yaug, batch_voffset = augment_spectra_uniform(
                batch_yobs,
                batch_wavegrid,
                vmin=vmin,
                vmax=vmax,
                interpolate=interpolate,
                extrapolate=extrapolate,
                out_dtype=out_dtype,
            )

            batch_device = batch_yobs.device
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
