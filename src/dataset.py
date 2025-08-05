import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from .utils import get_free_memory
from .interpolate import augment_spectra_uniform

##############################################################################
##############################################################################
#                           *Dataset et gestion des données*                 #
##############################################################################
##############################################################################


# * -- Classe principale pour charger les spectres du dataset du RV Data Challenge --
# TODO: Ajouter la gestion des autres datasets
class SpectrumDataset(Dataset):
    """
    Dataset pour charger des spectres
    """

    def __init__(
        self,
        n_specs=None,
        wavemin=None,
        wavemax=None,
        data_dtype=torch.float32,
        data_root_dir="data",
    ):
        print("Initialisation du SpectrumDataset...")

        # Stocker le répertoire racine pour la sauvegarde
        self.data_root_dir = data_root_dir

        self.init_data(
            n_specs=n_specs,
            wavemin=wavemin,
            wavemax=wavemax,
            data_dtype=data_dtype,
            data_root_dir=data_root_dir,
        )
        print(self)
        print("Déplacement des données vers le GPU si disponible...")
        print(f"CUDA disponible: {get_free_memory() / 1e9:.3f} GB")
        self.move_to_cuda()
        print("Dataset initialisé avec succès.")
        print(f"CUDA disponible: {get_free_memory() / 1e9:.3f} GB")

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
            dtype_info = " (DOUBLE PRECISION - haute précision)"

        return (
            f"\n======== SpectrumDataset ========\n"
            f"n_specs={self.spectra.shape[0]}, n_pixels={self.spectra.shape[1]}\n"
            f"spectra_shape={self.spectra.shape} | {self.spectra.dtype}{dtype_info}\n"
            f"wavegrid_shape={self.wavegrid.shape} | {self.wavegrid.dtype}\n"
            f"template_shape={self.template.shape} | {self.template.dtype}\n"
            f"jdb_shape={self.jdb.shape} | {self.jdb.dtype})\n"
            f"Memory footprint: ~{self._estimate_memory_usage():.2f} MB\n"
            f"======== End of SpectrumDataset ========\n"
        )

    def init_data(
        self,
        n_specs=None,
        wavemin=None,
        wavemax=None,
        data_dtype=torch.float32,
        data_root_dir="data",
    ):
        dataset_filepath = f"{data_root_dir}/rv_datachallenge/Sun_B57001_E61001_planet-FallChallenge1/HARPN/STAR1136_HPN_flux_YVA.npy"

        analyse_material = np.load(
            f"{data_root_dir}/rv_datachallenge/Sun_B57001_E61001_planet-FallChallenge1/HARPN/STAR1136_HPN_Analyse_material.p",
            allow_pickle=True,
        )

        wavegrid = analyse_material["wave"].to_numpy()
        template = analyse_material["stellar_template"].to_numpy()

        analyse_summary = pd.read_csv(
            f"{data_root_dir}/rv_datachallenge/Sun_B57001_E61001_planet-FallChallenge1/HARPN/STAR1136_HPN_Analyse_summary.csv"
        )

        if wavemin is None:
            wavemin = wavegrid.min()
        if wavemax is None:
            wavemax = wavegrid.max()

        wave_mask = (wavegrid >= wavemin) & (wavegrid <= wavemax)
        wavegrid = wavegrid[wave_mask]
        template = template[wave_mask]

        if n_specs is None:
            n_specs = analyse_summary.shape[0]
        data = np.load(dataset_filepath)
        data = data[:n_specs, wave_mask]

        self.spectra = torch.tensor(data).to(dtype=data_dtype).contiguous()
        self.wavegrid = torch.tensor(wavegrid).to(dtype=data_dtype).contiguous()
        self.template = torch.tensor(template).to(dtype=data_dtype).contiguous()
        self.jdb = (
            torch.tensor(analyse_summary["jdb"][:n_specs])
            .to(dtype=data_dtype)
            .contiguous()
        )
        self.n_specs = n_specs
        self.n_pixels = wavegrid.shape[0]
        self.data_dtype = data_dtype
        self.wavemin = wavemin
        self.wavemax = wavemax

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
        total_memory += tensor_memory_mb(self.jdb)

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
        self.jdb = self.jdb.to(dtype=new_dtype)
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
        Retourne un dict contenant tout ce qu’il faut pour
        recharger le dataset dans les mêmes conditions.
        """
        return {
            "n_specs": self.n_specs,
            "wavemin": float(self.wavemin),
            "wavemax": float(self.wavemax),
            "data_dtype": self.data_dtype,
            "data_root_dir": self.data_root_dir,
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
            .cuda()
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
    dataset = SpectrumDataset()

    dataset_metadata = dataset.to_dict()
    print("Metadata du dataset:", dataset_metadata)
    new_dataset = SpectrumDataset(
        **dataset_metadata,  # Recharger le dataset avec les mêmes paramètres
    )
