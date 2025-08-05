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

        # Construction des chemins à partir du répertoire racine
        dataset_filepath = f"{data_root_dir}/rv_datachallenge/Sun_B57001_E61001_planet-FallChallenge1/HARPN/STAR1136_HPN_flux_YVA.npy"
        material_filepath = f"{data_root_dir}/rv_datachallenge/Sun_B57001_E61001_planet-FallChallenge1/HARPN/STAR1136_HPN_Analyse_material.p"
        summary_filepath = f"{data_root_dir}/rv_datachallenge/Sun_B57001_E61001_planet-FallChallenge1/HARPN/STAR1136_HPN_Analyse_summary.csv"

        self.init_data(
            n_specs=n_specs,
            wavemin=wavemin,
            wavemax=wavemax,
            data_dtype=data_dtype,
            dataset_filepath=dataset_filepath,
            material_filepath=material_filepath,
            summary_filepath=summary_filepath,
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
        return (
            f"\n======== SpectrumDataset ========\n"
            f"n_specs={self.spectra.shape[0]}, n_pixels={self.spectra.shape[1]}\n"
            f"spectra_shape={self.spectra.shape} | {self.spectra.dtype}\n"
            f"wavegrid_shape={self.wavegrid.shape} | {self.wavegrid.dtype}\n"
            f"template_shape={self.template.shape} | {self.template.dtype}\n"
            f"jdb_shape={self.jdb.shape} | {self.jdb.dtype})\n"
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
