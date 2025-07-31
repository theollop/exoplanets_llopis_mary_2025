import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from src.utils import get_free_memory
from src.interpolate import augment_spectra_uniform

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
        batch_size=16,
    ):
        print("Initialisation du SpectrumDataset...")
        self.init_data(
            n_specs=n_specs,
            wavemin=wavemin,
            wavemax=wavemax,
            data_dtype=data_dtype,
            batch_size=batch_size,
        )
        print(self)
        print("Déplacement des données vers le GPU si disponible...")
        print("CUDA disponible:", get_free_memory() / 1e9, "GB")
        self.move_to_cuda()
        print("Dataset initialisé avec succès.")
        print("CUDA disponible:", get_free_memory() / 1e9, "GB")

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
        batch_size=16,
    ):
        filepath = "/home/tliopis/Codes/lagrange_llopis_mary_2025/data/rv_datachallenge/Sun_B57001_E61001_planet-FallChallenge1/HARPN/STAR1136_HPN_flux_YVA.npy"

        analyse_material = np.load(
            "/home/tliopis/Codes/lagrange_llopis_mary_2025/data/rv_datachallenge/Sun_B57001_E61001_planet-FallChallenge1/HARPN/STAR1136_HPN_Analyse_material.p",
            allow_pickle=True,
        )

        wavegrid = analyse_material["wave"].to_numpy()
        template = analyse_material["stellar_template"].to_numpy()

        analyse_summary = pd.read_csv(
            "/home/tliopis/Codes/lagrange_llopis_mary_2025/data/rv_datachallenge/Sun_B57001_E61001_planet-FallChallenge1/HARPN/STAR1136_HPN_Analyse_summary.csv"
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
        data = np.load(filepath)
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
        self.batch_size = batch_size
        self.b_wavegrid = self.wavegrid.expand(batch_size, -1).contiguous()

    def move_to_cuda(self):
        """
        Déplace les données du dataset vers le GPU si disponible.
        """
        if torch.cuda.is_available():
            self.spectra = self.spectra.cuda()
            self.wavegrid = self.wavegrid.cuda()
            self.template = self.template.cuda()
            self.b_wavegrid = self.b_wavegrid.cuda()
        else:
            print("CUDA n'est pas disponible, les données restent sur le CPU.")


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


def get_dataloader(
    n_specs=None,
    wavemin=None,
    wavemax=None,
    data_dtype=torch.float32,
    batch_size=16,
    shuffle=True,
    collate_fn=None,
    vmin=-3,
    vmax=3,
    interpolate="linear",
    extrapolate="linear",
    out_dtype=torch.float32,
):
    """
    Crée un DataLoader pour le dataset donné.

    Args:
        n_specs (int): Nombre de spectres à charger.
        wavemin (float): Longueur d'onde minimale.
        wavemax (float): Longueur d'onde maximale.
        data_dtype (torch.dtype): Type de données des spectres.
        batch_size (int): Taille du batch.
        shuffle (bool): Si True, les données seront mélangées.
        collate_fn (callable): Fonction de collate personnalisée.
        vmin (float): Vitesse minimale pour l'augmentation des spectres.
        vmax (float): Vitesse maximale pour l'augmentation des spectres.
        interpolate (str): Méthode d'interpolation à utiliser.
        extrapolate (str): Méthode d'extrapolation à utiliser.
        out_dtype (torch.dtype): Type de données de sortie des spectres augmentés.

    Returns:
        DataLoader: Un DataLoader configuré pour le dataset.
    """

    dataset = SpectrumDataset(
        n_specs=n_specs,
        wavemin=wavemin,
        wavemax=wavemax,
        data_dtype=data_dtype,
        batch_size=batch_size,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
        or generate_collate_fn(
            dataset,
            vmin=vmin,
            vmax=vmax,
            interpolate=interpolate,
            extrapolate=extrapolate,
            out_dtype=out_dtype,
        ),
    )
