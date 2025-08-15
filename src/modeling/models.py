import torch
import torch.nn as nn
from src.interpolate import shift_spectra_linear
import os
import torch.optim as optim
from typing import Optional

##############################################################################
##############################################################################
#                           *Réseaux de neurones*                           #
##############################################################################
##############################################################################


class MLP(nn.Module):
    """
    Réseau Multi-Perceptron classique
    """

    def __init__(
        self,
        n_in,
        n_out,
        n_hidden=(16, 16, 16),
        act=(nn.LeakyReLU(), nn.LeakyReLU(), nn.LeakyReLU(), nn.LeakyReLU()),
        dropout=0,
    ):
        super(MLP, self).__init__()

        n_ = [n_in, *n_hidden, n_out]

        layers = []

        for i in range(0, len(n_) - 1):
            layers.append(nn.Linear(in_features=n_[i], out_features=n_[i + 1]))
            layers.append(act[i])
            layers.append(nn.Dropout(p=dropout))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        conv_ksize,
        conv_stride,
        conv_padding,
        maxpool_ksize,
        maxpool_stride,
        maxpool_padding,
        maxpool_ceil_mode,
        act=nn.LeakyReLU(),
        dropout=0,
    ):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=conv_ksize,
            stride=conv_stride,
            padding=conv_padding,
        )

        self.instancenorm = nn.InstanceNorm1d(num_features=out_channels)

        self.activation = act
        self.dropout = nn.Dropout(p=dropout)

        # Si on veut rajouter une couche maxpool (pas le cas du dernier convblock de spender)
        if (
            (maxpool_ksize is not None)
            and (maxpool_padding is not None)
            and (maxpool_stride is not None)
            and (maxpool_ceil_mode is not None)
        ):
            self.maxpool = nn.MaxPool1d(
                kernel_size=maxpool_ksize,
                stride=maxpool_stride,
                padding=maxpool_padding,
                ceil_mode=maxpool_ceil_mode,
            )
        else:
            self.maxpool = None

    def forward(self, x):
        x = self.conv(x)
        x = self.instancenorm(x)
        x = self.activation(x)

        if self.maxpool is not None:
            x = self.maxpool(x)

        return x


class SPENDER(nn.Module):
    """

    * self.n_pixels_in              -> Nb de pixel du spectre en entrée           dtype=int

    * self.wave_block          -> tensor de taille [B, n_pixels_in]           dtype=float32

    """

    def __init__(self, n_pixels_in, S=3):
        super(SPENDER, self).__init__()

        # ---------- Encoder ----------

        # ConvBlock n°1
        self.convblock1 = ConvBlock(
            in_channels=1,
            out_channels=128,
            conv_ksize=5,
            conv_stride=1,
            conv_padding=2,
            maxpool_ksize=5,
            maxpool_stride=5,
            maxpool_padding=0,
            maxpool_ceil_mode=False,
            act=nn.PReLU(num_parameters=128),
            dropout=0,
        )

        # ConvBlock n°2
        self.convblock2 = ConvBlock(
            in_channels=128,
            out_channels=256,
            conv_ksize=11,
            conv_stride=1,
            conv_padding=5,
            maxpool_ksize=11,
            maxpool_stride=11,
            maxpool_padding=0,
            maxpool_ceil_mode=False,
            act=nn.PReLU(num_parameters=256),
            dropout=0,
        )

        # ConvBlock n°3
        self.convblock3 = ConvBlock(
            in_channels=256,
            out_channels=512,
            conv_ksize=21,
            conv_stride=1,
            conv_padding=10,
            maxpool_ksize=None,
            maxpool_stride=None,
            maxpool_padding=None,
            maxpool_ceil_mode=None,
            act=nn.PReLU(num_parameters=512),
            dropout=0,
        )

        # Softmax du bloc d'attention
        self.softmax = nn.Softmax(dim=-1)

        # MLP pour convertir la sortie de l'attention block en vecteur de l'espace latent
        self.latentMLP = MLP(
            n_in=256,
            n_out=S,
            n_hidden=(128, 64, 32),
            act=(nn.PReLU(128), nn.PReLU(64), nn.PReLU(32), nn.PReLU(S)),
            dropout=0,
        )

        # ---------- Decoder ----------
        self.decoder = MLP(
            n_in=S,
            n_out=n_pixels_in,
            n_hidden=(64, 256, 1024),
            act=(nn.PReLU(64), nn.PReLU(256), nn.PReLU(1024), nn.PReLU(n_pixels_in)),
            dropout=0,
        )

        # self.LSF = nn.Conv1d(1, 1, 5, bias=False, padding='same')

        self.current_latent = None

    def forward(self, x):
        x = x.unsqueeze(1)

        # Encoding
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)

        C = x.shape[1] // 2  # Nombre de canaux
        h, k = torch.split(x, [C, C], dim=1)  # On divise en deux
        a = self.softmax(k)
        e = torch.sum(
            h * a, dim=-1
        )  # On somme selon les longueurs d'ondes -> sortie [B, C]
        s = self.latentMLP(e)  # Vecteur latent

        # Decoding
        x = self.decoder(s)

        x = x.unsqueeze(1)

        # Convolve
        # x = self.LSF(x)

        x = x.squeeze(1)

        return x, s


class RVEstimator(nn.Module):
    """Some Information about RVEstimator"""

    def __init__(self, n_pixels_in, dropout=0):
        super(RVEstimator, self).__init__()

        # ConvBlock n°1
        self.convblock1 = ConvBlock(
            in_channels=1,
            out_channels=128,
            conv_ksize=5,
            conv_stride=1,
            conv_padding=2,
            maxpool_ksize=5,
            maxpool_stride=5,
            maxpool_padding=0,
            maxpool_ceil_mode=False,
            act=nn.PReLU(num_parameters=128),
            dropout=0,
        )

        # ConvBlock n°2
        self.convblock2 = ConvBlock(
            in_channels=128,
            out_channels=64,
            conv_ksize=10,
            conv_stride=1,
            conv_padding=5,
            maxpool_ksize=10,
            maxpool_stride=10,
            maxpool_padding=0,
            maxpool_ceil_mode=False,
            act=nn.PReLU(num_parameters=64),
            dropout=0,
        )

        self.softmax = nn.Softmax(dim=-1)
        self.flatten = nn.Flatten()

        # Calcul dynamique de la taille d'entrée du MLP
        with torch.no_grad():
            dummy = torch.zeros(1, n_pixels_in)
            x = dummy.unsqueeze(1)
            x = self.convblock1(x)
            x = self.convblock2(x)
            x = self.softmax(x)
            x = self.flatten(x)
            n_features_out = x.shape[1]
        self.n_features_out = n_features_out

        self.mlp = MLP(
            n_in=self.n_features_out,
            n_out=1,
            n_hidden=(128, 64, 32),
            act=(nn.PReLU(128), nn.PReLU(64), nn.PReLU(32), nn.Identity()),
            dropout=dropout,
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.softmax(x)
        x = self.flatten(x)
        x = self.mlp(x)

        return x.squeeze(1)


class AESTRA(nn.Module):
    """
    Modèle combiné :
    - phase='rvonly'   → on n’exécute que RVEstimator
    - phase='joint'→ on exécute les deux et renvoie (y_prime, rv_pred, s…)
    """

    def __init__(
        self,
        n_pixels,
        b_obs,
        b_rest,
        S=3,
        sigma_v=1.0,
        sigma_s=1.0,
        sigma_y=1.0,
        k_reg_init=1.0,
        cycle_length=1000,  # Nombre d'itérations pour le cycle de régularisation
        dropout=0.0,
        device="cuda",  # ⚠️ NOUVEAU: Paramètre device explicite
        dtype=torch.float32,  # ⚠️ NOUVEAU: Paramètre dtype pour la précision
    ):
        """
        Args:
            n_pixels (int): Nombre de pixels du spectre d'entrée.
            S (int): Dimension de l'espace latent pour SPENDER.
            dropout (float): Taux de dropout pour les couches MLP.
            b_obs (torch.Tensor): Spectre b_obs de référence pour les observations (b_obs dans l'article). [n_pixels] (tensor non pas un paramètre celui-ci est converti ensuite en paramètre)
            b_rest (torch.Tensor): Spectre b_rest de référence pour les observations (b_rest dans l'article). [n_pixels] (tensor non pas un paramètre celui-ci est converti ensuite en paramètre)
            device (str): Device à utiliser ("cuda" ou "cpu")
            dtype (torch.dtype): Type de données pour les poids du modèle (torch.float16, torch.float32, torch.float64)
        """
        super().__init__()

        # ⚠️ CORRECTION: Création des modules sans forcer .cuda()
        self.device = device
        self.dtype = dtype
        self.spender = SPENDER(n_pixels, S=S)
        self.rvestimator = RVEstimator(n_pixels, dropout=dropout)

        # Déplacement vers le device approprié seulement si CUDA est disponible
        if device == "cuda" and torch.cuda.is_available():
            self.spender = self.spender.cuda()
            self.rvestimator = self.rvestimator.cuda()

        # Conversion vers le dtype spécifié
        self.spender = self.spender.to(dtype=dtype)
        self.rvestimator = self.rvestimator.to(dtype=dtype)

        self.b_obs = nn.Parameter(b_obs.to(dtype=dtype), requires_grad=False)
        self.b_rest = nn.Parameter(b_rest.to(dtype=dtype), requires_grad=True)
        # phase par défaut
        self.phase = "joint"
        self.sigma_v = sigma_v
        self.sigma_s = sigma_s
        self.sigma_y = sigma_y
        self.k_reg_init = k_reg_init
        self.cycle_length = cycle_length

    def set_phase(self, phase: str):
        self.phase = phase

    def set_trainable(
        self,
        b_obs=False,
        b_rest=True,
        rvestimator=True,
        spender=True,
    ):
        """
        Définit si les spectres b_obs et b_rest sont entraînables.(par défaut b_obs non entraînable et b_rest entraînable dans l'article)
        Args:
            b_obs_trainable (bool): Si True, b_obs est entraînable.
            b_rest_trainable (bool): Si True, b_rest est entraînable.
        """
        self.b_obs.requires_grad = b_obs
        self.b_rest.requires_grad = b_rest

        for p in self.rvestimator.parameters():
            p.requires_grad = rvestimator
        for p in self.spender.parameters():
            p.requires_grad = spender

        print(
            f"b_obs trainable: {b_obs}, b_rest trainable: {b_rest}, "
            f"rvestimator trainable: {rvestimator}, spender trainable: {spender}"
        )
        self.rvestimator_trainable = rvestimator
        self.spender_trainable = spender

    def convert_dtype(self, new_dtype):
        """
        Convertit le modèle vers un nouveau type de données.

        Args:
            new_dtype (torch.dtype): Nouveau type de données (torch.float16, torch.float32, torch.float64)
        """
        old_dtype = self.dtype
        self.dtype = new_dtype

        # Conversion des modules
        self.spender = self.spender.to(dtype=new_dtype)
        self.rvestimator = self.rvestimator.to(dtype=new_dtype)

        # Conversion des paramètres b_obs et b_rest
        self.b_obs.data = self.b_obs.data.to(dtype=new_dtype)
        self.b_rest.data = self.b_rest.data.to(dtype=new_dtype)

        print(f"Modèle converti de {old_dtype} vers {new_dtype}")
        return self

    def get_losses(
        self, batch, extrapolate="linear", iteration_count=None, get_aug_data=True
    ):
        """
        Calcule les pertes en fonction de la phase du modèle.

        Args:
            batch: tuple contenant (batch_yobs, batch_yaug, batch_voffset_true, batch_wavegrid, batch_weights_fid, batch_indices)
            extrapolate: méthode d'extrapolation pour le shift Doppler
            batch_weights: poids pour la perte FID (facultatif)
        """
        (
            batch_yobs,
            batch_yaug,
            batch_voffset_true,
            batch_wavegrid,
            batch_weights_fid,
            batch_indices,
        ) = batch
        losses = {
            "fid": torch.tensor(0),
            "c": torch.tensor(0),
            "reg": torch.tensor(0),
            "rv": torch.tensor(0),
        }
        if self.rvestimator_trainable:
            batch_vobs_pred, batch_vaug_pred = self.get_rvestimator_pred(
                batch_yobs=batch_yobs,
                batch_yaug=batch_yaug,
            )

            batch_voffset_pred = batch_vaug_pred - batch_vobs_pred

            losses["rv"] = loss_rv(
                batch_voffset_true=batch_voffset_true,
                batch_voffset_pred=batch_voffset_pred,
                sigma_v=self.sigma_v,
            )

        if self.spender_trainable:
            batch_yobs_prime, batch_yact, _, s, s_aug = self.get_spender_pred(
                batch_yobs=batch_yobs,
                batch_yaug=batch_yaug,
                batch_wavegrid=batch_wavegrid,
                batch_vobs_pred=batch_vobs_pred,
                extrapolate=extrapolate,
                get_aug_data=get_aug_data,
            )

            losses["fid"] = loss_fid(
                batch_yobs_prime=batch_yobs_prime,
                batch_yobs=batch_yobs,
                batch_weights=batch_weights_fid,
            )

            if get_aug_data:
                losses["c"] = loss_c(s, s_aug, sigma_s=self.sigma_s)
            else:
                losses["c"] = torch.tensor(0)

            losses["reg"] = loss_reg(
                batch_yact,
                k_reg_init=self.k_reg_init,
                sigma_y=self.sigma_y,
                iteration_count=iteration_count,
                cycle_length=self.cycle_length,
            )

        return losses

    def get_rvestimator_pred(self, batch_yobs, batch_yaug):
        batch_robs = batch_yobs - self.b_obs.unsqueeze(0)
        batch_raug = batch_yaug - self.b_obs.unsqueeze(0)

        batch_vobs_pred = self.rvestimator(batch_robs)
        batch_vaug_pred = self.rvestimator(batch_raug)

        return batch_vobs_pred, batch_vaug_pred

    def get_spender_pred(
        self,
        batch_yobs,
        batch_yaug,
        batch_wavegrid,
        batch_vobs_pred,
        extrapolate="linear",
        get_aug_data=True,
    ):
        batch_robs = batch_yobs - self.b_obs.unsqueeze(0)

        batch_yact, s = self.spender(batch_robs)

        batch_yrest = self.b_rest.unsqueeze(0) + batch_yact

        batch_yobs_prime = shift_spectra_linear(
            spectra=batch_yrest,
            wavegrid=batch_wavegrid,
            velocities=batch_vobs_pred,
            extrapolate=extrapolate,
        )

        if get_aug_data:
            batch_raug = batch_yaug - self.b_obs.unsqueeze(0)
            batch_yact_aug, s_aug = self.spender(batch_raug)
        else:
            batch_yact_aug, s_aug = None, None

        return batch_yobs_prime, batch_yact, batch_yact_aug, s, s_aug


def loss_rv(batch_voffset_true, batch_voffset_pred, sigma_v=1.0):
    return torch.mean((batch_voffset_true - batch_voffset_pred) ** 2 / (sigma_v**2))


def loss_fid(batch_yobs_prime, batch_yobs, batch_weights=None):
    if batch_weights is None:
        batch_weights = torch.ones_like(batch_yobs_prime)

    return torch.mean(batch_weights * (batch_yobs - batch_yobs_prime) ** 2)


def loss_c(s, s_aug, sigma_s=1.0):
    S = s.shape[1]

    return torch.mean(torch.sigmoid((s - s_aug) ** 2 / (S * sigma_s**2)) - 0.5)


def get_k_reg(k_reg_init: float, iteration_count: int = None, cycle_length: int = 1000):
    """
    Retourne la valeur de k_reg en fonction du nombre d'itérations.
    La valeur de k_reg augmente linéairement de 0 à 1 sur un cycle de cycle_length itérations.
    """
    if iteration_count is None or cycle_length == 0:
        return k_reg_init

    k_reg = (iteration_count % cycle_length) / cycle_length
    return k_reg


def loss_reg(
    batch_yact, k_reg_init, sigma_y=1.0, iteration_count=None, cycle_length=1000
):
    current_k_reg = get_k_reg(k_reg_init, iteration_count, cycle_length)
    return (current_k_reg / sigma_y**2) * torch.mean(batch_yact**2)


def save_checkpoint(
    model: AESTRA,
    optimizer: optim.Optimizer,
    path: str,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
):
    """
    Sauvegarde minimaliste :
     - model_state_dict
     - optimizer_state_dict
     - scheduler_state_dict (optionnel)
     - model.phase
    """
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_phase": model.phase,
    }
    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()
        ckpt["scheduler_class"] = scheduler.__class__.__name__

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ckpt, path)
    print(f"✅ Checkpoint saved to {path}")


def load_checkpoint(
    path: str,
    model: AESTRA,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    device: Optional[str] = None,
) -> Optional[optim.lr_scheduler._LRScheduler]:
    """
    Recharge :
     - les poids dans `model`
     - l’état dans `optimizer`
     - (éventuellement) dans `scheduler`
     - la phase dans `model.phase`
    Nécessite que `model`, `optimizer` (et `scheduler`, si présent dans le ckpt)
    aient déjà été instanciés AVANT l’appel.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(path, map_location=device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.set_phase(ckpt.get("model_phase", "joint"))
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        print(f"✅ Scheduler state restored ({ckpt['scheduler_class']})")
    else:
        scheduler = None

    print(f"✅ Loaded checkpoint from {path}  (phase={model.phase})")
    return scheduler


if __name__ == "__main__":
    pass
