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
        device="cuda",
        dtype=torch.float32,
        smooth_alpha: float = 0.0,  # Poids pour la perte de lissage (L2 sur dérivée)
        smooth_order: int = 1,  # 1 = pente, 2 = courbure
        sigma_l: float = 1.0,  # Poids pour la perte de fidélité
        sigma_corr: float = 1.0,  # Poids pour la perte de corrélation
        include_activity_proxies: bool = False,  # Inclure les proxies d'activité
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
        # Poids de lissage pour y_act (L2 sur la 1ère dérivée)
        self.smooth_alpha = float(smooth_alpha)
        self.smooth_order = int(smooth_order)

        self.sigma_l = sigma_l
        self.sigma_corr = sigma_corr

        self.include_activity_proxies = include_activity_proxies

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
        self,
        batch,
        extrapolate="linear",
        iteration_count=None,
        get_aug_data=True,
    ):
        """
        Calcule les pertes en fonction de la phase du modèle.

        Args:
            batch: tuple contenant (batch_yobs, batch_yaug, batch_voffset_true, batch_wavegrid, batch_weights_fid, batch_indices, batch_yact_true, batch_activity_proxies_norm)
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
            batch_yact_true,
            batch_activity_proxies_norm,
        ) = batch
        losses = {
            "fid": torch.tensor(0),
            "c": torch.tensor(0),
            "reg": torch.tensor(0),
            "rv": torch.tensor(0),
            "smooth": torch.tensor(0),
            "corr": torch.tensor(0),
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
                batch_activity_proxies_norm=batch_activity_proxies_norm,
                include_activity_proxies=self.include_activity_proxies,
            )

            losses["fid"] = loss_fid(
                batch_yobs_prime=batch_yobs_prime,
                batch_yobs=batch_yobs,
                batch_weights=batch_weights_fid,
                sigma_l=self.sigma_l,
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

            # Smoothness regularization on decoder activity (L2 on first derivative)
            if self.smooth_alpha is not None and float(self.smooth_alpha) > 0.0:
                losses["smooth"] = loss_smooth(
                    batch_yact, alpha=self.smooth_alpha, order=self.smooth_order
                )
            else:
                losses["smooth"] = torch.tensor(0)

        if (
            self.sigma_corr > 0.0
            and self.rvestimator_trainable
            and self.spender_trainable
        ):
            losses["corr"] = self.sigma_corr * corr_loss_pairs(
                v_obs=batch_vobs_pred,
                v_aug=batch_vaug_pred,
                v_offset=batch_voffset_true,
                S_obs=s,
                S_aug=s_aug,
                use_avg_S=True,
                stopgrad_S=True,
                eps=1e-8,
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
        include_activity_proxies=False,
        batch_activity_proxies_norm=None,
    ):
        batch_robs = batch_yobs - self.b_obs.unsqueeze(0)

        # Optionally concatenate activity proxies (ensure proper shape/device/dtype)
        if include_activity_proxies and batch_activity_proxies_norm is not None:
            if batch_activity_proxies_norm.dim() == 1:
                batch_activity_proxies_norm = batch_activity_proxies_norm.unsqueeze(
                    0
                ).expand(batch_robs.size(0), -1)
            if batch_activity_proxies_norm.size(0) != batch_robs.size(0):
                # Try to expand a single-row tensor; otherwise, skip to avoid crash
                if batch_activity_proxies_norm.size(0) == 1:
                    batch_activity_proxies_norm = batch_activity_proxies_norm.expand(
                        batch_robs.size(0), -1
                    )
                else:
                    batch_activity_proxies_norm = None
            if batch_activity_proxies_norm is not None:
                batch_activity_proxies_norm = batch_activity_proxies_norm.to(
                    device=batch_robs.device, dtype=batch_robs.dtype
                )
                batch_robs = torch.cat([batch_robs, batch_activity_proxies_norm], dim=1)

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
            if include_activity_proxies and batch_activity_proxies_norm is not None:
                if batch_activity_proxies_norm.dim() == 1:
                    batch_activity_proxies_norm = batch_activity_proxies_norm.unsqueeze(
                        0
                    ).expand(batch_raug.size(0), -1)
                if batch_activity_proxies_norm.size(0) == 1:
                    batch_activity_proxies_norm = batch_activity_proxies_norm.expand(
                        batch_raug.size(0), -1
                    )
                if batch_activity_proxies_norm.size(0) == batch_raug.size(0):
                    batch_activity_proxies_norm = batch_activity_proxies_norm.to(
                        device=batch_raug.device, dtype=batch_raug.dtype
                    )
                    batch_raug = torch.cat(
                        [batch_raug, batch_activity_proxies_norm], dim=1
                    )
            batch_yact_aug, s_aug = self.spender(batch_raug)
        else:
            batch_yact_aug, s_aug = None, None

        return batch_yobs_prime, batch_yact, batch_yact_aug, s, s_aug


class AESTRAM(nn.Module):
    """
    AESTRA masked (par raies): même structure conceptuelle que AESTRA mais entrée aplatie
    provenant d'un tenseur [B, L, W+1] (optionnellement [B, L+3, W+1] si 3 proxies d'activité).

    Convention d'encodage par "ligne" (row):
      - Chaque "row" fait W+1 colonnes: [flux sur W pixels | pos_norm (1)]
      - Les dernières 3 rows (si include_activity_proxies=True) contiennent les proxies activité;
        elles sont ignorées pour le shift/L_fid (mais visibles des réseaux).

    Le modèle opère sur le vecteur aplati de taille n_in_flat = (L + P) * (W+1),
    où P=3 si proxies inclus, 0 sinon. Pour calculer L_fid, on recompose [B, L, W]
    depuis la sortie du décodeur (y_rest_flat), puis on applique le shift Doppler
    par raie en utilisant la wavegrid des fenêtres.
    """

    def __init__(
        self,
        n_lines: int,
        window_size: int,
        b_obs_flat: torch.Tensor,
        b_rest_flat: torch.Tensor,
        include_activity_proxies: bool = False,
        dropout: float = 0.0,
        S: int = 3,
        sigma_v: float = 1.0,
        sigma_s: float = 1.0,
        sigma_y: float = 1.0,
        sigma_l: float = 1.0,
        sigma_corr: float = 1.0,
        k_reg_init: float = 1.0,
        cycle_length: int = 1000,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.L = int(n_lines)
        self.W = int(window_size)
        self.include_activity_proxies = bool(include_activity_proxies)
        self.P = 3 if self.include_activity_proxies else 0
        self.row_len = self.W + 1  # flux(W) + pos_norm(1)
        self.n_in_flat = (self.L + self.P) * self.row_len

        assert (
            b_obs_flat.numel() == self.n_in_flat
            and b_rest_flat.numel() == self.n_in_flat
        ), "b_obs_flat et b_rest_flat doivent matcher (L+P)*(W+1)"

        self.device = device
        self.dtype = dtype

        # Modules partagés avec AESTRA
        self.spender = SPENDER(self.n_in_flat, S=S)
        self.rvestimator = RVEstimator(self.n_in_flat, dropout=dropout)

        if device == "cuda" and torch.cuda.is_available():
            self.spender = self.spender.cuda()
            self.rvestimator = self.rvestimator.cuda()

        self.spender = self.spender.to(dtype=dtype)
        self.rvestimator = self.rvestimator.to(dtype=dtype)

        self.b_obs = nn.Parameter(b_obs_flat.to(dtype=dtype), requires_grad=False)
        self.b_rest = nn.Parameter(b_rest_flat.to(dtype=dtype), requires_grad=True)

        # Hyperparamètres pertes
        self.sigma_v = sigma_v
        self.sigma_s = sigma_s
        self.sigma_y = sigma_y
        self.sigma_l = sigma_l
        self.sigma_corr = sigma_corr
        self.k_reg_init = k_reg_init
        self.cycle_length = cycle_length

        # Phases/flags d'entraînement
        self.phase = "joint"
        self.rvestimator_trainable = True
        self.spender_trainable = True

    def set_phase(self, phase: str):
        self.phase = phase

    def set_trainable(self, b_obs=False, b_rest=True, rvestimator=True, spender=True):
        self.b_obs.requires_grad = b_obs
        self.b_rest.requires_grad = b_rest
        for p in self.rvestimator.parameters():
            p.requires_grad = rvestimator
        for p in self.spender.parameters():
            p.requires_grad = spender
        self.rvestimator_trainable = rvestimator
        self.spender_trainable = spender

    def convert_dtype(self, new_dtype):
        self.dtype = new_dtype
        self.spender = self.spender.to(dtype=new_dtype)
        self.rvestimator = self.rvestimator.to(dtype=new_dtype)
        self.b_obs.data = self.b_obs.data.to(dtype=new_dtype)
        self.b_rest.data = self.b_rest.data.to(dtype=new_dtype)
        return self

    # Helpers de (dé)composition
    def _view_rows(self, x_flat: torch.Tensor) -> torch.Tensor:
        # [B, (L+P)*(W+1)] -> [B, L+P, W+1]
        B = x_flat.size(0)
        return x_flat.view(B, self.L + self.P, self.row_len)

    def _slice_lines_flux(self, x_rows: torch.Tensor) -> torch.Tensor:
        # x_rows: [B, L+P, W+1] -> flux lignes [B, L, W]
        return x_rows[:, : self.L, : self.W]

    def _expand_weights_pix(self, line_w: torch.Tensor) -> torch.Tensor:
        # line_w: [B, L] -> [B, L, W]
        return line_w.unsqueeze(-1).expand(-1, -1, self.W)

    def get_rvestimator_pred(self, xobs_flat: torch.Tensor, xaug_flat: torch.Tensor):
        robs = xobs_flat - self.b_obs.unsqueeze(0)
        raug = xaug_flat - self.b_obs.unsqueeze(0)
        vobs = self.rvestimator(robs)
        vaug = self.rvestimator(raug)
        return vobs, vaug

    def get_spender_pred(
        self,
        xobs_flat: torch.Tensor,
        xaug_flat: Optional[torch.Tensor],
        wavegrid_lines: torch.Tensor,  # [B, L, W]
        vobs_pred: torch.Tensor,  # [B]
        extrapolate: str = "linear",
        get_aug_data: bool = True,
    ):
        # Obs branch
        robs = xobs_flat - self.b_obs.unsqueeze(0)
        yact_obs, s_obs = self.spender(robs)
        yrest_obs = self.b_rest.unsqueeze(0) + yact_obs  # [B, n_in_flat]

        yrest_rows = self._view_rows(yrest_obs)
        yrest_flux = self._slice_lines_flux(yrest_rows)  # [B, L, W]

        # Shift par raie
        B, L, W = yrest_flux.shape
        y_flat = yrest_flux.reshape(B * L, W)
        w_flat = wavegrid_lines.reshape(B * L, W)
        v_flat = vobs_pred.view(-1, 1).repeat_interleave(L, dim=0)
        yobs_prime_flat = shift_spectra_linear(
            spectra=y_flat, wavegrid=w_flat, velocities=v_flat, extrapolate=extrapolate
        )
        yobs_prime = yobs_prime_flat.view(B, L, W)

        if get_aug_data and xaug_flat is not None:
            raug = xaug_flat - self.b_obs.unsqueeze(0)
            yact_aug, s_aug = self.spender(raug)
        else:
            yact_aug, s_aug = None, None

        return yobs_prime, yact_obs, yact_aug, s_obs, s_aug

    def get_losses(
        self,
        batch,
        xobs_flat: torch.Tensor,
        xaug_flat: torch.Tensor,
        extrapolate: str = "linear",
        iteration_count: Optional[int] = None,
        get_aug_data: bool = True,
    ):
        """
        Args:
          batch: tuple renvoyé par le collate "lignes":
            (yobs_lines[B,L,W], yaug_lines[B,L,W], voffset[B], wavegrid_lines[B,L,W], line_weights[B,L], indices, _, _)
          xobs_flat: [B, (L+P)*(W+1)] vecteur aplati d'entrée pour le modèle
          xaug_flat: idem pour la version augmentée
        """
        (
            yobs_lines,
            yaug_lines,
            voffset_true,
            wavegrid_lines,
            line_weights,
            _indices,
            _a,
            _p,
        ) = batch

        losses = {
            k: yobs_lines.new_tensor(0.0)
            for k in ["fid", "c", "reg", "rv", "smooth", "corr"]
        }

        # RV
        if self.rvestimator_trainable:
            vobs_pred, vaug_pred = self.get_rvestimator_pred(xobs_flat, xaug_flat)
            voffset_pred = vaug_pred - vobs_pred
            losses["rv"] = loss_rv(voffset_true, voffset_pred, sigma_v=self.sigma_v)
        else:
            # Dummy pour la suite
            vobs_pred = yobs_lines.new_zeros(yobs_lines.size(0))

        # SPENDER / yprime & pertes
        if self.spender_trainable:
            yobs_prime, yact, yact_aug, s, s_aug = self.get_spender_pred(
                xobs_flat=xobs_flat,
                xaug_flat=xaug_flat if get_aug_data else None,
                wavegrid_lines=wavegrid_lines,
                vobs_pred=vobs_pred,
                extrapolate=extrapolate,
                get_aug_data=get_aug_data,
            )

            # L_fid pondérée par les poids de raies
            w_pix = self._expand_weights_pix(line_weights)  # [B,L,W]
            diff = yobs_lines - yobs_prime
            losses["fid"] = self.sigma_l * torch.mean(w_pix * diff * diff)

            # s-contrast
            if get_aug_data and (s is not None) and (yact_aug is not None):
                losses["c"] = loss_c(s, s_aug, sigma_s=self.sigma_s)
            else:
                losses["c"] = yobs_lines.new_tensor(0.0)

            # Régularisation activité (sur yact à plat)
            losses["reg"] = loss_reg(
                yact,
                k_reg_init=self.k_reg_init,
                sigma_y=self.sigma_y,
                iteration_count=iteration_count,
                cycle_length=self.cycle_length,
            )
        else:
            s = s_aug = None

        # Corrélation Δv vs S
        if (
            self.sigma_corr > 0
            and self.rvestimator_trainable
            and self.spender_trainable
            and (s is not None)
            and (s_aug is not None)
        ):
            losses["corr"] = self.sigma_corr * corr_loss_pairs(
                v_obs=vobs_pred,
                v_aug=vobs_pred + voffset_true,  # approx si pas dispo
                v_offset=voffset_true,
                S_obs=s,
                S_aug=s_aug,
                use_avg_S=True,
                stopgrad_S=True,
            )

        return losses


def loss_rv(batch_voffset_true, batch_voffset_pred, sigma_v=1.0):
    return torch.mean((batch_voffset_true - batch_voffset_pred) ** 2 / (sigma_v**2))


def loss_fid(batch_yobs_prime, batch_yobs, batch_weights=None, sigma_l=1.0):
    if batch_weights is None:
        batch_weights = torch.ones_like(batch_yobs_prime)

    return sigma_l * torch.mean(batch_weights * (batch_yobs - batch_yobs_prime) ** 2)


def loss_c(s, s_aug, sigma_s=1.0):
    S = s.shape[1]

    return torch.mean(torch.sigmoid((s - s_aug) ** 2 / (S * sigma_s**2)) - 0.5)


def loss_smooth(batch_yact, alpha=1.0, order=1, weight=None):
    """
    Smoothness penalty: L2 on the n-th derivative along the wavelength axis.

    Args:
        batch_yact: tensor shape [M*B, P]
        alpha: multiplicative weight
        order: derivative order (1 = slope, 2 = curvature)
        weight: optional tensor [P] or [M*B, P] to weight the penalty
    Returns:
        scalar tensor
    """
    if batch_yact is None:
        return batch_yact.new_tensor(0.0)

    if order == 1:
        d = batch_yact[:, 1:] - batch_yact[:, :-1]
    elif order == 2:
        d = batch_yact[:, 2:] - 2 * batch_yact[:, 1:-1] + batch_yact[:, :-2]
    else:
        raise ValueError("order must be 1 or 2")

    if weight is not None:
        if order > 1:
            weight = weight[:, 1:-1]  # aligne la taille
        else:
            weight = weight[:, :-1]
        d = d * weight

    return alpha * torch.mean(d**2)


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


def _zscore(x, dim=0, eps=1e-8):
    x = x - x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, unbiased=False, keepdim=True).clamp_min(eps)
    return x / std


def corr_loss_v_vs_S(
    v: torch.Tensor,  # (B,)
    S: torch.Tensor,  # (B, Sdim)
    stopgrad_S: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    L_corr = mean_k rho(v, S[:,k])^2  (batch-wise)
    Conseillé: stopgrad_S=True pour cibler la tête RV.
    """
    if v.ndim != 1:
        v = v.view(-1)
    assert S.shape[0] == v.shape[0], "Batch mismatch v/S"
    B = v.shape[0]
    if B < 4:
        return v.new_zeros(())
    if stopgrad_S:
        S = S.detach()

    v_n = _zscore(v, dim=0, eps=eps)  # (B,)
    S_n = _zscore(S, dim=0, eps=eps)  # (B, Sdim)
    # rho_k = mean_b [ v_n[b] * S_n[b,k] ]
    rho = (v_n.unsqueeze(1) * S_n).mean(dim=0)  # (Sdim,)
    return (rho**2).mean()  # scalaire


def corr_loss_pairs(
    v_obs: torch.Tensor,  # (B,)
    v_aug: torch.Tensor,  # (B,)
    v_offset: torch.Tensor,  # (B,)
    S_obs: torch.Tensor,  # (B, Sdim)
    S_aug: torch.Tensor,  # (B, Sdim) (doit être proche de S_obs si encodeur invariant)
    use_avg_S: bool = True,
    stopgrad_S: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    L_corr_pairs = mean_k rho(Δv, S[:,k])^2
    Δv = (v_aug - v_obs - v_offset)
    S = (S_obs + S_aug)/2 par défaut (plus robuste si légère non-invariance).
    """
    dv = v_aug - v_obs - v_offset
    S = 0.5 * (S_obs + S_aug) if use_avg_S else S_obs
    return corr_loss_v_vs_S(dv, S, stopgrad_S=stopgrad_S, eps=eps)


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
