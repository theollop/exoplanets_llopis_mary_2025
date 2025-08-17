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
    SPENDER avec conditionnement par proxies d'activité au niveau latent.

    Args:
        n_pixels_in (int): taille du spectre en entrée/sortie.
        S (int): dimension de l'espace latent.
        proxies_dim (int): nb de proxies d'activité fournis (ex: 3 pour FWHM, depth, BIS).
        proxies_proj_dim (int): dimension de projection des proxies en mode 'concat'.
        conditioning (str): 'concat' (concaténation sur e) ou 'film' (FiLM γ, β sur e).

    Notes:
        - On ne concatène PAS les proxies aux pixels d'entrée; ils conditionnent le goulot latent.
        - e ∈ R^{256} (attention); en 'concat', e devient [256 + proxies_proj_dim].
        - En 'film', on applique e' = (1+γ)*e + β, avec γ, β prédit depuis les proxies.
    """

    def __init__(
        self,
        n_pixels_in: int,
        S: int = 3,
        proxies_dim: int = 0,
        proxies_proj_dim: int = 32,
        conditioning: str = "concat",  # "concat" ou "film"
    ):
        super(SPENDER, self).__init__()

        assert conditioning in ("concat", "film"), (
            "conditioning must be 'concat' or 'film'"
        )

        self.S = int(S)
        self.proxies_dim = int(proxies_dim)
        self.proxies_proj_dim = int(proxies_proj_dim)
        self.conditioning = conditioning

        # ---------- Encoder (identique à ta version) ----------
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

        # Softmax du bloc d'attention (sur la dimension longueur d'onde)
        self.softmax = nn.Softmax(dim=-1)

        # ---------- Conditioning sur e ----------
        latent_in = 256  # e est de taille 256 (après split h/k et attention)

        # (a) Mode CONCAT : projeter les proxies puis concaténer à e
        if self.proxies_dim > 0 and self.conditioning == "concat":
            self.proj_proxies = nn.Sequential(
                nn.LayerNorm(self.proxies_dim),
                nn.Linear(self.proxies_dim, self.proxies_proj_dim),
                nn.PReLU(self.proxies_proj_dim),
            )
            latent_in = 256 + self.proxies_proj_dim
        else:
            self.proj_proxies = None

        # (b) Mode FiLM : prédire (γ, β) de taille 256 depuis les proxies
        if self.proxies_dim > 0 and self.conditioning == "film":
            self.film_gamma = nn.Sequential(
                nn.LayerNorm(self.proxies_dim),
                nn.Linear(self.proxies_dim, 256),
                nn.Tanh(),  # borne γ ∈ (-1,1)
            )
            self.film_beta = nn.Sequential(
                nn.LayerNorm(self.proxies_dim),
                nn.Linear(self.proxies_dim, 256),
            )
        else:
            self.film_gamma, self.film_beta = None, None

        # ---------- MLP latent ----------
        self.latentMLP = MLP(
            n_in=latent_in,
            n_out=self.S,
            n_hidden=(128, 64, 32),
            act=(nn.PReLU(128), nn.PReLU(64), nn.PReLU(32), nn.PReLU(self.S)),
            dropout=0,
        )

        # ---------- Decoder ----------
        self.decoder = MLP(
            n_in=self.S,
            n_out=n_pixels_in,
            n_hidden=(64, 256, 1024),
            act=(nn.PReLU(64), nn.PReLU(256), nn.PReLU(1024), nn.PReLU(n_pixels_in)),
            dropout=0,
        )

        self.current_latent = None  # pour debug/visualisation

    def _prepare_proxies(
        self, proxies: torch.Tensor, B: int, device, dtype
    ) -> torch.Tensor:
        """Met en forme les proxies pour correspondre au batch (B, P)."""
        if proxies is None or self.proxies_dim == 0:
            return None
        if proxies.ndim == 1:
            proxies = proxies.unsqueeze(0).expand(B, -1)
        elif proxies.size(0) == 1 and B > 1:
            proxies = proxies.expand(B, -1)
        assert proxies.size(0) == B and proxies.size(1) == self.proxies_dim, (
            f"Proxies doivent être de forme [B, {self.proxies_dim}]"
        )
        return proxies.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, proxies: torch.Tensor = None):
        """
        Args:
            x: [B, P] spectre (résiduel) en entrée.
            proxies: [B, proxies_dim] ou [proxies_dim], facultatif.
        Returns:
            yact: [B, P] activité reconstruite (rest-frame)
            s:    [B, S] latent d'activité
        """
        # Encoder
        x = x.unsqueeze(1)  # [B,1,P]
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)  # [B,512,L']

        # Attention: split en (h, k), softmax sur k puis pondération de h
        C = x.shape[1] // 2  # 256
        h, k = torch.split(x, [C, C], dim=1)  # [B,256,L'], [B,256,L']
        a = self.softmax(k)  # poids sur L'
        e = torch.sum(h * a, dim=-1)  # [B,256]

        # Préparer proxies
        B = e.size(0)
        proxies = self._prepare_proxies(proxies, B, device=e.device, dtype=e.dtype)

        # Conditioning sur e
        if self.proxies_dim > 0 and proxies is not None:
            if self.conditioning == "concat":
                p = (
                    self.proj_proxies(proxies)
                    if self.proj_proxies is not None
                    else proxies
                )
                e = torch.cat([e, p], dim=1)  # [B, 256 + proj]
            elif self.conditioning == "film":
                gamma = self.film_gamma(proxies)  # [B,256]
                beta = self.film_beta(proxies)  # [B,256]
                e = (1.0 + gamma) * e + beta

        # Latent + Décoder
        s = self.latentMLP(e)  # [B,S]
        yact = self.decoder(s)  # [B,P]

        self.current_latent = s
        return yact, s


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
        b_rest_equal_b_obs=False,
        b_rest_true=None,
        loss_activity=False,
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
        sigma_corr: float = 0.0,  # Poids pour la perte de corrélation
        include_activity_proxies: bool = False,  # Inclure les proxies d'activité
        activity_proxies_dim: int = 0,
        proxies_proj_dim: int = 32,
        conditioning_mode: str = "concat",
        alpha_act: float = 1.0,
        beta_brest: float = 1.0,
        consistency_mode: str = "mse",
    ):
        """
        Args:
            n_pixels (int): Nombre de pixels du spectre d'entrée.
            S (int): Dimension de l'espace latent pour SPENDER.
            dropout (float): Taux de dropout pour les couches MLP.
            b_obs (torch.Tensor): Spectre b_obs de référence pour les observations (b_obs dans l'article). [n_pixels]
            b_rest (torch.Tensor): Spectre b_rest de référence pour les observations (b_rest dans l'article). [n_pixels]
            device (str): Device à utiliser ("cuda" ou "cpu")
            dtype (torch.dtype): Type de données pour les poids du modèle (torch.float16, torch.float32, torch.float64)
        """
        super().__init__()

        self.device = device
        self.dtype = dtype

        # phase par défaut et hyperparamètres
        self.phase = "joint"
        self.sigma_v = sigma_v
        self.sigma_s = sigma_s
        self.sigma_y = sigma_y
        self.k_reg_init = k_reg_init
        self.cycle_length = cycle_length
        self.smooth_alpha = float(smooth_alpha)
        self.smooth_order = int(smooth_order)
        self.sigma_l = sigma_l
        self.sigma_corr = sigma_corr
        self.include_activity_proxies = include_activity_proxies
        self.activity_proxies_dim = int(activity_proxies_dim)
        self.proxies_proj_dim = int(proxies_proj_dim)
        self.conditioning_mode = conditioning_mode

        self.loss_activity = bool(loss_activity)

        self.alpha_act = alpha_act
        self.beta_brest = beta_brest
        self.consistency_mode = consistency_mode

        self.spender = SPENDER(
            n_pixels_in=n_pixels,
            S=S,
            proxies_dim=(
                self.activity_proxies_dim if self.include_activity_proxies else 0
            ),
            proxies_proj_dim=self.proxies_proj_dim,
            conditioning=self.conditioning_mode,
        )
        self.rvestimator = RVEstimator(n_pixels, dropout=dropout)
        self.b_rest_equal_b_obs = bool(b_rest_equal_b_obs)

        self.b_obs = nn.Parameter(b_obs.to(dtype=dtype), requires_grad=False)
        self.b_rest = nn.Parameter(b_rest.to(dtype=dtype), requires_grad=True)
        self.b_rest_true = b_rest_true
        if self.b_rest_true is not None:
            self.b_rest_true = self.b_rest_true.to(dtype=dtype, device=device)
        if self.b_rest_equal_b_obs:
            self.b_rest = self.b_obs

        # Déplacement vers le device approprié seulement si CUDA est disponible
        if device == "cuda" and torch.cuda.is_available():
            self.spender = self.spender.cuda()
            self.rvestimator = self.rvestimator.cuda()

        # Conversion vers le dtype spécifié
        self.spender = self.spender.to(dtype=dtype)
        self.rvestimator = self.rvestimator.to(dtype=dtype)

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

        device, dtype = batch_yobs.device, batch_yobs.dtype
        zeros = lambda: torch.zeros((), device=device, dtype=dtype)

        losses = {
            k: zeros()
            for k in ["fid", "c", "reg", "rv", "smooth", "corr", "activity", "template"]
        }

        # --- RV head ---
        batch_vobs_pred = None
        batch_vaug_pred = None
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
        else:
            # sécurité si la tête RV est gelée
            B = batch_yobs.size(0)
            batch_vobs_pred = torch.zeros(B, device=device, dtype=dtype)

        # --- SPENDER / reconstruction ---
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

            # Consistency sur les latents (dépend de get_aug_data)
            if get_aug_data and s is not None and s_aug is not None:
                if self.consistency_mode == "mse":
                    losses["c"] = loss_c_mse(s, s_aug, sigma_s=self.sigma_s)
                elif self.consistency_mode == "sigmoid":
                    losses["c"] = loss_c(s, s_aug, sigma_s=self.sigma_s)

            # --- Ces pertes DOIVENT être calculées indépendamment de get_aug_data ---
            # Régularisation L2 sur y_act
            losses["reg"] = loss_reg(
                batch_yact,
                k_reg_init=self.k_reg_init,
                sigma_y=self.sigma_y,
                iteration_count=iteration_count,
                cycle_length=self.cycle_length,
            )

            # Lissage éventuel de y_act
            if self.smooth_alpha is not None and float(self.smooth_alpha) > 0.0:
                losses["smooth"] = loss_smooth(
                    batch_yact, alpha=self.smooth_alpha, order=self.smooth_order
                )

            # Supervision b_rest (template)
            if self.b_rest_true is not None:
                losses["template"] = loss_b_rest(
                    b_rest_true=self.b_rest_true,
                    b_rest_pred=self.b_rest,
                    beta_brest=self.beta_brest,
                )

            # Supervision activité y_act (si GT dispo)
            if batch_yact_true is not None and self.loss_activity:
                losses["activity"] = loss_activity(
                    batch_yact=batch_yact,
                    batch_yact_true=batch_yact_true,
                    alpha_act=self.alpha_act,
                )

        # Corrélation latents / RV (optionnelle)
        if (
            self.sigma_corr > 0.0
            and self.rvestimator_trainable
            and self.spender_trainable
            and get_aug_data
            and s is not None
            and s_aug is not None
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

        proxies_obs = None
        if include_activity_proxies and batch_activity_proxies_norm is not None:
            if batch_activity_proxies_norm.ndim == 1:
                proxies_obs = batch_activity_proxies_norm.unsqueeze(0).expand(
                    batch_robs.size(0), -1
                )
            else:
                proxies_obs = batch_activity_proxies_norm
            proxies_obs = proxies_obs.to(
                device=batch_robs.device, dtype=batch_robs.dtype
            )

        batch_yact, s = self.spender(batch_robs, proxies=proxies_obs)

        base_rest = self.b_obs if self.b_rest_equal_b_obs else self.b_rest
        batch_yrest = base_rest.unsqueeze(0) + batch_yact

        batch_yobs_prime = shift_spectra_linear(
            spectra=batch_yrest,
            wavegrid=batch_wavegrid,
            velocities=batch_vobs_pred,
            extrapolate=extrapolate,
        )

        if get_aug_data:
            batch_raug = batch_yaug - self.b_obs.unsqueeze(0)
            proxies_aug = None
            if include_activity_proxies and batch_activity_proxies_norm is not None:
                if batch_activity_proxies_norm.ndim == 1:
                    proxies_aug = batch_activity_proxies_norm.unsqueeze(0).expand(
                        batch_raug.size(0), -1
                    )
                else:
                    proxies_aug = batch_activity_proxies_norm
                proxies_aug = proxies_aug.to(
                    device=batch_raug.device, dtype=batch_raug.dtype
                )

            batch_yact_aug, s_aug = self.spender(batch_raug, proxies=proxies_aug)
        else:
            batch_yact_aug, s_aug = None, None

        return batch_yobs_prime, batch_yact, batch_yact_aug, s, s_aug


def loss_rv(batch_voffset_true, batch_voffset_pred, sigma_v=1.0):
    return torch.mean((batch_voffset_true - batch_voffset_pred) ** 2 / (sigma_v**2))


def loss_fid(batch_yobs_prime, batch_yobs, batch_weights=None, sigma_l=1.0):
    if batch_weights is None:
        batch_weights = torch.ones_like(batch_yobs_prime)

    return sigma_l * torch.mean(batch_weights * (batch_yobs - batch_yobs_prime) ** 2)


def loss_c(s, s_aug, sigma_s=1.0):
    S = s.shape[1]

    return torch.mean(torch.sigmoid((s - s_aug) ** 2 / (S * sigma_s**2)) - 0.5)


def loss_c_mse(s, s_aug, sigma_s=1.0):
    # s, s_aug: [B, S]
    S = s.shape[1]
    return torch.mean(((s - s_aug) ** 2) / (S * (sigma_s**2)))


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

    k_reg = k_reg_init + (iteration_count % cycle_length) / cycle_length
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


def loss_activity(batch_yact, batch_yact_true, alpha_act=1.0):
    """
    Activity loss: L2 between the original and augmented spectra.

    Args:
        batch_yact: tensor shape [M*B, P]
        batch_yaug: tensor shape [M*B, P]
        alpha: multiplicative weight

    Returns:
        scalar tensor
    """
    if batch_yact is None or batch_yact_true is None:
        return batch_yact.new_tensor(0.0)

    return alpha_act * torch.mean((batch_yact - batch_yact_true) ** 2)


def loss_b_rest(b_rest_true, b_rest_pred, beta_brest=1.0):
    if b_rest_true is None:
        return b_rest_true.new_tensor(0.0)

    return beta_brest * torch.mean((b_rest_true - b_rest_pred) ** 2)


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
