import torch
import numpy as np
from torchcubicspline import natural_cubic_spline_coeffs
from typing import Literal

##############################################################################
##############################################################################
#          *Fonctions d'interpolations et de shift doppler (crucial)*        #
##############################################################################
##############################################################################


def cubic_evaluate(coeffs, tnew):
    t = coeffs[0]
    a, b, c, d = [item.squeeze(-1) for item in coeffs[1:]]
    maxlen = b.size(-1) - 1
    index = torch.bucketize(tnew, t) - 1
    index = index.clamp(
        0, maxlen
    )  # clamp because t may go outside of [t[0], t[-1]]; this is fine
    # will never access the last element of self._t; this is correct behaviour
    fractional_part = tnew - t[index]

    batch_size, spec_size = tnew.shape
    batch_ind = torch.arange(batch_size, device=tnew.device)
    batch_ind = batch_ind.repeat((spec_size, 1)).T

    inner = c[batch_ind, index] + d[batch_ind, index] * fractional_part
    inner = b[batch_ind, index] + inner * fractional_part
    return a[batch_ind, index] + inner * fractional_part


def shift_spectra_cubic(
    spectra: torch.Tensor,  # (B, n_pixel)
    wavegrid: torch.Tensor,  # (B, n_pixel)
    velocities: torch.Tensor,  # (B,)
    return_mask: bool = False,
):
    """
    Décale les spectres par spline cubique et renvoie un mask des valeurs extrapolées.

    Returns
    -------
    out : torch.Tensor, shape (B, n_pixel)
        Spectres interpolés.
    extrap_mask : torch.BoolTensor, shape (B, n_pixel), optional
        True là où shifted est hors de [wavegrid.min(), wavegrid.max()].
    """
    c = 299_792_458.0
    B, N = spectra.shape

    # 1) Calcul du shifted (B, n_pixel)
    vel = velocities.view(-1, 1)
    doppler = torch.sqrt((1 - vel / c) / (1 + vel / c))
    shifted = wavegrid * doppler  # broadcast → (B, N)

    # 2) Construction du masque d'extrapolation
    #    on compare shifted (B,N) avec les bornes de wavegrid (scalaire)
    low = shifted < wavegrid[:, :1]  # (B,1) broadcasté
    high = shifted > wavegrid[:, -1:]  # (B,1) broadcasté
    extrap_mask = low | high  # True si extrapolé

    # 3) Calcul des coefficients de la spline cubique (naturelle)
    coeffs = natural_cubic_spline_coeffs(wavegrid[0], spectra.unsqueeze(-1))

    # 4) Évaluation cubique
    out = cubic_evaluate(coeffs, shifted)

    if return_mask:
        return out, extrap_mask
    else:
        return out


def shift_spectra_linear(
    spectra: torch.Tensor,
    wavegrid: torch.Tensor,
    velocities: torch.Tensor,
    extrapolate: Literal["constant", "zero", "one", "linear"] = "constant",
    return_mask: bool = False,
) -> torch.Tensor:
    """
        Doppler shift un ensemble de spectres en fonction d'une grille de longueurs d'onde et d'un ensemble de vitesses.

        ! Tous les spectres doivent être de manière idéalement contiguë pour de meilleures performances !
    Args:
        spectra (torch.Tensor): Batch de spectres à décaler, de forme [B, n_pixel].
        wavegrid (torch.Tensor): Batch de grilles de longueurs d'onde, de forme [B, n_pixel] (la même répétée).
        velocities (torch.Tensor): Vitesse de décalage Doppler, de forme [B, 1] ou [B].
        extrapolate (str): Méthode d'extrapolation à utiliser si la grille de longueurs d'onde est en dehors des limites des spectres.
            ('constant', 'zero', 'one', 'linear'). Par défaut, 'constant'.

    Returns:
        torch.Tensor: Spectres décalés, de forme [B, n_pixel].

    Example:
        >>> from dataset import SpectrumDataset
        >>> B = 32  # Batch size
        >>> dataset = SpectrumDataset(n_specs=100, wavemin=5000, wavemax=5050, data_dtype=torch.float32)
        >>> batch_yobs = dataset.spectra
        >>> batch_wave = dataset.wavegrid.unsqueeze(0).expand(B, -1).contiguous()  # [B, n_pixel]
        >>> batch_voffset = torch.from_numpy(np.random.uniform(-3, 3, size=(B, 1))).cuda()
        >>> velocities = batch_voffset.view(-1, 1)
        >>> batch_yaug, extrap_mask = shift_spectra_linear(
        ...     spectra=batch_yobs,
        ...     wavegrid=batch_wave,
        ...     velocities=batch_voffset,
        ...     extrapolate="linear",
        ... )
    """
    # Constantes
    c = 299_792_458.0

    # Vérifications de base
    assert spectra.shape == wavegrid.shape, (
        "Le spectre et la grille doivent avoir la même forme"
    )
    assert spectra.shape[0] == velocities.shape[0], (
        "Le nombre de spectres et le nombre de vitesses doivent correspondre"
    )
    assert extrapolate in ["constant", "zero", "one", "linear"], (
        "Extrapolation doit être l'une des valeurs suivantes : 'constant', 'zero', 'one', 'linear'"
    )

    velocities = velocities.view(-1, 1)

    # Facteur Doppler
    doppler = torch.sqrt((1 + velocities / c) / (1 - velocities / c))

    # Grille décalée
    shifted = wavegrid * doppler

    # Interpolation via searchsorted
    idx = torch.searchsorted(shifted, wavegrid)

    # On récupère les indices pour les bords qui vont être extrapolés
    # idx == 0 signifie que la valeur de wavegrid est inférieure à la première valeur de shifted
    # idx == wavegrid.shape[-1] signifie que la valeur de wavegrid est supérieure à la dernière valeur de shifted
    mask_low = idx == 0
    mask_high = idx == wavegrid.shape[-1]
    extrap_mask = mask_low | mask_high

    idx = torch.clamp(idx, 1, spectra.shape[-1] - 1)

    # Extract intervals
    left_idx = idx - 1
    right_idx = idx

    λ_left = shifted.gather(-1, left_idx)
    λ_right = shifted.gather(-1, right_idx)
    f_left = spectra.gather(-1, left_idx)
    f_right = spectra.gather(-1, right_idx)

    # Linear interpolation
    t = (wavegrid - λ_left) / (λ_right - λ_left + 1e-12)
    result = f_left + t * (f_right - f_left)

    # Extrapolation si demandé
    if extrapolate == "zero":
        result = torch.where(extrap_mask, 0.0, result)
    elif extrapolate == "one":
        result = torch.where(extrap_mask, 1.0, result)
    elif extrapolate == "constant":
        # pour constant, on utilise f_left et f_right sur ce même mask
        result = torch.where(mask_low, f_left, result)
        result = torch.where(mask_high, f_right, result)

    if return_mask:
        return result, extrap_mask
    else:
        return result


def augment_spectra_uniform(
    batch_yobs: torch.Tensor,
    batch_wave: torch.Tensor,
    vmin: float = -3.0,
    vmax: float = 3.0,
    interpolate: Literal["linear", "cubic"] = "linear",
    extrapolate: Literal["constant", "zero", "one", "linear"] = "constant",
    out_dtype: torch.dtype = torch.float32,
):
    """
    Augmente les spectres en appliquant un décalage Doppler uniforme selon la méthode AESTRA.

    Supporte la conversion vers torch.float16 (half precision) pour économiser la mémoire GPU.

    Args:
        batch_yobs (torch.Tensor): Spectres observés, de forme [B, n_pixel].
        batch_wave (torch.Tensor): Grille de longueurs d'onde, de forme [B, n_pixel].
        vmin (float): Vitesse minimale de décalage Doppler.
        vmax (float): Vitesse maximale de décalage Doppler.
        interpolate (str): Méthode d'interpolation à utiliser ('linear' ou 'cubic').
            -> cubic utilise la méthode d'AESTRA mais différences minimes avec linéaire et beaucoup plus lent.
        extrapolate (str): Méthode d'extrapolation à utiliser si la grille de longueurs d'onde est en dehors des limites des spectres.
        out_dtype (torch.dtype): Type de données de sortie des spectres augmentés.
            - torch.float16: Half precision (économise ~50% de mémoire GPU)
            - torch.float32: Single precision (par défaut)
            - torch.float64: Double precision
    Returns:
        torch.Tensor: Spectres augmentés, de forme [B, n_pixel] de type out_dtype.
        torch.Tensor: Vitesse de décalage Doppler appliquée, de forme [B] de type out_dtype.

    Note:
        L'utilisation de torch.float16 peut réduire significativement l'utilisation mémoire GPU
        mais peut introduire une légère perte de précision numérique. Idéal pour les gros batches.

    todo: éventuellement ajouter une option permettant de couper les valeurs extrapolées des spectres mais pour de petites vitesses l'extrapolation n'est pas très importante.
    """
    B, n_pixel = batch_yobs.shape

    # Génération des vitesses aléatoires
    batch_voffset = (
        torch.from_numpy(np.random.uniform(vmin, vmax, size=(B, 1)))
        .to(batch_yobs.device)
        .double()
    )

    if interpolate == "linear":
        batch_yaug = shift_spectra_linear(
            spectra=batch_yobs,
            wavegrid=batch_wave,
            velocities=batch_voffset,
            extrapolate=extrapolate,
            return_mask=False,
        )
    elif interpolate == "cubic":
        batch_yaug = shift_spectra_cubic(
            wavegrid=batch_wave,
            spectra=batch_yobs,
            velocities=batch_voffset,
            return_mask=False,
        )
    else:
        raise ValueError("interpolate doit être 'linear' ou 'cubic'")

    if out_dtype == torch.float16:
        return batch_yaug.half(), batch_voffset.squeeze(-1).half()
    elif out_dtype == torch.float32:
        return batch_yaug.float(), batch_voffset.squeeze(-1).float()
    elif out_dtype == torch.float64:
        return batch_yaug.double(), batch_voffset.squeeze(-1).double()
    else:
        raise ValueError(
            f"Type de sortie non supporté: {out_dtype}. Utilisez torch.float16, torch.float32 ou torch.float64."
        )


if __name__ == "__main__":
    from dataset import SpectrumDataset
    import matplotlib.pyplot as plt

    B = 32  # Taille du batch
    dataset = SpectrumDataset(
        n_specs=100, wavemin=5000, wavemax=5050, data_dtype=torch.float32
    )

    batch_yobs = dataset.spectra[:B]  # [B, n_pixel]

    batch_wave = (
        dataset.wavegrid.unsqueeze(0).expand(B, -1).contiguous()
    )  # [B, n_pixel]
    batch_yaug_lin, batch_voffset_lin = augment_spectra_uniform(
        batch_yobs=batch_yobs,
        batch_wave=batch_wave,
        vmin=-3,
        vmax=3,
        interpolate="linear",
        extrapolate="constant",
    )  # [B, n_pixel]

    batch_yaug_cub, batch_voffset_cub = augment_spectra_uniform(
        batch_yobs=batch_yobs,
        batch_wave=batch_wave,
        vmin=-3,
        vmax=3,
        interpolate="cubic",
        extrapolate="constant",
    )  # [B, n_pixel]

    # Affichage des résultats
    random_index = np.random.randint(0, B)
    yobs = batch_yobs[random_index].cpu().numpy()
    yaug_lin = batch_yaug_lin[random_index].cpu().numpy()
    yaug_cub = batch_yaug_cub[random_index].cpu().numpy()
    wave = batch_wave[random_index].cpu().numpy()
    v_lin = batch_voffset_lin[random_index].item()
    v_cub = batch_voffset_cub[random_index].item()

    plt.figure(figsize=(10, 5))
    plt.plot(wave, yobs, label="Spectre observé", color="blue")
    plt.plot(wave, yaug_lin, label="Spectre augmenté linéaire", color="orange")
    plt.title(f"Spectres pour l'indice {random_index} (v = {v_lin:.2f} m/s)")
    plt.xlabel("Longueur d'onde (nm)")
    plt.ylabel("Intensité")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(wave, yobs, label="Spectre observé", color="blue")
    plt.plot(wave, yaug_cub, label="Spectre augmenté cubique", color="orange")
    plt.title(f"Spectres pour l'indice {random_index} (v = {v_cub:.2f} m/s)")
    plt.xlabel("Longueur d'onde (nm)")
    plt.ylabel("Intensité")
    plt.legend()
    plt.grid()
    plt.show()
