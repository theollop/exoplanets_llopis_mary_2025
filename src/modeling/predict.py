import torch
from src.modeling.models import AESTRA
from src.dataset import SpectrumDataset
import numpy as np
import matplotlib.pyplot as plt
from src.modeling.train import load_experiment_checkpoint
from src.interpolate import augment_spectra_uniform
from src.plots_aestra import (
    plot_latent_distance_distribution,
    plot_activity_perturbation,
)


def get_vencode(model, batch):
    """
    Get the velocity encoding from the batch of observations.
    The batch should contain the observed spectra and the model's b_obs parameter.

    Args:
        model (AESTRA): The AESTRA model instance.
        batch (tuple): A tuple containing the batch data, where the first element is the observed spectra.
    Returns:
        torch.Tensor: The velocity encoding of the observed spectra.
    """
    batch_yobs, _, _, _ = batch
    batch_robs = batch_yobs - model.b_obs.unsqueeze(0)
    batch_vencode = model.rv_estimator(batch_robs)

    return batch_vencode


def get_spender_output(model, batch, aug_output=False):
    """

    Get the output of the spender module from the batch of observations.
    The batch should contain the observed spectra and the model's b_obs parameter.
    Args:
        model (AESTRA): The AESTRA model instance.
        batch (tuple): A tuple containing the batch data, where the first element is the observed spectra.
        aug_output (bool): If True, also return the augmented output.
    Returns:
        tuple: A tuple containing the predicted activity and the state from the spender module.
               If aug_output is True, it also includes the augmented activity and state.
    """

    batch_yobs, batch_yaug, _, batch_wavegrid = batch

    batch_robs = batch_yobs - model.b_obs.unsqueeze(0)

    batch_yact, batch_s = model.spender(batch_robs)
    if aug_output:
        batch_yact_aug, batch_s_aug = model.spender(batch_yaug)
        return batch_yact, batch_s, batch_yact_aug, batch_s_aug

    return batch_yact, batch_s


# Figure 3: Calcul des distances latentes
# ==============================================================================
# Cette fonction calcule les distances latentes pour les spectres originaux et augmentés.
# Elle est utilisée pour reproduire la Figure 3 de l'article.
# Elle prend en entrée le modèle AESTRA et le dataset contenant les spectres.
# Elle retourne les distances latentes pour les paires aléatoires et augmentées.


def compute_latent_distances(model, dataset):
    """
    Calcule les distances latentes pour reproduire la Figure 3.

    Args:
        model: Modèle AESTRA entraîné
        dataset: Dataset contenant les spectres
        n_specs: Nombre de spectres à utiliser pour le calcul
        n_random_pairs: Nombre de paires aléatoires à générer

    Returns:
        tuple: (delta_s_rand, delta_s_aug) - distances latentes pour paires aléatoires et augmentées
    """
    model.eval()

    # Limitation au nombre de spectres disponibles
    n_specs = len(dataset)

    # Sélection aléatoire des spectres
    indices = torch.randperm(len(dataset))[:n_specs]
    selected_spectra = dataset.spectra[indices]

    # Grille de longueurs d'onde répétée pour tous les spectres
    batch_wavegrid = dataset.wavegrid.unsqueeze(0).repeat(n_specs, 1).contiguous()

    with torch.no_grad():
        # Calcul des encodages latents pour les spectres originaux
        batch_robs = selected_spectra - model.b_obs.unsqueeze(0)
        _, latent_s = model.spender(batch_robs)

        # Génération des spectres augmentés avec décalage Doppler
        batch_yaug, batch_voffset = augment_spectra_uniform(
            batch_yobs=selected_spectra,
            batch_wave=batch_wavegrid,
            vmin=-3.0,
            vmax=3.0,
            interpolate="linear",
            extrapolate="linear",
            out_dtype=torch.float32,
        )

        # Calcul des encodages latents pour les spectres augmentés
        batch_raug = batch_yaug - model.b_obs.unsqueeze(0)
        _, latent_s_aug = model.spender(batch_raug)

        # Calcul des distances latentes pour les paires de données augmentées (∆s_aug)
        delta_s_aug = torch.norm(latent_s - latent_s_aug, dim=1).cpu().numpy()

        # Génération de paires aléatoires pour calculer ∆s_rand
        n_random_pairs = n_specs

        # Génération d'indices de paires aléatoires sans remise
        all_pairs = [(i, j) for i in range(n_specs) for j in range(i + 1, n_specs)]
        selected_pairs = np.random.choice(
            len(all_pairs), size=n_random_pairs, replace=False
        )

        delta_s_rand = []
        for pair_idx in selected_pairs:
            i, j = all_pairs[pair_idx]
            dist = torch.norm(latent_s[i] - latent_s[j], dim=0).cpu().numpy()
            delta_s_rand.append(dist)

        delta_s_rand = np.array(delta_s_rand)

    return delta_s_rand, delta_s_aug


# Figure 2: Calcul de la perturbation d'activité
def compute_activity_perturbation(model, dataset, idx=0, perturbation_scale=1.0):
    """
    Calcule les spectres d'activité perturbés pour reproduire la Figure 2.

    Args:
        model: Modèle AESTRA entraîné
        dataset: Dataset contenant les spectres
        idx: Index du spectre à utiliser
        perturbation_scale: Échelle de la perturbation appliquée

    Returns:
        tuple: (y_act_original, y_act_perturbed_list, wavelength)
            - y_act_original: Spectre d'activité original (courbe noire)
            - y_act_perturbed_list: Liste des spectres perturbés pour chaque composante
            - wavelength: Grille de longueurs d'onde
    """
    model.eval()

    batch_yobs = dataset.spectra[idx].unsqueeze(0)  # Sélection d'un spectre spécifique
    wavelength = dataset.wavegrid.cpu().numpy()

    with torch.no_grad():
        # Calcul des encodages latents pour les spectres originaux
        batch_robs = batch_yobs - model.b_obs.unsqueeze(0)
        batch_yact_original, batch_s = model.spender(batch_robs)

        # Spectre d'activité original (courbe noire dans la Figure 2)
        y_act_original = batch_yact_original.squeeze(0).cpu().numpy()

        s = batch_s.squeeze(0)
        y_act_perturbed_list = []

        # Perturbation de chaque composante du vecteur latent
        latent_dim = s.shape[0]  # Dimension du vecteur latent
        for dim in range(
            min(3, latent_dim)
        ):  # Limiter à 3 composantes comme dans la Figure 2
            s_perturbed = s.clone()
            # Perturbation déterministe pour la reproductibilité
            # Si s[dim] est un scalaire, on utilise une perturbation fixe
            if s[dim].numel() == 1:
                perturbation = perturbation_scale * 0.1  # Perturbation fixe
            else:
                perturbation = perturbation_scale * torch.std(s[dim])
            s_perturbed[dim] += perturbation

            # Génération du nouveau spectre d'activité avec le vecteur latent perturbé
            y_act_perturbed = model.spender.decoder(s_perturbed.unsqueeze(0))
            y_act_perturbed_list.append(y_act_perturbed.squeeze(0).cpu().numpy())

    return y_act_original, y_act_perturbed_list, wavelength


if __name__ == "__main__":
    # Chargement du modèle entraîné
    checkpoint_path = "models/aestra_base_config_final.pth"
    print(f"Chargement du checkpoint: {checkpoint_path}")

    exp_data = load_experiment_checkpoint(checkpoint_path)
    model = exp_data["model"]
    dataset = exp_data["dataset"]
    cfg_name = exp_data["cfg_name"]
    start_epoch = exp_data["epoch"]
    current_phase = exp_data["current_phase"]

    print(f"Modèle chargé: {cfg_name}, epoch {start_epoch}, phase {current_phase}")
    print(f"Dataset: {len(dataset)} spectres, {dataset.n_pixels} pixels")

    # Calcul des distances latentes (comme dans la Figure 3)
    print("Calcul des distances latentes...")
    delta_s_rand, delta_s_aug = compute_latent_distances(
        model=model,
        dataset=dataset,
    )

    print(f"Nombre de paires aléatoires: {len(delta_s_rand)}")
    print(f"Nombre de paires augmentées: {len(delta_s_aug)}")
    print(f"Moyenne ∆s_rand: {np.mean(delta_s_rand):.3e}")
    print(f"Moyenne ∆s_aug: {np.mean(delta_s_aug):.3e}")
    print(
        f"Min ∆s_rand: {np.min(delta_s_rand):.3e}, Max ∆s_rand: {np.max(delta_s_rand):.3e}"
    )
    print(
        f"Min ∆s_aug: {np.min(delta_s_aug):.3e}, Max ∆s_aug: {np.max(delta_s_aug):.3e}"
    )

    # Création et sauvegarde du plot
    save_path = "reports/figures/latent_distance_distribution_base_config2.png"
    plot_latent_distance_distribution(
        delta_s_rand=delta_s_rand, delta_s_aug=delta_s_aug, save_path=save_path
    )

    # Calcul et plot de la perturbation d'activité (Figure 2)
    print("Calcul de la perturbation d'activité...")
    y_act_original, y_act_perturbed_list, wavelength = compute_activity_perturbation(
        model=model,
        dataset=dataset,
        idx=0,  # Premier spectre du dataset
        perturbation_scale=1,
    )

    print(f"Spectre d'activité original: {y_act_original.shape}")
    print(f"Nombre de spectres perturbés: {len(y_act_perturbed_list)}")

    # Sauvegarde de la Figure 2
    save_path_fig2 = "reports/figures/activity_perturbation_figure2.png"
    plot_activity_perturbation(
        y_act_original=y_act_original,
        y_act_perturbed_list=y_act_perturbed_list,
        wavelength=wavelength,
        save_path=save_path_fig2,
        wave_range=(5000, 5010),  # Gamme de longueurs d'onde comme dans la Figure 2
        show_plot=True,
    )
