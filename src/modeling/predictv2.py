import torch
import numpy as np
from astropy.timeseries import LombScargle

from src.modeling.train import load_experiment_checkpoint
from torch.utils.data import DataLoader
from src.dataset import generate_collate_fn
from src.utils import clear_gpu_memory
import matplotlib.pyplot as plt
from src.interpolate import shift_spectra_linear


def inject_dataset(
    dataset, amplitudes: list[float], periods: list[float], batch_size=None
):
    """
    Injects artificial planetary signals into the dataset by shifting spectra according to RV variations.

    Args:
        dataset: SpectrumDataset containing spectra, wavegrid and time values (jdb)
        amplitudes: List of semi-amplitudes (Kp) in m/s for each planet
        periods: List of periods (P) in days for each planet
        batch_size: If None, process all spectra at once. Otherwise, process in batches.

    Returns:
        torch.Tensor: Modified spectra with injected planetary signals
    """
    # Calculate RV velocities for all time points
    velocities = np.zeros(len(dataset.jdb))
    time_values = (
        dataset.jdb.cpu().numpy() if hasattr(dataset.jdb, "cpu") else dataset.jdb
    )

    for Kp, P in zip(amplitudes, periods):
        velocities += Kp * np.sin(2 * np.pi * time_values / P)

    # Convert to tensor and ensure same device as dataset
    velocities = torch.tensor(
        velocities, dtype=dataset.spectra.dtype, device=dataset.spectra.device
    )

    if batch_size is None:
        # Process all spectra at once
        print(f"Processing all {len(dataset)} spectra at once...")
        injected_spectra = shift_spectra_linear(
            spectra=dataset.spectra,
            wavegrid=dataset.wavegrid.unsqueeze(0)
            .expand(len(dataset), -1)
            .contiguous(),
            velocities=velocities,
        )
    else:
        # Process in batches
        print(f"Processing {len(dataset)} spectra in batches of {batch_size}...")
        injected_spectra_list = []

        for i in range(0, len(dataset), batch_size):
            end_idx = min(i + batch_size, len(dataset))
            batch_spectra = dataset.spectra[i:end_idx]
            batch_velocities = velocities[i:end_idx]
            batch_wavegrid = (
                dataset.wavegrid.unsqueeze(0).expand(end_idx - i, -1).contiguous()
            )

            batch_injected = shift_spectra_linear(
                spectra=batch_spectra,
                wavegrid=batch_wavegrid,
                velocities=batch_velocities,
            )

            injected_spectra_list.append(batch_injected)

        injected_spectra = torch.cat(injected_spectra_list, dim=0)

    return injected_spectra


def predict(
    model, dataset, batch_size=64, perturbation_value=1.0, amplitudes=None, periods=None
):
    """
    Extracts latent vectors and RV values from the model and dataset.
    """
    if amplitudes is not None and periods is not None:
        # Inject planetary signals into the dataset
        print("Injecting planetary signals into the dataset...")
        dataset.spectra = inject_dataset(
            dataset, amplitudes=amplitudes, periods=periods, batch_size=batch_size
        )
    all_s = []
    all_saug = []
    all_rvs = []
    all_yact = []
    all_yact_aug = []
    all_yobs_prime = []
    all_yact_perturbed = {}  # Dictionnaire pour organiser par dimension

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=generate_collate_fn(dataset=dataset),
    )

    for batch in dataloader:
        batch_yobs, batch_yaug, batch_voffset_true, batch_wavegrid = batch

        batch_vobs_pred, _ = model.get_rvestimator_pred(
            batch_yobs=batch_yobs,
            batch_yaug=batch_yaug,
        )

        batch_yobs_prime, batch_yact, batch_yact_aug, batch_s, batch_saug = (
            model.get_spender_pred(
                batch_yobs=batch_yobs,
                batch_yaug=batch_yaug,
                batch_wavegrid=batch_wavegrid,
                batch_vobs_pred=batch_vobs_pred,
            )
        )

        latent_dim = batch_s.shape[1]

        # Initialiser les listes pour chaque dimension si pas encore fait
        if not all_yact_perturbed:
            for dim in range(latent_dim):
                all_yact_perturbed[dim] = []

        for dim in range(latent_dim):
            batch_s_dim_perturbed = batch_s.clone()
            batch_s_dim_perturbed[:, dim] += perturbation_value
            batch_yact_perturbed = model.spender.decoder(batch_s_dim_perturbed)
            all_yact_perturbed[dim].append(batch_yact_perturbed.cpu().detach().numpy())

        all_s.append(batch_s.cpu().detach().numpy())
        all_saug.append(batch_saug.cpu().detach().numpy())
        all_rvs.append(batch_voffset_true.cpu().detach().numpy())
        all_yact.append(batch_yact.cpu().detach().numpy())
        all_yact_aug.append(batch_yact_aug.cpu().detach().numpy())
        all_yobs_prime.append(batch_yobs_prime.cpu().detach().numpy())

    all_s = np.concatenate(all_s, axis=0)
    all_saug = np.concatenate(all_saug, axis=0)
    all_rvs = np.concatenate(all_rvs, axis=0)
    all_yact = np.concatenate(all_yact, axis=0)
    all_yact_aug = np.concatenate(all_yact_aug, axis=0)
    all_yobs_prime = np.concatenate(all_yobs_prime, axis=0)

    # Concaténer et réorganiser all_yact_perturbed en (latent_dim, n_specs, n_pixels)
    latent_dim = len(all_yact_perturbed)
    all_yact_perturbed_array = np.array(
        [np.concatenate(all_yact_perturbed[dim], axis=0) for dim in range(latent_dim)]
    )

    return {
        "all_s": all_s,
        "all_saug": all_saug,
        "all_rvs": all_rvs,
        "all_yact": all_yact,
        "all_yact_aug": all_yact_aug,
        "all_yobs_prime": all_yobs_prime,
        "all_yact_perturbed": all_yact_perturbed_array,
    }


def compute_latent_distances(all_s, all_saug, seed=None):
    """
    Fig 3.
    - Δs_rand : distances entre paires aléatoires de s_obs (i != j), tirées sans remise au sein de chaque paire.
    - Δs_aug  : distances entre s_obs_i et s_aug_i.
    Retourne (Δs_rand, Δs_aug).
    """
    n = all_s.shape[0]
    if seed is not None:
        np.random.seed(seed)

    # Pour chaque échantillon k, on tire une paire d'indices [i, j] avec i != j
    inds = np.array([np.random.choice(n, size=2, replace=False) for _ in range(n)])

    delta_s_rand = np.linalg.norm(all_s[inds[:, 0]] - all_s[inds[:, 1]], axis=1)

    delta_s_aug = np.linalg.norm(all_s - all_saug, axis=1)
    return delta_s_rand, delta_s_aug


def compute_periodogram(y_values, time_values):
    """
    Computes the Lomb-Scargle periodogram for the given y values and time values.
    Optionally highlights known periods if provided.

    Args:
        y_values: y values to analyze
        time_values: Corresponding time values for the y measurements
        known_periods: List of known periods to highlight in the plot (optional)

    Returns:
        Frequencies and power of the periodogram
    """
    # Calcul du périodogramme avec LombScargle
    # Définition de la grille de périodes
    min_period = 1.0  # 1 jour minimum
    max_period = (
        time_values.max() - time_values.min()
    ) / 3  # 1/3 de la durée totale maximum

    # Grille logarithmique de périodes
    periods = np.logspace(np.log10(min_period), np.log10(max_period), 10000)

    # Calcul des fréquences correspondantes
    frequencies = 1.0 / periods

    print(f"Calcul du périodogramme pour {len(periods)} périodes...")

    # Calcul du périodogramme Lomb-Scargle pour les y originales
    ls = LombScargle(time_values, y_values)
    power = ls.power(frequencies)

    return periods, power


def compute_correlation(latent_s, rv_values):
    """
    Computes the correlation between latent space dimensions and RV values.
    and within latent space dimensions values.
    """
    # Calcul de la corrélation entre les dimensions latentes et les RV
    correlation_matrix = np.corrcoef(latent_s.T, rv_values, rowvar=False)

    # Extraire les corrélations entre les dimensions latentes et les RV
    latent_rv_correlation = correlation_matrix[:-1, -1]

    # Extraire les corrélations entre les dimensions latentes elles-mêmes
    latent_latent_correlation = correlation_matrix[:-1, :-1]

    return latent_rv_correlation, latent_latent_correlation


def main(checkpoint_path: str):
    exp_data = load_experiment_checkpoint(path=checkpoint_path)

    model = exp_data["model"]
    dataset = exp_data["dataset"]

    prediction = predict(model, dataset)

    all_s = prediction["all_s"]
    all_saug = prediction["all_saug"]
    all_rvs = prediction["all_rvs"]
    all_yact = prediction["all_yact"]
    all_yact_aug = prediction["all_yact_aug"]
    all_yobs_prime = prediction["all_yobs_prime"]
    all_yact_perturbed = prediction["all_yact_perturbed"]

    return True


# ==== Fonctions de plots ====
def plot_latent_distance_distribution(
    delta_s_rand, delta_s_aug, save_path=None, show_plot=False
):
    """
    Crée le plot de la Figure 3 avec les distributions de distances latentes.

    Args:
        delta_s_rand: Distances latentes pour les paires aléatoires
        delta_s_aug: Distances latentes pour les paires augmentées
        save_path: Chemin pour sauvegarder la figure (optionnel)
    """
    # Configuration du plot
    plt.figure(figsize=(8, 6))

    # Calcul des statistiques pour les légendes
    mean_rand = np.mean(delta_s_rand)
    mean_aug = np.mean(delta_s_aug)

    # Détermination automatique de la plage des valeurs
    all_values = np.concatenate([delta_s_rand, delta_s_aug])
    min_val = np.min(
        all_values[all_values > 0]
    )  # Éviter les valeurs nulles pour le log
    max_val = np.max(all_values)

    # Extension de la plage pour une meilleure visualisation
    x_min = min_val * 0.5
    x_max = max_val * 2.0

    # Création des histogrammes avec bins adaptés aux données réelles
    bins = np.logspace(np.log10(x_min), np.log10(x_max), 50)

    plt.hist(
        delta_s_rand,
        bins=bins,
        alpha=0.7,
        color="blue",
        label=f"(∆s_rand): {mean_rand:.3e}",
        density=False,
    )
    plt.hist(
        delta_s_aug,
        bins=bins,
        alpha=0.7,
        color="red",
        label=f"(∆s_aug): {mean_aug:.3e}",
        density=False,
    )

    # Configuration des axes et labels
    plt.xlabel("latent distance ∆s", fontsize=12)
    plt.ylabel("N", fontsize=12)
    plt.xscale("log")
    plt.xlim(x_min, x_max)

    # Ajout de la légende
    plt.legend(fontsize=12)

    # Configuration de la grille
    plt.grid(True, alpha=0.3)

    plt.subplots_adjust(bottom=0.15)

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Figure sauvegardée: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main(
        checkpoint_path="experiments/aestra_local_experiment/models/aestra_base_config_final.pth"
    )
    # Clear GPU memory after running the script
    clear_gpu_memory()
