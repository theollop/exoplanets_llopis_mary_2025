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

import pandas as pd
import os
from astropy.timeseries import LombScargle
import glob


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
    batch_vencode = model.rvestimator(batch_robs)

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


def compute_rv_periodogram(model, dataset, star_name=None):
    """
    Calcule le périodogramme des vitesses radiales en utilisant LombScargle.

    Args:
        model: Modèle AESTRA entraîné
        dataset: Dataset contenant les spectres et les temps JDB
        star_name: Nom de l'étoile pour identifier le fichier de transit (optionnel)

    Returns:
        tuple: (periods, power, rv_values, times)
            - periods: Périodes en jours
            - power: Puissance du périodogramme
            - rv_values: Valeurs de vitesses radiales calculées
            - times: Temps JDB correspondants
    """
    model.eval()

    print("Calcul des vitesses radiales à partir des spectres...")

    # Récupération des temps depuis le dataset
    times = dataset.jdb.cpu().numpy()

    # Calcul des vitesses radiales pour tous les spectres
    rv_values = []

    with torch.no_grad():
        # Traitement par batch pour économiser la mémoire
        batch_size = 32
        n_specs = len(dataset)

        for i in range(0, n_specs, batch_size):
            end_idx = min(i + batch_size, n_specs)
            batch_spectra = dataset.spectra[i:end_idx]

            # Calcul de l'encodage de vitesse pour ce batch
            batch_robs = batch_spectra - model.b_obs.unsqueeze(0)
            batch_vencode = model.rvestimator(batch_robs)

            # Conversion en vitesses radiales (supposant que vencode donne directement RV)
            rv_batch = batch_vencode.cpu().numpy()

            # Si vencode est multidimensionnel, prendre la moyenne ou une composante spécifique
            if rv_batch.ndim > 1:
                rv_batch = rv_batch.mean(axis=1)

            rv_values.extend(rv_batch)

    rv_values = np.array(rv_values)

    print(f"Nombre de mesures RV: {len(rv_values)}")
    print(f"Période d'observation: {times.max() - times.min():.1f} jours")
    print(f"RV min/max: {rv_values.min():.3f} / {rv_values.max():.3f}")

    # Calcul du périodogramme avec LombScargle
    # Définition de la grille de périodes
    min_period = 1.0  # 1 jour minimum
    max_period = (times.max() - times.min()) / 3  # 1/3 de la durée totale maximum

    # Grille logarithmique de périodes
    periods = np.logspace(np.log10(min_period), np.log10(max_period), 10000)

    # Calcul des fréquences correspondantes
    frequencies = 1.0 / periods

    # Calcul du périodogramme Lomb-Scargle
    ls = LombScargle(times, rv_values)
    power = ls.power(frequencies)

    print(f"Périodogramme calculé pour {len(periods)} périodes")
    print(f"Période min/max: {min_period:.1f} / {max_period:.1f} jours")

    return periods, power, rv_values, times


def get_transit_periods(star_name="STAR1136"):
    """
    Récupère les périodes des planètes connues depuis le fichier Transit_information.csv

    Args:
        star_name: Nom de l'étoile (ex: "STAR1134", "STAR1136")

    Returns:
        list: Liste des périodes en jours, ou None si le fichier n'est pas trouvé
    """
    # Utiliser STAR1136 par défaut car c'est l'étoile configurée dans base_config.yaml
    data_path = "data/rv_datachallenge"
    pattern = f"{data_path}/*/{star_name}_HPN_Transit_information.csv"
    files = glob.glob(pattern)

    if files:
        transit_file = files[0]
    else:
        print(f"Fichier Transit_information.csv pour {star_name} non trouvé")
        return None

    try:
        transit_info = pd.read_csv(transit_file)
        # Filtrer les périodes non-nulles
        periods = transit_info["p"].dropna().values
        print(f"Périodes connues trouvées dans {transit_file}: {periods}")
        return periods
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier {transit_file}: {e}")
        return None


def plot_rv_periodogram(
    periods,
    power,
    rv_values,
    times,
    known_periods=None,
    save_path="reports/figures/rv_periodogram_with_known_periods.png",
    show_plot=False,
):
    """
    Trace le périodogramme des vitesses radiales avec des zooms sur les périodes d'intérêt.

    Args:
        periods: Périodes en jours
        power: Puissance du périodogramme
        rv_values: Valeurs de vitesses radiales
        times: Temps JDB
        known_periods: Liste des périodes connues des planètes (optionnel)
        save_path: Chemin pour sauvegarder la figure
        show_plot: Afficher la figure ou non
    """

    # Création de la figure avec plusieurs sous-graphiques
    if known_periods is not None and len(known_periods) > 0:
        fig = plt.figure(figsize=(18, 14))

        # Graphique principal du périodogramme
        ax1 = plt.subplot(4, 3, (1, 3))
        ax1.semilogx(periods, power, "b-", linewidth=0.8)
        ax1.set_xlabel("Période (jours)")
        ax1.set_ylabel("Puissance LS")
        ax1.set_title("Périodogramme Lomb-Scargle des vitesses radiales")
        ax1.grid(True, alpha=0.3)

        # Marquer les périodes connues
        for i, period in enumerate(known_periods):
            if periods.min() <= period <= periods.max():
                ax1.axvline(
                    period,
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                    label=f"Planète {i + 1}: {period:.1f}d",
                )

        ax1.legend()

        # Graphique des vitesses radiales en fonction du temps
        ax2 = plt.subplot(4, 3, (4, 6))
        ax2.plot(times, rv_values, "ko-", markersize=2, linewidth=0.5)
        ax2.set_xlabel("JDB")
        ax2.set_ylabel("Vitesse radiale")
        ax2.set_title("Série temporelle des vitesses radiales")
        ax2.grid(True, alpha=0.3)

        # Zooms sur les 3 périodes d'intérêt
        zoom_positions = [7, 8, 9]  # Positions dans la grille 4x3
        for i, period in enumerate(known_periods[:3]):
            if periods.min() <= period <= periods.max() and i < 3:
                ax_zoom = plt.subplot(4, 3, zoom_positions[i])

                # Définir la fenêtre de zoom autour de la période connue
                zoom_factor = 0.2  # ±20% autour de la période
                period_min = period * (1 - zoom_factor)
                period_max = period * (1 + zoom_factor)

                # Masque pour la zone de zoom
                zoom_mask = (periods >= period_min) & (periods <= period_max)

                if np.any(zoom_mask):
                    ax_zoom.plot(
                        periods[zoom_mask], power[zoom_mask], "b-", linewidth=1.5
                    )
                    ax_zoom.axvline(
                        period, color="red", linestyle="--", alpha=0.8, linewidth=2
                    )
                    ax_zoom.set_xlabel("Période (jours)")
                    ax_zoom.set_ylabel("Puissance LS")
                    ax_zoom.set_title(f"Zoom Planète {i + 1}: {period:.1f}d")
                    ax_zoom.grid(True, alpha=0.3)

                    # Trouver le pic local le plus proche
                    local_max_idx = np.argmax(power[zoom_mask])
                    if len(periods[zoom_mask]) > 0:
                        local_max_period = periods[zoom_mask][local_max_idx]
                        local_max_power = power[zoom_mask][local_max_idx]
                        ax_zoom.plot(
                            local_max_period,
                            local_max_power,
                            "ro",
                            markersize=8,
                            label=f"Max local: {local_max_period:.1f}d",
                        )
                        ax_zoom.legend(fontsize=8)

    else:
        # Si pas de périodes connues, graphique simple
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Périodogramme
        ax1.semilogx(periods, power, "b-", linewidth=0.8)
        ax1.set_xlabel("Période (jours)")
        ax1.set_ylabel("Puissance LS")
        ax1.set_title("Périodogramme Lomb-Scargle des vitesses radiales")
        ax1.grid(True, alpha=0.3)

        # Vitesses radiales
        ax2.plot(times, rv_values, "ko-", markersize=3, linewidth=0.5)
        ax2.set_xlabel("JDB")
        ax2.set_ylabel("Vitesse radiale")
        ax2.set_title("Série temporelle des vitesses radiales")
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Sauvegarde
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Périodogramme sauvegardé: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def analyze_rv_periodogram_from_checkpoint(
    checkpoint_path="models/aestra_base_config_final.pth",
    star_name="STAR1136",
    save_path="reports/figures/rv_periodogram_analysis.png",
    show_plot=False,
):
    """
    Fonction autonome pour calculer et tracer le périodogramme depuis un checkpoint.

    Args:
        checkpoint_path: Chemin vers le checkpoint du modèle
        star_name: Nom de l'étoile pour chercher les périodes connues
        save_path: Chemin pour sauvegarder la figure
        show_plot: Afficher la figure ou non

    Returns:
        dict: Dictionnaire contenant les résultats de l'analyse
    """

    print(f"Chargement du checkpoint: {checkpoint_path}")

    # Chargement du modèle et du dataset
    exp_data = load_experiment_checkpoint(checkpoint_path)
    model = exp_data["model"]
    dataset = exp_data["dataset"]
    cfg_name = exp_data["cfg_name"]
    start_epoch = exp_data["epoch"]
    current_phase = exp_data["current_phase"]

    print(f"Modèle chargé: {cfg_name}, epoch {start_epoch}, phase {current_phase}")
    print(f"Dataset: {len(dataset)} spectres, {dataset.n_pixels} pixels")

    # Récupération des périodes connues
    known_periods = get_transit_periods(star_name)

    # Calcul du périodogramme
    print("Calcul du périodogramme des vitesses radiales...")
    periods, power, rv_values, times = compute_rv_periodogram(
        model=model, dataset=dataset
    )

    # Analyse des pics
    threshold = 0.1 * np.max(power)
    peak_indices = np.where(power > threshold)[0]

    peaks_info = []
    if len(peak_indices) > 0:
        peak_periods = periods[peak_indices]
        peak_powers = power[peak_indices]

        sorted_indices = np.argsort(peak_powers)[::-1][:10]
        for i, idx in enumerate(sorted_indices):
            period = peak_periods[idx]
            power_val = peak_powers[idx]
            peaks_info.append({"rank": i + 1, "period": period, "power": power_val})

    # Création du plot
    plot_rv_periodogram(
        periods=periods,
        power=power,
        rv_values=rv_values,
        times=times,
        known_periods=known_periods,
        save_path=save_path,
        show_plot=show_plot,
    )

    # Retour des résultats
    results = {
        "periods": periods,
        "power": power,
        "rv_values": rv_values,
        "times": times,
        "known_periods": known_periods,
        "peaks_info": peaks_info,
        "max_power": np.max(power),
        "best_period": periods[np.argmax(power)],
        "observation_span": times.max() - times.min(),
        "n_observations": len(rv_values),
    }

    return results


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
        perturbation_scale=0.1,
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

    # Calcul et plot du périodogramme des vitesses radiales
    print("Calcul du périodogramme des vitesses radiales...")

    # Récupération des périodes connues des planètes pour STAR1136
    known_periods = get_transit_periods("STAR1136")

    # Calcul du périodogramme
    periods, power, rv_values, times = compute_rv_periodogram(
        model=model, dataset=dataset
    )

    print(f"Périodogramme calculé avec {len(periods)} points")
    print(
        f"Puissance max: {np.max(power):.3e} à la période {periods[np.argmax(power)]:.2f} jours"
    )

    # Affichage des pics principaux
    # Trouver les pics significatifs (seuil à 90% du maximum)
    threshold = 0.1 * np.max(power)
    peak_indices = np.where(power > threshold)[0]
    if len(peak_indices) > 0:
        peak_periods = periods[peak_indices]
        peak_powers = power[peak_indices]

        # Trier par puissance décroissante et prendre les 5 premiers
        sorted_indices = np.argsort(peak_powers)[::-1][:5]
        print("Pics principaux détectés:")
        for i, idx in enumerate(sorted_indices):
            period = peak_periods[idx]
            power_val = peak_powers[idx]
            print(f"  {i + 1}. Période: {period:.2f} jours, Puissance: {power_val:.3e}")

    # Création et sauvegarde du plot du périodogramme
    save_path_periodo = "reports/figures/rv_periodogram_with_known_periods.png"
    plot_rv_periodogram(
        periods=periods,
        power=power,
        rv_values=rv_values,
        times=times,
        known_periods=known_periods,
        save_path=save_path_periodo,
        show_plot=True,
    )

    print("\n" + "=" * 60)
    print("RÉSUMÉ DE L'ANALYSE DU PÉRIODOGRAMME")
    print("=" * 60)
    print(f"Nombre d'observations: {len(rv_values)}")
    print(f"Durée d'observation: {times.max() - times.min():.1f} jours")
    print(f"Plage RV: {rv_values.min():.2f} à {rv_values.max():.2f}")
    if known_periods is not None:
        print(f"Périodes connues: {known_periods}")
    print(f"Meilleure période détectée: {periods[np.argmax(power)]:.2f} jours")
    print(f"Puissance maximale: {np.max(power):.3e}")
    print("=" * 60)


# Exemple d'utilisation autonome des fonctions:
# ==============================================
#
# from src.modeling.predict import analyze_rv_periodogram_from_checkpoint
#
# # Analyse complète depuis un checkpoint
# results = analyze_rv_periodogram_from_checkpoint(
#     checkpoint_path="models/aestra_base_config_final.pth",
#     save_path="mon_periodogramme.png",
#     show_plot=True
# )
#
# print(f"Meilleure période: {results['best_period']:.2f} jours")
# print(f"Puissance max: {results['max_power']:.3e}")
# print(f"Pics principaux: {results['peaks_info'][:3]}")
#
