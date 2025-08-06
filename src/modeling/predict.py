"""
Module d'analyse et de prédiction AESTRA.

Ce module contient les fonctions principales pour analyser les modèles AESTRA entraînés :
- Calcul des distances latentes (Figure 3)
- Analyse de la perturbation d'activité (Figure 2)
- Calcul du périodogramme des vitesses radiales
- Visualisation 3D de l'espace latent

Optimisé pour éviter les erreurs de mémoire (OOM) avec traitement par batches.
"""

import torch
import numpy as np
import pandas as pd
import os
import glob
from astropy.timeseries import LombScargle

from src.modeling.train import load_experiment_checkpoint
from src.interpolate import augment_spectra_uniform
from src.plots_aestra import (
    plot_latent_distance_distribution,
    plot_activity_perturbation,
    plot_latent_space_3d,
    plot_rv_periodogram,
)
from src.utils import clear_gpu_memory


# =============================================================================
# ANALYSIS FUNCTIONS - Fonctions d'analyse principales
# =============================================================================


def compute_latent_distances(model, dataset, batch_size=32):
    """
    Calcule les distances latentes pour reproduire la Figure 3.
    Version optimisée par batches pour éviter les OOM errors.

    Args:
        model: Modèle AESTRA entraîné
        dataset: Dataset contenant les spectres
        batch_size: Taille des batches pour le traitement (défaut 32)

    Returns:
        tuple: (delta_s_rand, delta_s_aug) - distances latentes pour paires aléatoires et augmentées
    """
    model.eval()

    n_specs = len(dataset)
    print(
        f"Calcul des distances latentes sur {n_specs} spectres (batch_size={batch_size})..."
    )

    # Traitement par batches pour éviter les OOM
    delta_s_aug_list = []
    latent_s_list = []

    with torch.no_grad():
        print("Calcul des encodages latents par batches...")

        for i in range(0, n_specs, batch_size):
            end_idx = min(i + batch_size, n_specs)
            current_batch_size = end_idx - i

            # Sélection des spectres pour ce batch
            batch_spectra = dataset.spectra[i:end_idx]

            # Grille de longueurs d'onde pour ce batch
            batch_wavegrid = (
                dataset.wavegrid.unsqueeze(0).repeat(current_batch_size, 1).contiguous()
            )

            # Calcul des encodages latents pour les spectres originaux
            batch_robs = batch_spectra - model.b_obs.unsqueeze(0)
            _, latent_s_batch = model.spender(batch_robs)

            # Stockage des encodages latents (sur CPU pour économiser GPU)
            latent_s_list.append(latent_s_batch.cpu())

            # Génération des spectres augmentés avec décalage Doppler
            batch_yaug, batch_voffset = augment_spectra_uniform(
                batch_yobs=batch_spectra,
                batch_wave=batch_wavegrid,
                vmin=-3.0,
                vmax=3.0,
                interpolate="linear",
                extrapolate="linear",
                out_dtype=torch.float32,
            )

            # Calcul des encodages latents pour les spectres augmentés
            batch_raug = batch_yaug - model.b_obs.unsqueeze(0)
            _, latent_s_aug_batch = model.spender(batch_raug)

            # Calcul des distances latentes pour les paires de données augmentées (∆s_aug)
            delta_s_aug_batch = (
                torch.norm(latent_s_batch - latent_s_aug_batch, dim=1).cpu().numpy()
            )
            delta_s_aug_list.extend(delta_s_aug_batch)

            # Nettoyage mémoire périodique
            if (i // batch_size) % 5 == 0:
                clear_gpu_memory()

        # Concaténation des encodages latents
        latent_s = torch.cat(latent_s_list, dim=0)
        delta_s_aug = np.array(delta_s_aug_list)

        print(f"Génération de {n_specs} paires aléatoires...")

        # Génération efficace de paires aléatoires pour calculer ∆s_rand
        delta_s_rand = []
        # Éviter de générer toutes les paires en mémoire
        for _ in range(n_specs):
            i, j = np.random.choice(n_specs, size=2, replace=False)
            dist = torch.norm(latent_s[i] - latent_s[j], dim=0).numpy()
            delta_s_rand.append(dist)

        delta_s_rand = np.array(delta_s_rand)

        # Nettoyage final
        clear_gpu_memory()

    print(
        f"✅ Distances calculées: {len(delta_s_rand)} paires aléatoires, {len(delta_s_aug)} paires augmentées"
    )
    return delta_s_rand, delta_s_aug


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


def extract_latent_vectors_and_rv(model, dataset, batch_size=32, remove_outliers=None):
    """
    Extrait les vecteurs latents et calcule les vitesses radiales correspondantes.
    Version optimisée par batches pour éviter les OOM errors.

    Args:
        model: Modèle AESTRA entraîné
        dataset: Dataset contenant les spectres
        batch_size: Taille des batches pour le traitement (défaut 32)
        remove_outliers: Liste des indices à supprimer (défaut [334, 464])

    Returns:
        tuple: (latent_s, rv_values)
            - latent_s: Vecteurs latents (N, D)
            - rv_values: Valeurs RV correspondantes (N,)
    """
    model.eval()

    if remove_outliers is None:
        remove_outliers = [334, 464]

    print("Extraction des vecteurs latents et calcul des RV...")

    # Création d'un masque pour exclure les indices outliers
    n_specs = len(dataset)
    valid_indices = [i for i in range(n_specs) if i not in remove_outliers]

    if remove_outliers:
        print(f"Suppression de {len(remove_outliers)} outliers: {remove_outliers}")
        print(f"Traitement de {len(valid_indices)} spectres sur {n_specs} total")

    # Collecte des vecteurs latents et des RV
    latent_vectors = []
    rv_values = []

    with torch.no_grad():
        # Traitement par batches seulement des indices valides
        for i in range(0, len(valid_indices), batch_size):
            end_idx = min(i + batch_size, len(valid_indices))
            batch_indices = valid_indices[i:end_idx]

            # Sélection des spectres pour ce batch
            batch_spectra = dataset.spectra[batch_indices]

            # Calcul des vecteurs latents
            batch_robs = batch_spectra - model.b_obs.unsqueeze(0)
            _, latent_s_batch = model.spender(batch_robs)

            # Calcul des RV pour ce batch
            batch_vencode = model.rvestimator(batch_robs)
            rv_batch = batch_vencode.cpu().numpy()

            # Si vencode est multidimensionnel, prendre la moyenne
            if rv_batch.ndim > 1:
                rv_batch = rv_batch.mean(axis=1)

            latent_vectors.append(latent_s_batch.cpu())
            rv_values.extend(rv_batch)

            # Nettoyage mémoire périodique
            if (i // batch_size) % 10 == 0:
                clear_gpu_memory()

    # Concaténation des résultats
    latent_s = torch.cat(latent_vectors, dim=0).numpy()
    rv_values = np.array(rv_values)

    # Nettoyage final
    clear_gpu_memory()

    print(
        f"✅ Extraction terminée: {latent_s.shape[0]} spectres, dim latente: {latent_s.shape[1]}"
    )
    if remove_outliers:
        print(f"Outliers supprimés: {len(remove_outliers)} spectres exclus")
    return latent_s, rv_values


def compute_rv_periodogram(
    model, dataset, star_name=None, batch_size=64, remove_outliers=None
):
    """
    Calcule le périodogramme des vitesses radiales en utilisant LombScargle.
    Version optimisée par batches pour éviter les OOM errors.

    Args:
        model: Modèle AESTRA entraîné
        dataset: Dataset contenant les spectres et les temps JDB
        star_name: Nom de l'étoile pour identifier le fichier de transit (optionnel)
        batch_size: Taille des batches pour le traitement (défaut 64)
        remove_outliers: Liste des indices à supprimer (défaut [334, 464])

    Returns:
        tuple: (periods, power, rv_values, times)
            - periods: Périodes en jours
            - power: Puissance du périodogramme
            - rv_values: Valeurs de vitesses radiales calculées
            - times: Temps JDB correspondants
    """
    model.eval()

    if remove_outliers is None:
        remove_outliers = [334, 464]

    print("Calcul des vitesses radiales à partir des spectres...")

    # Création d'un masque pour exclure les indices outliers
    n_specs = len(dataset)
    valid_indices = [i for i in range(n_specs) if i not in remove_outliers]

    if remove_outliers:
        print(f"Suppression de {len(remove_outliers)} outliers: {remove_outliers}")
        print(f"Traitement de {len(valid_indices)} spectres sur {n_specs} total")

    # Récupération des temps depuis le dataset (seulement pour les indices valides)
    all_times = dataset.jdb.cpu().numpy()
    times = all_times[valid_indices]

    # Calcul des vitesses radiales pour tous les spectres valides
    rv_values = []

    with torch.no_grad():
        # Traitement par batch pour économiser la mémoire
        print(
            f"Traitement de {len(valid_indices)} spectres par batches de {batch_size}..."
        )

        for i in range(0, len(valid_indices), batch_size):
            end_idx = min(i + batch_size, len(valid_indices))
            batch_indices = valid_indices[i:end_idx]

            # Sélection des spectres pour ce batch
            batch_spectra = dataset.spectra[batch_indices]

            # Calcul de l'encodage de vitesse pour ce batch
            batch_robs = batch_spectra - model.b_obs.unsqueeze(0)
            batch_vencode = model.rvestimator(batch_robs)

            # Conversion en vitesses radiales (supposant que vencode donne directement RV)
            rv_batch = batch_vencode.cpu().numpy()

            # Si vencode est multidimensionnel, prendre la moyenne ou une composante spécifique
            if rv_batch.ndim > 1:
                rv_batch = rv_batch.mean(axis=1)

            rv_values.extend(rv_batch)

            # Nettoyage mémoire périodique
            if (i // batch_size) % 10 == 0:
                clear_gpu_memory()

            # Progress indication
            if (i // batch_size) % 50 == 0:
                progress = (i / len(valid_indices)) * 100
                print(
                    f"Progress: {progress:.1f}% ({i}/{len(valid_indices)} spectres traités)"
                )

    rv_values = np.array(rv_values)

    # Nettoyage final
    clear_gpu_memory()

    print(f"Nombre de mesures RV: {len(rv_values)}")
    print(f"Période d'observation: {times.max() - times.min():.1f} jours")
    print(f"RV min/max: {rv_values.min():.3f} / {rv_values.max():.3f}")
    if remove_outliers:
        print(f"Outliers supprimés: {len(remove_outliers)} spectres exclus")

    # Calcul du périodogramme avec LombScargle
    # Définition de la grille de périodes
    min_period = 1.0  # 1 jour minimum
    max_period = (times.max() - times.min()) / 3  # 1/3 de la durée totale maximum

    # Grille logarithmique de périodes
    periods = np.logspace(np.log10(min_period), np.log10(max_period), 10000)

    # Calcul des fréquences correspondantes
    frequencies = 1.0 / periods

    print(f"Calcul du périodogramme pour {len(periods)} périodes...")

    # Calcul du périodogramme Lomb-Scargle
    ls = LombScargle(times, rv_values)
    power = ls.power(frequencies)

    print(f"Périodogramme calculé pour {len(periods)} périodes")
    print(f"Période min/max: {min_period:.1f} / {max_period:.1f} jours")

    return periods, power, rv_values, times


# =============================================================================
# UTILITY FUNCTIONS - Fonctions utilitaires
# =============================================================================


def get_transit_periods(star_name="STAR1136", data_root_dir="data"):
    """
    Récupère les périodes des planètes connues depuis le fichier Transit_information.csv

    Args:
        star_name: Nom de l'étoile (ex: "STAR1134", "STAR1136")

    Returns:
        list: Liste des périodes en jours, ou None si le fichier n'est pas trouvé
    """
    # Utiliser STAR1136 par défaut car c'est l'étoile configurée dans base_config.yaml
    data_path = f"{data_root_dir}/rv_datachallenge"
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


def analyze_periodogram_peaks(periods, power, threshold_ratio=0.1):
    """
    Analyse les pics du périodogramme pour identifier les périodes significatives.

    Args:
        periods: Périodes en jours
        power: Puissance du périodogramme
        threshold_ratio: Ratio du pic maximum pour définir le seuil (défaut 0.1 = 10%)

    Returns:
        list: Liste des dictionnaires contenant les informations des pics
    """
    threshold = threshold_ratio * np.max(power)
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

    return peaks_info


# =============================================================================
# HIGH-LEVEL ANALYSIS FUNCTIONS - Fonctions d'analyse haut niveau
# =============================================================================


def analyze_rv_periodogram_from_checkpoint(
    checkpoint_path="models/aestra_base_config_final.pth",
    star_name="STAR1136",
    save_path="reports/figures/rv_periodogram_analysis.png",
    show_plot=False,
    data_root_dir="data",  # Ajout pour la compatibilité avec les chemins de données
    remove_outliers=None,  # Nouvelle option pour supprimer les outliers
):
    """
    Fonction autonome pour calculer et tracer le périodogramme depuis un checkpoint.

    Args:
        checkpoint_path: Chemin vers le checkpoint du modèle
        star_name: Nom de l'étoile pour chercher les périodes connues
        save_path: Chemin pour sauvegarder la figure
        show_plot: Afficher la figure ou non
        data_root_dir: Répertoire racine des données pour charger le dataset
        remove_outliers: Liste des indices à supprimer (défaut [334, 464])

    Returns:
        dict: Dictionnaire contenant les résultats de l'analyse
    """

    print(f"Chargement du checkpoint: {checkpoint_path}")

    # Chargement du modèle et du dataset
    exp_data = load_experiment_checkpoint(checkpoint_path, data_root_dir=data_root_dir)
    model = exp_data["model"]
    dataset = exp_data["dataset"]
    cfg_name = exp_data["cfg_name"]
    start_epoch = exp_data["epoch"]
    current_phase = exp_data["current_phase"]

    print(f"Modèle chargé: {cfg_name}, epoch {start_epoch}, phase {current_phase}")
    print(f"Dataset: {len(dataset)} spectres, {dataset.n_pixels} pixels")

    # Récupération des périodes connues
    known_periods = get_transit_periods(star_name, data_root_dir=data_root_dir)

    # Calcul du périodogramme
    print("Calcul du périodogramme des vitesses radiales...")
    periods, power, rv_values, times = compute_rv_periodogram(
        model=model,
        dataset=dataset,
        batch_size=64,  # Ajustable selon la mémoire disponible
        remove_outliers=remove_outliers,
    )

    # Analyse des pics
    peaks_info = analyze_periodogram_peaks(periods, power, threshold_ratio=0.1)

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


def analyze_latent_space_from_checkpoint(
    checkpoint_path="models/aestra_base_config_final.pth",
    save_path="reports/figures/latent_space_3d_analysis.png",
    show_plot=False,
    data_root_dir="data",  # Ajout pour la compatibilité avec les chemins de données
    remove_outliers=None,  # Nouvelle option pour supprimer les outliers
):
    """
    Fonction autonome pour analyser l'espace latent depuis un checkpoint.

    Args:
        checkpoint_path: Chemin vers le checkpoint du modèle
        save_path: Chemin pour sauvegarder la figure
        show_plot: Afficher la figure ou non
        data_root_dir: Répertoire racine des données pour charger le dataset
        remove_outliers: Liste des indices à supprimer (défaut [334, 464])

    Returns:
        dict: Dictionnaire contenant les résultats de l'analyse
    """

    print(f"Chargement du checkpoint: {checkpoint_path}")

    # Chargement du modèle et du dataset
    exp_data = load_experiment_checkpoint(checkpoint_path, data_root_dir=data_root_dir)
    model = exp_data["model"]
    dataset = exp_data["dataset"]
    cfg_name = exp_data["cfg_name"]

    print(f"Modèle chargé: {cfg_name}")
    print(f"Dataset: {len(dataset)} spectres, {dataset.n_pixels} pixels")

    # Extraction des vecteurs latents et RV
    latent_s, rv_values = extract_latent_vectors_and_rv(
        model=model, dataset=dataset, batch_size=32, remove_outliers=remove_outliers
    )

    # Création du plot 3D si possible
    plot_created = plot_latent_space_3d(
        latent_s=latent_s,
        rv_values=rv_values,
        save_path=save_path,
        show_plot=show_plot,
    )

    # Retour des résultats
    results = {
        "latent_s": latent_s,
        "rv_values": rv_values,
        "latent_dim": latent_s.shape[1],
        "n_spectra": latent_s.shape[0],
        "plot_created": plot_created,
        "rv_stats": {
            "min": np.min(rv_values),
            "max": np.max(rv_values),
            "mean": np.mean(rv_values),
            "std": np.std(rv_values),
        },
    }

    return results


def full_analysis_from_checkpoint(
    checkpoint_path="models/aestra_base_config_final.pth",
    star_name="STAR1136",
    save_dir="reports/figures",
    show_plots=False,
    data_root_dir="data",  # Ajout pour la compatibilité avec les chemins de données
    remove_outliers=None,  # Nouvelle option pour supprimer les outliers
):
    """
    Fonction pour effectuer une analyse complète depuis un checkpoint.

    Args:
        checkpoint_path: Chemin vers le checkpoint du modèle
        star_name: Nom de l'étoile
        save_dir: Répertoire pour sauvegarder les figures
        show_plots: Afficher les figures ou non
        remove_outliers: Liste des indices à supprimer (défaut [334, 464])

    Returns:
        dict: Dictionnaire contenant tous les résultats de l'analyse
    """

    print("=" * 60)
    print("ANALYSE COMPLÈTE AESTRA")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Étoile: {star_name}")
    print(f"Répertoire de sauvegarde: {save_dir}")
    if remove_outliers:
        print(f"Outliers à supprimer: {remove_outliers}")

    # Chargement du modèle et du dataset
    exp_data = load_experiment_checkpoint(checkpoint_path, data_root_dir=data_root_dir)
    model = exp_data["model"]
    dataset = exp_data["dataset"]
    cfg_name = exp_data["cfg_name"]

    print(f"\nModèle chargé: {cfg_name}")
    print(f"Dataset: {len(dataset)} spectres, {dataset.n_pixels} pixels")

    os.makedirs(save_dir, exist_ok=True)

    results = {}

    # 1. Analyse des distances latentes (Figure 3)
    print("\n1. Calcul des distances latentes...")
    delta_s_rand, delta_s_aug = compute_latent_distances(
        model=model, dataset=dataset, batch_size=32
    )

    save_path = os.path.join(save_dir, "latent_distance_distribution.png")
    plot_latent_distance_distribution(
        delta_s_rand=delta_s_rand, delta_s_aug=delta_s_aug, save_path=save_path
    )

    results["latent_distances"] = {
        "delta_s_rand": delta_s_rand,
        "delta_s_aug": delta_s_aug,
        "mean_rand": np.mean(delta_s_rand),
        "mean_aug": np.mean(delta_s_aug),
    }

    # 2. Analyse de l'espace latent 3D
    print("\n2. Analyse de l'espace latent 3D...")
    latent_analysis = analyze_latent_space_from_checkpoint(
        checkpoint_path=checkpoint_path,
        save_path=os.path.join(save_dir, "latent_space_3d.png"),
        show_plot=show_plots,
        remove_outliers=remove_outliers,
    )
    results["latent_space"] = latent_analysis

    # 3. Analyse de la perturbation d'activité (Figure 2)
    print("\n3. Analyse de la perturbation d'activité...")
    y_act_original, y_act_perturbed_list, wavelength = compute_activity_perturbation(
        model=model, dataset=dataset, idx=0, perturbation_scale=1.0
    )

    save_path = os.path.join(save_dir, "activity_perturbation.png")
    plot_activity_perturbation(
        y_act_original=y_act_original,
        y_act_perturbed_list=y_act_perturbed_list,
        wavelength=wavelength,
        save_path=save_path,
        wave_range=(5000, 5050),
        show_plot=show_plots,
    )

    results["activity_perturbation"] = {
        "y_act_original": y_act_original,
        "y_act_perturbed_list": y_act_perturbed_list,
        "wavelength": wavelength,
    }

    # 4. Analyse du périodogramme RV
    print("\n4. Analyse du périodogramme des vitesses radiales...")
    periodogram_analysis = analyze_rv_periodogram_from_checkpoint(
        checkpoint_path=checkpoint_path,
        star_name=star_name,
        save_path=os.path.join(save_dir, "rv_periodogram.png"),
        show_plot=show_plots,
        data_root_dir=data_root_dir,
        remove_outliers=remove_outliers,
    )
    results["periodogram"] = periodogram_analysis

    # Résumé
    print("\n" + "=" * 60)
    print("RÉSUMÉ DE L'ANALYSE")
    print("=" * 60)
    print(f"Dimension latente: {results['latent_space']['latent_dim']}D")
    print(f"Nombre de spectres: {results['latent_space']['n_spectra']}")
    print(
        f"Plage RV: {results['latent_space']['rv_stats']['min']:.2f} à {results['latent_space']['rv_stats']['max']:.2f}"
    )
    print(
        f"Distance latente moyenne (paires aléatoires): {results['latent_distances']['mean_rand']:.3e}"
    )
    print(
        f"Distance latente moyenne (paires augmentées): {results['latent_distances']['mean_aug']:.3e}"
    )
    print(
        f"Meilleure période détectée: {results['periodogram']['best_period']:.2f} jours"
    )
    if results["periodogram"]["known_periods"] is not None:
        print(f"Périodes connues: {results['periodogram']['known_periods']}")
    print("=" * 60)

    return results


# =============================================================================
# MAIN EXECUTION - Exécution principale
# =============================================================================


if __name__ == "__main__":
    # Exemple d'utilisation - analyse complète
    checkpoint_path = "models/aestra_colab_config_final.pth"

    # Analyse complète avec suppression des outliers par défaut
    results = full_analysis_from_checkpoint(
        checkpoint_path=checkpoint_path,
        star_name="STAR1136",
        save_dir="reports/figures",
        show_plots=True,
        remove_outliers=[334, 464],  # Suppression des outliers par défaut
    )

    print("\n✅ Analyse terminée. Résultats sauvegardés dans reports/figures/")
    print(
        f"Meilleure période détectée: {results['periodogram']['best_period']:.2f} jours"
    )
    print(f"Dimension de l'espace latent: {results['latent_space']['latent_dim']}D")
    print(f"Nombre de spectres analysés: {results['latent_space']['n_spectra']}")


# =============================================================================
# USAGE EXAMPLES - Exemples d'utilisation
# =============================================================================

"""
Exemples d'utilisation des fonctions:

# 1. Analyse complète depuis un checkpoint avec suppression des outliers
results = full_analysis_from_checkpoint(
    checkpoint_path="models/aestra_base_config_final.pth",
    star_name="STAR1136",
    save_dir="mon_analyse",
    show_plots=True,
    remove_outliers=[334, 464]  # Suppression des outliers par défaut
)

# 2. Analyse complète sans suppression d'outliers
results = full_analysis_from_checkpoint(
    checkpoint_path="models/aestra_base_config_final.pth",
    star_name="STAR1136",
    save_dir="mon_analyse",
    show_plots=True,
    remove_outliers=[]  # Aucun outlier à supprimer
)

# 3. Analyse seulement du périodogramme avec outliers supprimés
periodogram_results = analyze_rv_periodogram_from_checkpoint(
    checkpoint_path="models/aestra_base_config_final.pth",
    save_path="mon_periodogramme.png",
    show_plot=True,
    remove_outliers=[334, 464]
)

# 4. Analyse seulement de l'espace latent avec outliers supprimés
latent_results = analyze_latent_space_from_checkpoint(
    checkpoint_path="models/aestra_base_config_final.pth",
    save_path="mon_espace_latent.png",
    show_plot=True,
    remove_outliers=[334, 464]
)

# 5. Utilisation modulaire des fonctions individuelles avec outliers supprimés
exp_data = load_experiment_checkpoint("models/aestra_base_config_final.pth")
model, dataset = exp_data["model"], exp_data["dataset"]

# Calcul des distances latentes (pas d'option remove_outliers pour cette fonction)
delta_s_rand, delta_s_aug = compute_latent_distances(model, dataset)

# Extraction des vecteurs latents et RV avec suppression d'outliers
latent_s, rv_values = extract_latent_vectors_and_rv(
    model, dataset, remove_outliers=[334, 464]
)

# Calcul du périodogramme avec suppression d'outliers
periods, power, rv_values, times = compute_rv_periodogram(
    model, dataset, remove_outliers=[334, 464]
)

# Périodes connues
known_periods = get_transit_periods("STAR1136")
"""
