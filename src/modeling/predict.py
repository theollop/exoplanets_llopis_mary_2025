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
import matplotlib.pyplot as plt


def detect_experiment_directories_from_checkpoint(checkpoint_path):
    """
    Détecte automatiquement les répertoires d'expérience à partir du chemin du checkpoint.

    Args:
        checkpoint_path: Chemin vers le checkpoint (ex: "experiments/mon_exp/models/model.pth")

    Returns:
        dict: Dictionnaire des répertoires d'expérience ou None si non détecté
    """
    import os

    # Normaliser le chemin
    checkpoint_path = os.path.normpath(checkpoint_path)
    path_parts = checkpoint_path.split(os.sep)

    # Chercher le pattern: .../experiments/experiment_name/models/...
    try:
        if "experiments" in path_parts and "models" in path_parts:
            exp_idx = path_parts.index("experiments")
            models_idx = path_parts.index("models")

            if models_idx == exp_idx + 2:  # experiments/exp_name/models
                # Reconstruire le chemin de base de l'expérience
                base_exp_path = os.sep.join(path_parts[:models_idx])

                # Définir les répertoires d'expérience
                exp_dirs = {
                    "base": base_exp_path,
                    "models": os.path.join(base_exp_path, "models"),
                    "figures": os.path.join(base_exp_path, "figures"),
                    "spectra": os.path.join(base_exp_path, "spectra"),
                    "logs": os.path.join(base_exp_path, "logs"),
                }

                print("🔍 Répertoires d'expérience détectés automatiquement:")
                print(f"   Base: {exp_dirs['base']}")
                print(f"   Figures: {exp_dirs['figures']}")

                return exp_dirs
    except (ValueError, IndexError):
        pass

    print(
        f"⚠️  Impossible de détecter les répertoires d'expérience depuis: {checkpoint_path}"
    )
    print("   Utilisation des chemins par défaut dans reports/figures/")
    return None


def plot_rv_periodogram_juxtaposed(
    periods,
    power,
    rv_values,
    times,
    known_periods=None,
    save_path="periodogram.png",
    show_plot=False,
    periods_corr=None,
    power_corr=None,
    rv_corrected=None,
    decorrelate_applied=False,
):
    """
    Crée un plot de périodogramme selon le layout souhaité :
    - En haut : Périodogramme complet avec courbes normale ET décorrélée superposées
    - Milieu : Deux séries temporelles côte à côte (RV normale gauche, RV décorrélée droite)
    - En bas : 3 zooms planètes avec courbes normale ET décorrélée superposées

    Args:
        periods: Périodes pour le périodogramme
        power: Puissance du périodogramme (RV originales)
        rv_values: Valeurs des vitesses radiales originales
        times: Temps des observations
        known_periods: Liste des périodes connues des planètes
        save_path: Chemin de sauvegarde
        show_plot: Afficher ou non le plot
        periods_corr: Périodes pour le périodogramme décorrélé
        power_corr: Puissance du périodogramme décorrélé
        rv_corrected: Valeurs des vitesses radiales décorrélées
        decorrelate_applied: Si la décorrélation a été appliquée
    """
    if decorrelate_applied and rv_corrected is not None and power_corr is not None:
        # Layout personnalisé selon la demande
        if known_periods is not None and len(known_periods) > 0:
            fig = plt.figure(figsize=(18, 14))

            # === TOUT EN HAUT : PÉRIODOGRAMME COMPLET AVEC LES DEUX COURBES ===
            ax1 = plt.subplot(3, 3, (1, 3))  # Occupe toute la première ligne
            ax1.semilogx(periods, power, "b-", linewidth=1.0, label="RV originales")
            ax1.semilogx(
                periods_corr, power_corr, "r-", linewidth=1.0, label="RV décorrélées"
            )
            ax1.set_ylabel("Puissance LS")
            ax1.set_title("Périodogramme Lomb-Scargle : Normal vs Décorrélé")
            ax1.grid(True, alpha=0.3)

            # Marquer les périodes connues
            for i, period in enumerate(known_periods):
                if periods.min() <= period <= periods.max():
                    ax1.axvline(
                        period,
                        color="orange",
                        linestyle="--",
                        alpha=0.8,
                        linewidth=2,
                        label=f"Planète {i + 1}: {period:.1f}d" if i < 3 else None,
                    )
            ax1.legend()

            # === MILIEU : DEUX SÉRIES TEMPORELLES CÔTE À CÔTE ===

            # Calcul d'une échelle Y commune pour une meilleure comparaison
            all_rv_values = np.concatenate([rv_values, rv_corrected])
            y_min = np.min(all_rv_values) * 1.1
            y_max = np.max(all_rv_values) * 1.1

            # Série temporelle RV originales (gauche, centré)
            ax2 = plt.subplot(3, 3, 4)
            ax2.plot(times, rv_values, "bo-", markersize=1.5, linewidth=0.5)
            ax2.set_ylabel("RV originales (m/s)")
            ax2.set_title(f"RV originales\n(std = {np.std(rv_values):.4f} m/s)")
            ax2.set_ylim(y_min, y_max)
            ax2.grid(True, alpha=0.3)

            # Série temporelle RV décorrélées (droite, centré)
            ax3 = plt.subplot(3, 3, 5)
            ax3.plot(times, rv_corrected, "ro-", markersize=1.5, linewidth=0.5)
            ax3.set_ylabel("RV décorrélées (m/s)")
            ax3.set_title(f"RV décorrélées\n(std = {np.std(rv_corrected):.4f} m/s)")
            ax3.set_ylim(y_min, y_max)
            ax3.grid(True, alpha=0.3)

            # === EN BAS : 3 ZOOMS PLANÈTES AVEC LES DEUX COURBES ===
            zoom_positions = [7, 8, 9]  # Dernière ligne
            for i, period in enumerate(known_periods[:3]):
                if periods.min() <= period <= periods.max() and i < 3:
                    ax_zoom = plt.subplot(3, 3, zoom_positions[i])

                    zoom_factor = 0.2  # ±20% autour de la période
                    period_min = period * (1 - zoom_factor)
                    period_max = period * (1 + zoom_factor)

                    # Masques pour les zooms
                    zoom_mask = (periods >= period_min) & (periods <= period_max)
                    zoom_mask_corr = (periods_corr >= period_min) & (
                        periods_corr <= period_max
                    )

                    if np.any(zoom_mask) and np.any(zoom_mask_corr):
                        # Courbes normale ET décorrélée sur le même zoom
                        ax_zoom.plot(
                            periods[zoom_mask],
                            power[zoom_mask],
                            "b-",
                            linewidth=1.5,
                            label="Normal",
                        )
                        ax_zoom.plot(
                            periods_corr[zoom_mask_corr],
                            power_corr[zoom_mask_corr],
                            "r-",
                            linewidth=1.5,
                            label="Décorrélé",
                        )

                        # Ligne de la période théorique
                        ax_zoom.axvline(
                            period,
                            color="orange",
                            linestyle="--",
                            alpha=0.8,
                            linewidth=2,
                        )

                        ax_zoom.set_xlabel("Période (jours)")
                        ax_zoom.set_ylabel("Puissance LS")
                        ax_zoom.set_title(f"Planète {i + 1}: {period:.1f}d")
                        ax_zoom.grid(True, alpha=0.3)
                        ax_zoom.legend(fontsize=8)

                        # Trouver et marquer les pics locaux
                        if len(periods[zoom_mask]) > 0:
                            local_max_idx = np.argmax(power[zoom_mask])
                            local_max_period = periods[zoom_mask][local_max_idx]
                            local_max_power = power[zoom_mask][local_max_idx]
                            ax_zoom.plot(
                                local_max_period, local_max_power, "bo", markersize=6
                            )

                        if len(periods_corr[zoom_mask_corr]) > 0:
                            local_max_idx_corr = np.argmax(power_corr[zoom_mask_corr])
                            local_max_period_corr = periods_corr[zoom_mask_corr][
                                local_max_idx_corr
                            ]
                            local_max_power_corr = power_corr[zoom_mask_corr][
                                local_max_idx_corr
                            ]
                            ax_zoom.plot(
                                local_max_period_corr,
                                local_max_power_corr,
                                "ro",
                                markersize=6,
                            )

        else:
            # Layout simple sans zooms
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))

            # Périodogramme combiné
            ax1.semilogx(periods, power, "b-", linewidth=0.8, label="RV originales")
            ax1.semilogx(
                periods_corr, power_corr, "r-", linewidth=0.8, label="RV décorrélées"
            )
            ax1.set_ylabel("Puissance LS")
            ax1.set_title("Périodogramme : Normal vs Décorrélé")
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # Séries temporelles
            ax2.plot(times, rv_values, "bo-", markersize=3, linewidth=0.5)
            ax2.set_ylabel("RV originales (m/s/s)")
            ax2.set_title(f"RV originales (std = {np.std(rv_values):.4f} m/s/s)")
            ax2.grid(True, alpha=0.3)

            ax3.plot(times, rv_corrected, "ro-", markersize=3, linewidth=0.5)
            ax3.set_xlabel("JDB")
            ax3.set_ylabel("RV décorrélées (m/s/s)")
            ax3.set_title(f"RV décorrélées (std = {np.std(rv_corrected):.4f} m/s/s)")
            ax3.grid(True, alpha=0.3)

            # Espace vide ou analyse supplémentaire
            ax4.text(
                0.5,
                0.5,
                "Comparaison\nNormal vs Décorrélé",
                ha="center",
                va="center",
                transform=ax4.transAxes,
                fontsize=14,
            )
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis("off")

    else:
        # Fallback vers la fonction existante si pas de décorrélation
        plot_rv_periodogram(
            periods=periods,
            power=power,
            rv_values=rv_values,
            times=times,
            known_periods=known_periods,
            save_path=save_path,
            show_plot=show_plot,
        )
        return

    plt.tight_layout()

    # Sauvegarde
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Périodogramme juxtaposé sauvegardé: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


# =============================================================================
# ANALYSIS FUNCTIONS - Fonctions d'analyse principales
# =============================================================================


def compute_latent_distances(model, dataset, batch_size=4):
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


def extract_latent_vectors_and_rv(
    model, dataset, batch_size=32, remove_outliers=None, decorrelate_rv=False
):
    """
    Extrait les vecteurs latents et calcule les vitesses radiales correspondantes.
    Version optimisée par batches pour éviter les OOM errors.

    Args:
        model: Modèle AESTRA entraîné
        dataset: Dataset contenant les spectres
        batch_size: Taille des batches pour le traitement (défaut 32)
        remove_outliers: Liste des indices à supprimer (défaut [334, 464])
        decorrelate_rv: Si True, applique la décorrélation des RV avec l'espace latent

    Returns:
        tuple: (latent_s, rv_values, rv_corrected, rv_bias, sigma_R)
            - latent_s: Vecteurs latents (N, D)
            - rv_values: Valeurs RV originales (N,)
            - rv_corrected: RV décorrélées (None si decorrelate_rv=False)
            - rv_bias: Biais estimés (None si decorrelate_rv=False)
            - sigma_R: Rayon caractéristique (None si decorrelate_rv=False)
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

    # Application de la décorrélation si demandée
    rv_corrected = None
    rv_bias = None
    sigma_R = None

    if decorrelate_rv:
        print("\n🔄 Application de la décorrélation des vitesses radiales...")
        rv_corrected, rv_bias, sigma_R = decorrelate_rv_with_latent_space(
            latent_s=latent_s,
            rv_values=rv_values,
            min_neighbors=10,
            neighbor_fraction=0.05,
        )
        print("✅ Décorrélation terminée")

    return latent_s, rv_values, rv_corrected, rv_bias, sigma_R


def compute_rv_periodogram(
    model,
    dataset,
    batch_size=4,
    remove_outliers=None,
    decorrelate_rv=False,
):
    """
    Calcule le périodogramme des vitesses radiales en utilisant LombScargle.
    Version optimisée par batches pour éviter les OOM errors.

    Args:
        model: Modèle AESTRA entraîné
        dataset: Dataset contenant les spectres et les temps JDB
        batch_size: Taille des batches pour le traitement (défaut 64)
        remove_outliers: Liste des indices à supprimer (défaut [334, 464])
        decorrelate_rv: Si True, applique la décorrélation des RV avec l'espace latent

    Returns:
        tuple: (periods, power, rv_values, times, periods_corr, power_corr, rv_corrected)
            - periods: Périodes en jours (original)
            - power: Puissance du périodogramme (original)
            - rv_values: Valeurs RV originales
            - times: Temps JDB correspondants
            - periods_corr: Périodes pour RV décorrélées (None si decorrelate_rv=False)
            - power_corr: Puissance pour RV décorrélées (None si decorrelate_rv=False)
            - rv_corrected: RV décorrélées (None si decorrelate_rv=False)
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

    # Calcul du périodogramme Lomb-Scargle pour les RV originales
    ls = LombScargle(times, rv_values)
    power = ls.power(frequencies)

    print(f"Périodogramme calculé pour {len(periods)} périodes")
    print(f"Période min/max: {min_period:.1f} / {max_period:.1f} jours")

    # Initialisation des variables pour les RV décorrélées
    periods_corr = None
    power_corr = None
    rv_corrected = None

    # Application de la décorrélation si demandée
    if decorrelate_rv:
        print("\n🔄 Application de la décorrélation des vitesses radiales...")

        # D'abord, nous devons récupérer les vecteurs latents
        # Extraction des vecteurs latents pour la décorrélation
        latent_vectors = []

        with torch.no_grad():
            for i in range(0, len(valid_indices), batch_size):
                end_idx = min(i + batch_size, len(valid_indices))
                batch_indices = valid_indices[i:end_idx]

                batch_spectra = dataset.spectra[batch_indices]
                batch_robs = batch_spectra - model.b_obs.unsqueeze(0)
                _, latent_s_batch = model.spender(batch_robs)

                latent_vectors.append(latent_s_batch.cpu())

                if (i // batch_size) % 10 == 0:
                    clear_gpu_memory()

        latent_s = torch.cat(latent_vectors, dim=0).numpy()

        # Application de la décorrélation
        rv_corrected, rv_bias, sigma_R = decorrelate_rv_with_latent_space(
            latent_s=latent_s,
            rv_values=rv_values,
            min_neighbors=10,
            neighbor_fraction=0.05,
        )

        # Calcul du périodogramme pour les RV décorrélées
        print("Calcul du périodogramme pour les RV décorrélées...")
        ls_corr = LombScargle(times, rv_corrected)
        power_corr = ls_corr.power(frequencies)
        periods_corr = periods  # Même grille de périodes

        print("✅ Décorrélation et périodogramme corrigé terminés")

    return periods, power, rv_values, times, periods_corr, power_corr, rv_corrected


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


def decorrelate_rv_with_latent_space(
    latent_s, rv_values, min_neighbors=10, neighbor_fraction=0.05
):
    """
    Décorrèle les vitesses radiales en utilisant un lissage gaussien dans l'espace latent.

    Implémente la méthode décrite dans le papier AESTRA (équations 10-11) :
    - vencode,i = vtrue,i + v0(si) + noise
    - v0(si) ≈ ⟨v⟩i = Σ(wij * vencode,j) / Σ(wij)
    - wij = exp(-|si - sj|²/(2σR²))

    Args:
        latent_s: Vecteurs latents (N, D) - positions dans l'espace latent
        rv_values: Vitesses radiales encodées (N,)
        min_neighbors: Nombre minimum de voisins à utiliser (défaut 10)
        neighbor_fraction: Fraction des voisins les plus proches à utiliser (défaut 0.05 = 5%)

    Returns:
        tuple: (rv_corrected, rv_bias, sigma_R)
            - rv_corrected: Vitesses radiales décorrélées
            - rv_bias: Biais v0 estimé pour chaque point
            - sigma_R: Rayon caractéristique de la distribution latente
    """
    n_spectra = len(rv_values)

    print(f"Décorrélation des RV avec l'espace latent ({n_spectra} spectres)...")

    # Calcul du rayon caractéristique σR comme distance moyenne aux 10 plus proches voisins
    print("Calcul du rayon caractéristique σR...")

    # Calcul des distances entre tous les points dans l'espace latent
    distances = np.zeros((n_spectra, n_spectra))
    for i in range(n_spectra):
        for j in range(n_spectra):
            if i != j:
                distances[i, j] = np.linalg.norm(latent_s[i] - latent_s[j])

    # Calcul de σR comme distance moyenne aux 10 plus proches voisins
    k_neighbors = min(10, n_spectra - 1)
    sigma_R_values = []

    for i in range(n_spectra):
        # Trier les distances pour ce point (exclure la distance à soi-même = 0)
        sorted_distances = np.sort(distances[i][distances[i] > 0])
        # Prendre les k plus proches voisins
        nearest_distances = sorted_distances[:k_neighbors]
        sigma_R_values.append(np.mean(nearest_distances))

    sigma_R = np.mean(sigma_R_values)
    print(f"Rayon caractéristique σR = {sigma_R:.4f}")

    # Calcul du nombre de voisins à utiliser
    n_neighbors = max(min_neighbors, int(neighbor_fraction * n_spectra))
    print(f"Utilisation de {n_neighbors} voisins les plus proches pour chaque point")

    # Calcul des biais v0 pour chaque point
    rv_bias = np.zeros(n_spectra)

    for i in range(n_spectra):
        # Trouver les voisins les plus proches dans l'espace latent
        neighbor_indices = np.argsort(distances[i])[
            : n_neighbors + 1
        ]  # +1 pour exclure le point lui-même
        neighbor_indices = neighbor_indices[neighbor_indices != i][
            :n_neighbors
        ]  # Exclure le point i

        # Calcul des poids gaussiens
        weights = np.zeros(len(neighbor_indices))
        weighted_rvs = 0.0
        total_weight = 0.0

        for j_idx, j in enumerate(neighbor_indices):
            # Distance dans l'espace latent
            dist_latent = distances[i, j]

            # Filtre : seulement les voisins dans un rayon de 3σR
            if dist_latent <= 3 * sigma_R:
                # Poids gaussien
                weight = np.exp(-(dist_latent**2) / (2 * sigma_R**2))
                weights[j_idx] = weight

                weighted_rvs += weight * rv_values[j]
                total_weight += weight

        # Calcul du biais v0(si) = moyenne pondérée des RV des voisins
        if total_weight > 0:
            rv_bias[i] = weighted_rvs / total_weight
        else:
            rv_bias[i] = 0.0  # Pas de voisins dans le rayon, pas de correction

    # Correction des vitesses radiales : vcorrect = vencode - v0
    rv_corrected = rv_values - rv_bias

    print("Décorrélation terminée:")
    print(f"  - RV avant correction: {rv_values.std():.4f} (std)")
    print(f"  - RV après correction: {rv_corrected.std():.4f} (std)")
    print(f"  - Biais moyen |v0|: {np.mean(np.abs(rv_bias)):.4f}")
    print(f"  - Biais max |v0|: {np.max(np.abs(rv_bias)):.4f}")

    return rv_corrected, rv_bias, sigma_R


def analyze_activity_signals_mosaic(
    model,
    dataset,
    n_spectra=4,
    n_lines=4,
    data_root_dir="data",
    save_path=None,  # Sera déterminé automatiquement si possible
    show_plot=False,
    remove_outliers=None,
    line_window=0.4,  # Fenêtre beaucoup plus serrée autour de chaque raie (0.4 Å au lieu de 2.0 Å)
    spectrum_indices=None,  # Indices des spectres à analyser (optionnel)
):
    """
    Analyse les signaux d'activité prédits pour les raies les plus importantes.
    Crée une mosaïque montrant les raies observées vs l'activité prédite.

    Args:
        model: Modèle AESTRA entraîné
        dataset: Dataset contenant les spectres
        n_spectra: Nombre de spectres à analyser (défaut 4)
        n_lines: Nombre de raies à analyser par spectre (défaut 4)
        data_root_dir: Répertoire racine des données
        save_path: Chemin pour sauvegarder la figure (None pour utiliser le répertoire par défaut)
        show_plot: Afficher la figure ou non
        remove_outliers: Liste des indices à supprimer
        line_window: Fenêtre autour de chaque raie en Angstroms (défaut 0.4 Å pour zoom serré)
        spectrum_indices: Indices spécifiques des spectres à analyser (optionnel)

    Returns:
        dict: Dictionnaire contenant les résultats de l'analyse
    """
    model.eval()

    if remove_outliers is None:
        remove_outliers = [334, 464]

    print("Analyse des signaux d'activité pour les raies importantes...")

    # Détermination automatique du chemin de sauvegarde si non fourni
    if save_path is None:
        save_path = "reports/figures/activity_signals_mosaic.png"

    # Création d'un masque pour exclure les indices outliers
    n_specs = len(dataset)
    valid_indices = [i for i in range(n_specs) if i not in remove_outliers]

    # Chargement du masque G2 pour obtenir les positions et poids des raies
    try:
        g2_mask = np.loadtxt(f"{data_root_dir}/rv_datachallenge/masks/G2_mask.txt")
        line_positions = g2_mask[:, 0]  # Première colonne : positions des raies
        line_weights = g2_mask[:, 1]  # Deuxième colonne : poids des raies
    except Exception as e:
        print(f"❌ Erreur lors du chargement du masque G2: {e}")
        return None

    # Récupération de la grille de longueurs d'onde
    wavegrid = dataset.wavegrid.cpu().numpy()

    # Filtrage des raies dans la gamme spectrale du dataset
    valid_lines_mask = (line_positions >= wavegrid.min()) & (
        line_positions <= wavegrid.max()
    )
    valid_line_positions = line_positions[valid_lines_mask]
    valid_line_weights = line_weights[valid_lines_mask]

    # Sélection des raies avec les poids les plus importants
    top_lines_indices = np.argsort(valid_line_weights)[::-1][
        : n_lines * 2
    ]  # Prendre plus que nécessaire
    selected_line_positions = valid_line_positions[top_lines_indices[:n_lines]]
    selected_line_weights = valid_line_weights[top_lines_indices[:n_lines]]

    print("Raies sélectionnées (position, poids):")
    for i, (pos, weight) in enumerate(
        zip(selected_line_positions, selected_line_weights)
    ):
        print(f"  {i + 1}. {pos:.3f} Å (poids: {weight:.3f})")

    # Sélection des spectres à analyser
    if spectrum_indices is not None:
        # Utiliser les indices fournis (en filtrant les outliers)
        selected_spectrum_indices = [
            idx for idx in spectrum_indices if idx in valid_indices
        ][:n_spectra]
    else:
        # Sélection aléatoire de spectres
        selected_spectrum_indices = np.random.choice(
            valid_indices, size=min(n_spectra, len(valid_indices)), replace=False
        )

    print(f"Spectres sélectionnés: {selected_spectrum_indices}")

    # Analyse des spectres sélectionnés
    results = {
        "selected_spectra": selected_spectrum_indices,
        "selected_lines": selected_line_positions,
        "line_weights": selected_line_weights,
        "analysis_data": [],
    }

    with torch.no_grad():
        for spec_idx in selected_spectrum_indices:
            # Extraction du spectre
            spectrum = dataset.spectra[spec_idx].unsqueeze(0)

            # Calcul des composantes AESTRA
            batch_robs = spectrum - model.b_obs.unsqueeze(0)
            batch_yact, batch_s = model.spender(batch_robs)

            # Conversion en numpy
            observed_spectrum = spectrum.squeeze(0).cpu().numpy()
            activity_spectrum = batch_yact.squeeze(0).cpu().numpy()

            # Stockage des données pour ce spectre
            spectrum_data = {
                "spectrum_index": spec_idx,
                "observed": observed_spectrum,
                "activity": activity_spectrum,
                "latent_vector": batch_s.squeeze(0).cpu().numpy(),
                "line_analysis": [],
            }

            # Analyse de chaque raie pour ce spectre
            for line_pos, line_weight in zip(
                selected_line_positions, selected_line_weights
            ):
                # Création du masque pour la fenêtre autour de la raie
                line_mask = (wavegrid >= line_pos - line_window / 2) & (
                    wavegrid <= line_pos + line_window / 2
                )

                if np.any(line_mask):
                    wave_window = wavegrid[line_mask]
                    obs_window = observed_spectrum[line_mask]
                    act_window = activity_spectrum[line_mask]

                    # Calcul de statistiques pour cette raie
                    activity_amplitude = np.max(act_window) - np.min(act_window)
                    activity_rms = np.sqrt(np.mean(act_window**2))

                    line_data = {
                        "line_position": line_pos,
                        "line_weight": line_weight,
                        "wavelength": wave_window,
                        "observed": obs_window,
                        "activity": act_window,
                        "activity_amplitude": activity_amplitude,
                        "activity_rms": activity_rms,
                    }

                    spectrum_data["line_analysis"].append(line_data)

            results["analysis_data"].append(spectrum_data)

    # Création de la mosaïque de plots
    _create_activity_mosaic_plot(results, save_path, show_plot)

    # Nettoyage mémoire
    clear_gpu_memory()

    print(
        f"✅ Analyse des signaux d'activité terminée: {len(selected_spectrum_indices)} spectres, {n_lines} raies"
    )
    return results


def _create_activity_mosaic_plot(results, save_path, show_plot):
    """
    Fonction auxiliaire pour créer la mosaïque de plots des signaux d'activité.
    """
    import matplotlib.pyplot as plt

    n_spectra = len(results["analysis_data"])
    n_lines = len(results["selected_lines"])

    # Configuration de la figure : 2 colonnes par raie x n_spectra lignes
    fig_width = 5 * n_lines  # 5 inches par paire de raies (réduit pour zoom serré)
    fig_height = 3.5 * n_spectra  # 3.5 inches par spectre (réduit aussi)
    fig, axes = plt.subplots(n_spectra, n_lines * 2, figsize=(fig_width, fig_height))

    # S'assurer que axes est toujours 2D
    if n_spectra == 1:
        axes = axes.reshape(1, -1)
    if n_lines == 1:
        axes = axes.reshape(-1, 2)

    fig.suptitle(
        f"Signaux d'activité - Zoom serré sur les raies importantes\n"
        f"{n_spectra} spectres × {n_lines} raies (fenêtre ≈±0.2 Å)",
        fontsize=16,
        fontweight="bold",
    )

    for spec_row, spectrum_data in enumerate(results["analysis_data"]):
        spec_idx = spectrum_data["spectrum_index"]

        for line_col, line_data in enumerate(spectrum_data["line_analysis"]):
            # Colonnes pour cette raie (observé et activité)
            col_obs = line_col * 2  # Colonne gauche : spectre observé
            col_act = line_col * 2 + 1  # Colonne droite : activité prédite

            line_pos = line_data["line_position"]
            wavelength = line_data["wavelength"]
            observed = line_data["observed"]
            activity = line_data["activity"]
            line_weight = line_data["line_weight"]

            # Plot du spectre observé (gauche)
            ax_obs = axes[spec_row, col_obs]
            ax_obs.plot(wavelength, observed, "b-", linewidth=1.5, alpha=0.8)
            ax_obs.axvline(
                line_pos, color="red", linestyle="--", alpha=0.7, linewidth=1
            )
            ax_obs.set_title(
                f"Observé - Spec {spec_idx}\nRaie {line_pos:.1f}Å (w={line_weight:.2f})",
                fontsize=10,
            )
            ax_obs.set_ylabel("Flux normalisé", fontsize=9)
            ax_obs.grid(True, alpha=0.3)

            # Plot de l'activité prédite (droite)
            ax_act = axes[spec_row, col_act]
            ax_act.plot(wavelength, activity, "orange", linewidth=1.5, alpha=0.8)
            ax_act.axvline(
                line_pos, color="red", linestyle="--", alpha=0.7, linewidth=1
            )
            ax_act.axhline(y=0, color="black", linestyle="-", alpha=0.5, linewidth=0.5)
            ax_act.set_title(
                f"Activité prédite\nAmp={line_data['activity_amplitude']:.4f}",
                fontsize=10,
            )
            ax_act.set_ylabel("Flux d'activité", fontsize=9)
            ax_act.grid(True, alpha=0.3)

            # Configuration de l'axe x pour la dernière ligne
            if spec_row == n_spectra - 1:
                ax_obs.set_xlabel("Longueur d'onde (Å)", fontsize=9)
                ax_act.set_xlabel("Longueur d'onde (Å)", fontsize=9)

    plt.tight_layout()

    # Sauvegarde
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Mosaïque des signaux d'activité sauvegardée: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


# =============================================================================
# LATENT PERIODOGRAM ANALYSIS - Analyse du périodogramme des composantes latentes
# =============================================================================


def compute_latent_periodogram(
    latent_vectors, times, component_indices=None, max_period_factor=3
):
    """
    Calcule le périodogramme pour chaque composante du vecteur latent.

    Args:
        latent_vectors: Vecteurs latents (N, D) où N=nombre de spectres, D=dimension latente
        times: Temps correspondants (N,)
        component_indices: Indices des composantes à analyser (None pour toutes)
        max_period_factor: Facteur pour la période maximum (défaut 3 = 1/3 de la durée totale)

    Returns:
        dict: Dictionnaire contenant periods, power_matrix, component_indices, times
    """
    n_spectra, latent_dim = latent_vectors.shape

    if component_indices is None:
        component_indices = list(range(latent_dim))
    else:
        component_indices = [i for i in component_indices if 0 <= i < latent_dim]

    print(
        f"Calcul du périodogramme pour {len(component_indices)} composantes latentes..."
    )

    # Définition de la grille de périodes (identique à compute_rv_periodogram)
    min_period = 1.0  # 1 jour minimum
    max_period = (times.max() - times.min()) / max_period_factor
    periods = np.logspace(np.log10(min_period), np.log10(max_period), 10000)
    frequencies = 1.0 / periods

    # Calcul du périodogramme pour chaque composante
    power_matrix = np.zeros((len(component_indices), len(periods)))

    for i, comp_idx in enumerate(component_indices):
        component_values = latent_vectors[:, comp_idx]

        # Calcul du périodogramme Lomb-Scargle
        ls = LombScargle(times, component_values)
        power = ls.power(frequencies)
        power_matrix[i, :] = power

        if (i + 1) % 10 == 0 or i == 0 or i == len(component_indices) - 1:
            print(f"  Composante {comp_idx}: périodogramme calculé")

    print(f"✅ Périodogrammes calculés pour {len(component_indices)} composantes")

    return {
        "periods": periods,
        "power_matrix": power_matrix,
        "component_indices": component_indices,
        "times": times,
        "latent_vectors": latent_vectors,
    }


def analyze_latent_periodogram_peaks(
    periodogram_data, known_periods=None, threshold_ratio=0.1, top_n_components=None
):
    """
    Analyse les pics significatifs dans les périodogrammes des composantes latentes.

    Args:
        periodogram_data: Résultat de compute_latent_periodogram
        known_periods: Périodes connues des planètes pour comparaison
        threshold_ratio: Ratio du pic maximum pour définir le seuil (défaut 0.1)
        top_n_components: Nombre de composantes les plus significatives à retourner

    Returns:
        dict: Analyse des pics pour chaque composante
    """
    periods = periodogram_data["periods"]
    power_matrix = periodogram_data["power_matrix"]
    component_indices = periodogram_data["component_indices"]

    results = {
        "component_analysis": {},
        "ranking": [],
        "known_periods": known_periods,
        "summary": {},
    }

    print("Analyse des pics significatifs...")

    for i, comp_idx in enumerate(component_indices):
        power = power_matrix[i, :]
        max_power = np.max(power)
        threshold = threshold_ratio * max_power

        # Identification des pics
        peak_indices = np.where(power > threshold)[0]

        if len(peak_indices) > 0:
            peak_periods = periods[peak_indices]
            peak_powers = power[peak_indices]

            # Tri par puissance décroissante
            sorted_indices = np.argsort(peak_powers)[::-1][:10]  # Top 10 pics

            component_peaks = []
            for j, idx in enumerate(sorted_indices):
                period = peak_periods[idx]
                power_val = peak_powers[idx]

                # Recherche de correspondance avec périodes connues
                closest_known = None
                if known_periods is not None:
                    distances = np.abs(known_periods - period)
                    closest_idx = np.argmin(distances)
                    closest_period = known_periods[closest_idx]
                    relative_diff = np.abs(period - closest_period) / closest_period

                    if relative_diff < 0.1:  # Tolérance de 10%
                        closest_known = {
                            "period": closest_period,
                            "difference": period - closest_period,
                            "relative_diff": relative_diff,
                        }

                component_peaks.append(
                    {
                        "rank": j + 1,
                        "period": period,
                        "power": power_val,
                        "relative_power": power_val / max_power,
                        "closest_known": closest_known,
                    }
                )

            results["component_analysis"][comp_idx] = {
                "max_power": max_power,
                "n_peaks": len(peak_indices),
                "peaks": component_peaks,
                "significance": max_power,  # Score de significance global
            }

            # Ajout au ranking global
            results["ranking"].append(
                {
                    "component": comp_idx,
                    "max_power": max_power,
                    "best_period": periods[np.argmax(power)],
                    "n_significant_peaks": len(component_peaks),
                }
            )

    # Tri du ranking par puissance maximum
    results["ranking"] = sorted(
        results["ranking"], key=lambda x: x["max_power"], reverse=True
    )

    # Sélection des top composantes si demandé
    if top_n_components is not None:
        results["top_components"] = results["ranking"][:top_n_components]

    # Résumé global
    results["summary"] = {
        "total_components": len(component_indices),
        "components_with_peaks": len(results["component_analysis"]),
        "most_significant_component": results["ranking"][0]["component"]
        if results["ranking"]
        else None,
        "avg_peaks_per_component": np.mean(
            [data["n_peaks"] for data in results["component_analysis"].values()]
        )
        if results["component_analysis"]
        else 0,
    }

    print(
        f"✅ Analyse terminée: {results['summary']['components_with_peaks']} composantes avec des pics significatifs"
    )

    return results


def plot_latent_periodogram_overview(
    periodogram_data,
    analysis_results=None,
    n_top_components=6,
    save_path=None,
    show_plot=False,
):
    """
    Crée un plot d'ensemble des périodogrammes des composantes latentes les plus significatives.

    Args:
        periodogram_data: Résultat de compute_latent_periodogram
        analysis_results: Résultat de analyze_latent_periodogram_peaks (optionnel)
        n_top_components: Nombre de composantes à afficher
        save_path: Chemin de sauvegarde
        show_plot: Afficher le plot

    Returns:
        bool: True si le plot a été créé avec succès
    """
    periods = periodogram_data["periods"]
    power_matrix = periodogram_data["power_matrix"]
    component_indices = periodogram_data["component_indices"]

    # Sélection des composantes à afficher
    if analysis_results and "ranking" in analysis_results:
        # Utiliser le ranking pour sélectionner les composantes les plus significatives
        top_components = [
            comp["component"] for comp in analysis_results["ranking"][:n_top_components]
        ]
        display_indices = [
            component_indices.index(comp)
            for comp in top_components
            if comp in component_indices
        ]
    else:
        # Sélection par puissance maximum
        max_powers = np.max(power_matrix, axis=1)
        top_indices = np.argsort(max_powers)[::-1][:n_top_components]
        display_indices = top_indices
        top_components = [component_indices[i] for i in display_indices]

    # Configuration de la figure
    n_cols = min(3, len(display_indices))
    n_rows = (len(display_indices) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    fig.suptitle(
        f"Périodogrammes des composantes latentes les plus significatives\n"
        f"Top {len(display_indices)} composantes sur {len(component_indices)} total",
        fontsize=14,
        fontweight="bold",
    )

    known_periods = analysis_results.get("known_periods") if analysis_results else None

    for plot_idx, (display_idx, comp_idx) in enumerate(
        zip(display_indices, top_components)
    ):
        if plot_idx >= len(axes):
            break

        ax = axes[plot_idx]
        power = power_matrix[display_idx, :]

        # Plot du périodogramme
        ax.semilogx(periods, power, "b-", linewidth=1.0, alpha=0.8)

        # Marquage des périodes connues si disponibles
        if known_periods is not None:
            for i, period in enumerate(known_periods):
                if periods.min() <= period <= periods.max():
                    ax.axvline(
                        period,
                        color="red",
                        linestyle="--",
                        alpha=0.7,
                        linewidth=1,
                        label=f"P{i + 1}: {period:.1f}d" if i < 3 else None,
                    )

        # Marquage du pic principal
        max_idx = np.argmax(power)
        max_period = periods[max_idx]
        max_power = power[max_idx]
        ax.plot(
            max_period, max_power, "ro", markersize=6, label=f"Max: {max_period:.1f}d"
        )

        ax.set_xlabel("Période (jours)")
        ax.set_ylabel("Puissance LS")
        ax.set_title(f"Composante latente {comp_idx}\n(Power max: {max_power:.4f})")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # Masquer les axes vides
    for plot_idx in range(len(display_indices), len(axes)):
        axes[plot_idx].set_visible(False)

    plt.tight_layout()

    # Sauvegarde
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Périodogrammes des composantes latentes sauvegardés: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return True


def plot_latent_periodogram_zoom(
    periodogram_data,
    analysis_results,
    component_idx,
    known_periods=None,
    save_path=None,
    show_plot=False,
):
    """
    Crée un plot détaillé avec zooms sur les pics d'intérêt pour une composante latente spécifique.
    Utilise le même layout que plot_rv_periodogram_juxtaposed.

    Args:
        periodogram_data: Résultat de compute_latent_periodogram
        analysis_results: Résultat de analyze_latent_periodogram_peaks
        component_idx: Index de la composante latente à analyser
        known_periods: Périodes connues des planètes
        save_path: Chemin de sauvegarde
        show_plot: Afficher le plot

    Returns:
        bool: True si le plot a été créé avec succès
    """
    periods = periodogram_data["periods"]
    power_matrix = periodogram_data["power_matrix"]
    component_indices = periodogram_data["component_indices"]
    times = periodogram_data["times"]
    latent_vectors = periodogram_data["latent_vectors"]

    # Vérification que la composante existe
    if component_idx not in component_indices:
        print(f"❌ Composante {component_idx} non trouvée dans les données")
        return False

    matrix_idx = component_indices.index(component_idx)
    power = power_matrix[matrix_idx, :]
    component_values = latent_vectors[:, component_idx]

    # Récupération de l'analyse pour cette composante
    comp_analysis = analysis_results["component_analysis"].get(component_idx, {})

    if known_periods is None:
        known_periods = analysis_results.get("known_periods", [])

    # Layout identique à plot_rv_periodogram_juxtaposed
    if known_periods is not None and len(known_periods) > 0:
        fig = plt.figure(figsize=(18, 14))

        # === TOUT EN HAUT : PÉRIODOGRAMME COMPLET ===
        ax1 = plt.subplot(3, 3, (1, 3))  # Occupe toute la première ligne
        ax1.semilogx(
            periods,
            power,
            "b-",
            linewidth=1.0,
            label=f"Composante latente {component_idx}",
        )
        ax1.set_ylabel("Puissance LS")
        ax1.set_title(f"Périodogramme - Composante latente {component_idx}")
        ax1.grid(True, alpha=0.3)

        # Marquer les périodes connues
        for i, period in enumerate(known_periods):
            if periods.min() <= period <= periods.max():
                ax1.axvline(
                    period,
                    color="orange",
                    linestyle="--",
                    alpha=0.8,
                    linewidth=2,
                    label=f"Planète {i + 1}: {period:.1f}d" if i < 3 else None,
                )
        ax1.legend()

        # === MILIEU : SÉRIE TEMPORELLE DE LA COMPOSANTE ===
        ax2 = plt.subplot(3, 3, (4, 6))  # Occupe toute la deuxième ligne
        ax2.plot(times, component_values, "go-", markersize=1.5, linewidth=0.5)
        ax2.set_ylabel(f"Valeur composante {component_idx}")
        ax2.set_xlabel("JDB")
        ax2.set_title(
            f"Série temporelle - Composante latente {component_idx}\n(std = {np.std(component_values):.4f})"
        )
        ax2.grid(True, alpha=0.3)

        # === EN BAS : 3 ZOOMS PLANÈTES ===
        zoom_positions = [7, 8, 9]  # Dernière ligne
        for i, period in enumerate(known_periods[:3]):
            if periods.min() <= period <= periods.max() and i < 3:
                ax_zoom = plt.subplot(3, 3, zoom_positions[i])

                zoom_factor = 0.2  # ±20% autour de la période
                period_min = period * (1 - zoom_factor)
                period_max = period * (1 + zoom_factor)

                # Masque pour le zoom
                zoom_mask = (periods >= period_min) & (periods <= period_max)

                if np.any(zoom_mask):
                    ax_zoom.plot(
                        periods[zoom_mask],
                        power[zoom_mask],
                        "b-",
                        linewidth=1.5,
                        label=f"Comp. {component_idx}",
                    )

                    # Ligne de la période théorique
                    ax_zoom.axvline(
                        period,
                        color="orange",
                        linestyle="--",
                        alpha=0.8,
                        linewidth=2,
                    )

                    ax_zoom.set_xlabel("Période (jours)")
                    ax_zoom.set_ylabel("Puissance LS")
                    ax_zoom.set_title(f"Planète {i + 1}: {period:.1f}d")
                    ax_zoom.grid(True, alpha=0.3)
                    ax_zoom.legend(fontsize=8)

                    # Trouver et marquer le pic local
                    if len(periods[zoom_mask]) > 0:
                        local_max_idx = np.argmax(power[zoom_mask])
                        local_max_period = periods[zoom_mask][local_max_idx]
                        local_max_power = power[zoom_mask][local_max_idx]
                        ax_zoom.plot(
                            local_max_period, local_max_power, "bo", markersize=6
                        )

    else:
        # Layout simple sans zooms
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))

        # Périodogramme
        ax1.semilogx(
            periods, power, "b-", linewidth=0.8, label=f"Composante {component_idx}"
        )
        ax1.set_ylabel("Puissance LS")
        ax1.set_title(f"Périodogramme - Composante latente {component_idx}")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Série temporelle
        ax2.plot(times, component_values, "go-", markersize=3, linewidth=0.5)
        ax2.set_ylabel(f"Valeur composante {component_idx}")
        ax2.set_xlabel("JDB")
        ax2.set_title(f"Série temporelle (std = {np.std(component_values):.4f})")
        ax2.grid(True, alpha=0.3)

        # Analyse des pics
        if comp_analysis and "peaks" in comp_analysis:
            peaks_text = "Pics principaux:\n"
            for peak in comp_analysis["peaks"][:5]:  # Top 5 pics
                peaks_text += f"P={peak['period']:.1f}d (pow={peak['power']:.4f})\n"
            ax3.text(
                0.1,
                0.9,
                peaks_text,
                transform=ax3.transAxes,
                fontsize=10,
                verticalalignment="top",
            )

        ax3.set_title("Analyse des pics")
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis("off")

        # Statistiques
        stats_text = "Statistiques:\n"
        stats_text += f"Puissance max: {np.max(power):.4f}\n"
        stats_text += f"Période du pic: {periods[np.argmax(power)]:.1f}d\n"
        stats_text += f"Std composante: {np.std(component_values):.4f}\n"
        if comp_analysis:
            stats_text += f"Nb pics significatifs: {comp_analysis.get('n_peaks', 0)}\n"

        ax4.text(
            0.1,
            0.9,
            stats_text,
            transform=ax4.transAxes,
            fontsize=10,
            verticalalignment="top",
        )
        ax4.set_title("Statistiques")
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis("off")

    plt.tight_layout()

    # Sauvegarde
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Zoom périodogramme composante {component_idx} sauvegardé: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return True


def analyze_latent_periodograms_from_checkpoint(
    checkpoint_path="models/aestra_colab_config_final.pth",
    star_name="STAR1136",
    component_indices=None,  # None pour toutes les composantes
    n_top_components=6,  # Nombre de composantes à analyser en détail
    save_dir=None,  # Sera déterminé automatiquement depuis l'expérience
    show_plots=False,
    data_root_dir="data",
    remove_outliers=None,
    exp_dirs=None,  # Répertoires d'expérience pour l'organisation automatique
):
    """
    Fonction complète pour analyser les périodogrammes des composantes latentes depuis un checkpoint.

    Args:
        checkpoint_path: Chemin vers le checkpoint du modèle
        star_name: Nom de l'étoile pour chercher les périodes connues
        component_indices: Indices des composantes à analyser (None pour toutes)
        n_top_components: Nombre de composantes les plus significatives à analyser en détail
        save_dir: Répertoire pour sauvegarder les figures
        show_plots: Afficher les figures ou non
        data_root_dir: Répertoire racine des données
        remove_outliers: Liste des indices à supprimer
        exp_dirs: Répertoires d'expérience

    Returns:
        dict: Résultats complets de l'analyse
    """
    print("=" * 60)
    print("ANALYSE DES PÉRIODOGRAMMES DES COMPOSANTES LATENTES")
    print("=" * 60)

    # Chargement du modèle et du dataset
    exp_data = load_experiment_checkpoint(checkpoint_path, data_root_dir=data_root_dir)
    model = exp_data["model"]
    dataset = exp_data["dataset"]
    cfg_name = exp_data["cfg_name"]

    print(f"Modèle chargé: {cfg_name}")
    print(f"Dataset: {len(dataset)} spectres, {dataset.n_pixels} pixels")

    # Détermination automatique du répertoire de sauvegarde
    if save_dir is None:
        if exp_dirs is not None:
            save_dir = exp_dirs["figures"]
        else:
            auto_exp_dirs = detect_experiment_directories_from_checkpoint(
                checkpoint_path
            )
            if auto_exp_dirs is not None:
                save_dir = auto_exp_dirs["figures"]
            else:
                save_dir = "reports/figures"

    # Extraction des vecteurs latents
    print("\n🔍 Extraction des vecteurs latents...")
    latent_s, rv_values, _, _, _ = extract_latent_vectors_and_rv(
        model=model,
        dataset=dataset,
        batch_size=4,
        remove_outliers=remove_outliers,
        decorrelate_rv=False,
    )

    # Récupération des temps
    if remove_outliers is None:
        remove_outliers = [334, 464]

    n_specs = len(dataset)
    valid_indices = [i for i in range(n_specs) if i not in remove_outliers]
    all_times = dataset.jdb.cpu().numpy()
    times = all_times[valid_indices]

    # Récupération des périodes connues
    known_periods = get_transit_periods(star_name, data_root_dir=data_root_dir)

    # Calcul des périodogrammes des composantes latentes
    print("\n📊 Calcul des périodogrammes des composantes latentes...")
    periodogram_data = compute_latent_periodogram(
        latent_vectors=latent_s,
        times=times,
        component_indices=component_indices,
    )

    # Analyse des pics
    print("\n🎯 Analyse des pics significatifs...")
    analysis_results = analyze_latent_periodogram_peaks(
        periodogram_data=periodogram_data,
        known_periods=known_periods,
        top_n_components=n_top_components,
    )

    # Création des plots
    print("\n🎨 Création des visualisations...")

    # 1. Plot d'ensemble des composantes les plus significatives
    overview_path = os.path.join(save_dir, "latent_periodograms_overview.png")
    plot_latent_periodogram_overview(
        periodogram_data=periodogram_data,
        analysis_results=analysis_results,
        n_top_components=n_top_components,
        save_path=overview_path,
        show_plot=show_plots,
    )

    # 2. Plots détaillés pour les composantes les plus significatives
    zoom_paths = []
    if "top_components" in analysis_results:
        for comp_data in analysis_results["top_components"][:3]:  # Top 3 composantes
            comp_idx = comp_data["component"]
            zoom_path = os.path.join(
                save_dir, f"latent_periodogram_zoom_comp_{comp_idx}.png"
            )

            plot_latent_periodogram_zoom(
                periodogram_data=periodogram_data,
                analysis_results=analysis_results,
                component_idx=comp_idx,
                known_periods=known_periods,
                save_path=zoom_path,
                show_plot=show_plots,
            )
            zoom_paths.append(zoom_path)

    # Préparation des résultats
    results = {
        "periodogram_data": periodogram_data,
        "analysis_results": analysis_results,
        "known_periods": known_periods,
        "latent_dim": latent_s.shape[1],
        "n_spectra": latent_s.shape[0],
        "overview_plot": overview_path,
        "zoom_plots": zoom_paths,
        "most_significant_component": analysis_results["summary"][
            "most_significant_component"
        ],
        "components_with_peaks": analysis_results["summary"]["components_with_peaks"],
    }

    print("\n✅ Analyse terminée!")
    print(f"📐 Dimension latente: {results['latent_dim']}D")
    print(f"📊 Composantes avec pics significatifs: {results['components_with_peaks']}")
    if results["most_significant_component"] is not None:
        print(
            f"🎯 Composante la plus significative: {results['most_significant_component']}"
        )

    return results


# =============================================================================
# HIGH-LEVEL ANALYSIS FUNCTIONS - Fonctions d'analyse haut niveau
# =============================================================================


def analyze_rv_periodogram_from_checkpoint(
    checkpoint_path="models/aestra_colab_config_final.pth",
    star_name="STAR1136",
    save_path=None,  # Sera déterminé automatiquement depuis l'expérience
    show_plot=False,
    data_root_dir="data",  # Ajout pour la compatibilité avec les chemins de données
    remove_outliers=None,  # Nouvelle option pour supprimer les outliers
    decorrelate_rv=False,  # Nouvelle option pour décorréler les RV
    exp_dirs=None,  # Répertoires d'expérience pour l'organisation automatique
):
    """
    Fonction autonome pour calculer et tracer le périodogramme depuis un checkpoint.

    Args:
        checkpoint_path: Chemin vers le checkpoint du modèle
        star_name: Nom de l'étoile pour chercher les périodes connues
        save_path: Chemin pour sauvegarder la figure (None pour auto-détection depuis l'expérience)
        show_plot: Afficher la figure ou non
        data_root_dir: Répertoire racine des données pour charger le dataset
        remove_outliers: Liste des indices à supprimer (défaut [334, 464])
        decorrelate_rv: Si True, applique la décorrélation et compare les deux périodogrammes
        exp_dirs: Répertoires d'expérience (fournis automatiquement par train.py)

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

    # Détermination automatique du chemin de sauvegarde si non fourni
    if save_path is None:
        if exp_dirs is not None:
            # Utiliser le répertoire figures de l'expérience
            save_path = os.path.join(exp_dirs["figures"], "rv_periodogram_analysis.png")
        else:
            # Auto-détection depuis le checkpoint_path
            auto_exp_dirs = detect_experiment_directories_from_checkpoint(
                checkpoint_path
            )
            if auto_exp_dirs is not None:
                save_path = os.path.join(
                    auto_exp_dirs["figures"], "rv_periodogram_analysis.png"
                )
            else:
                # Fallback vers le répertoire reports/figures
                save_path = "reports/figures/rv_periodogram_analysis.png"

    # Récupération des périodes connues
    known_periods = get_transit_periods(star_name, data_root_dir=data_root_dir)

    # Calcul du périodogramme
    print("Calcul du périodogramme des vitesses radiales...")
    periods, power, rv_values, times, periods_corr, power_corr, rv_corrected = (
        compute_rv_periodogram(
            model=model,
            dataset=dataset,
            batch_size=4,  # Ajustable selon la mémoire disponible
            remove_outliers=remove_outliers,
            decorrelate_rv=decorrelate_rv,
        )
    )

    # Analyse des pics pour les deux périodogrammes
    peaks_info = analyze_periodogram_peaks(periods, power, threshold_ratio=0.1)
    peaks_info_corr = None
    if decorrelate_rv and power_corr is not None:
        peaks_info_corr = analyze_periodogram_peaks(
            periods_corr, power_corr, threshold_ratio=0.1
        )

    # Création du plot avec juxtaposition
    plot_rv_periodogram_juxtaposed(
        periods=periods,
        power=power,
        rv_values=rv_values,
        times=times,
        known_periods=known_periods,
        save_path=save_path,
        show_plot=show_plot,
        periods_corr=periods_corr,
        power_corr=power_corr,
        rv_corrected=rv_corrected,
        decorrelate_applied=decorrelate_rv,
    )

    # Retour des résultats
    results = {
        "periods": periods,
        "power": power,
        "rv_values": rv_values,
        "times": times,
        "periods_corr": periods_corr,
        "power_corr": power_corr,
        "rv_corrected": rv_corrected,
        "decorrelate_applied": decorrelate_rv and rv_corrected is not None,
        "known_periods": known_periods,
        "peaks_info": peaks_info,
        "peaks_info_corr": peaks_info_corr,
        "max_power": np.max(power),
        "best_period": periods[np.argmax(power)],
        "max_power_corr": np.max(power_corr) if power_corr is not None else None,
        "best_period_corr": periods_corr[np.argmax(power_corr)]
        if power_corr is not None
        else None,
        "observation_span": times.max() - times.min(),
        "n_observations": len(rv_values),
    }

    return results


def analyze_latent_space_from_checkpoint(
    checkpoint_path="models/aestra_colab_config_final.pth",
    save_path=None,  # Sera déterminé automatiquement depuis l'expérience
    show_plot=False,
    data_root_dir="data",  # Ajout pour la compatibilité avec les chemins de données
    remove_outliers=None,  # Nouvelle option pour supprimer les outliers
    decorrelate_rv=False,  # Nouvelle option pour décorréler les RV
    exp_dirs=None,  # Répertoires d'expérience pour l'organisation automatique
):
    """
    Fonction autonome pour analyser l'espace latent depuis un checkpoint.

    Args:
        checkpoint_path: Chemin vers le checkpoint du modèle
        save_path: Chemin pour sauvegarder la figure (None pour auto-détection depuis l'expérience)
        show_plot: Afficher la figure ou non
        data_root_dir: Répertoire racine des données pour charger le dataset
        remove_outliers: Liste des indices à supprimer (défaut [334, 464])
        decorrelate_rv: Si True, applique la décorrélation des RV avec l'espace latent
        exp_dirs: Répertoires d'expérience (fournis automatiquement par train.py)

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

    # Détermination automatique du chemin de sauvegarde si non fourni
    if save_path is None:
        if exp_dirs is not None:
            # Utiliser le répertoire figures de l'expérience
            save_path = os.path.join(
                exp_dirs["figures"], "latent_space_3d_analysis.png"
            )
        else:
            # Auto-détection depuis le checkpoint_path
            auto_exp_dirs = detect_experiment_directories_from_checkpoint(
                checkpoint_path
            )
            if auto_exp_dirs is not None:
                save_path = os.path.join(
                    auto_exp_dirs["figures"], "latent_space_3d_analysis.png"
                )
            else:
                # Fallback vers le répertoire reports/figures
                save_path = "reports/figures/latent_space_3d_analysis.png"

    # Extraction des vecteurs latents et RV
    latent_s, rv_values, rv_corrected, rv_bias, sigma_R = extract_latent_vectors_and_rv(
        model=model,
        dataset=dataset,
        batch_size=4,
        remove_outliers=remove_outliers,
        decorrelate_rv=decorrelate_rv,
    )

    # Création du plot 3D si possible
    plot_created = plot_latent_space_3d(
        latent_s=latent_s,
        rv_values=rv_values,
        save_path=save_path,
        show_plot=show_plot,
    )
    if decorrelate_rv and rv_corrected is not None:
        base_path = save_path.replace(".png", "")
        corrected_save_path = f"{base_path}_corrected.png"

        plot_latent_space_3d(
            latent_s=latent_s,
            rv_values=rv_corrected,
            save_path=corrected_save_path,
            show_plot=show_plot,
            decorrelated=True,
        )

    # Retour des résultats
    results = {
        "latent_s": latent_s,
        "rv_values": rv_values,
        "rv_corrected": rv_corrected,
        "rv_bias": rv_bias,
        "sigma_R": sigma_R,
        "decorrelate_applied": decorrelate_rv and rv_corrected is not None,
        "latent_dim": latent_s.shape[1],
        "n_spectra": latent_s.shape[0],
        "plot_created": plot_created,
        "rv_stats": {
            "original": {
                "min": np.min(rv_values),
                "max": np.max(rv_values),
                "mean": np.mean(rv_values),
                "std": np.std(rv_values),
            },
            "corrected": {
                "min": np.min(rv_corrected) if rv_corrected is not None else None,
                "max": np.max(rv_corrected) if rv_corrected is not None else None,
                "mean": np.mean(rv_corrected) if rv_corrected is not None else None,
                "std": np.std(rv_corrected) if rv_corrected is not None else None,
            }
            if rv_corrected is not None
            else None,
        },
    }

    return results


def analyze_activity_signals_from_checkpoint(
    checkpoint_path="models/aestra_colab_config_final.pth",
    n_spectra=4,
    n_lines=4,
    save_path=None,  # Sera déterminé automatiquement depuis l'expérience
    show_plot=False,
    data_root_dir="data",
    remove_outliers=None,
    spectrum_indices=None,
    exp_dirs=None,  # Répertoires d'expérience pour l'organisation automatique
):
    """
    Fonction autonome pour analyser les signaux d'activité depuis un checkpoint.

    Args:
        checkpoint_path: Chemin vers le checkpoint du modèle
        n_spectra: Nombre de spectres à analyser (défaut 4)
        n_lines: Nombre de raies à analyser (défaut 4)
        save_path: Chemin pour sauvegarder la figure (None pour auto-détection depuis l'expérience)
        show_plot: Afficher la figure ou non
        data_root_dir: Répertoire racine des données
        remove_outliers: Liste des indices à supprimer (défaut [334, 464])
        spectrum_indices: Indices spécifiques des spectres à analyser (optionnel)
        exp_dirs: Répertoires d'expérience (fournis automatiquement par train.py)

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

    # Détermination automatique du chemin de sauvegarde si non fourni
    if save_path is None:
        if exp_dirs is not None:
            # Utiliser le répertoire figures de l'expérience
            save_path = os.path.join(exp_dirs["figures"], "activity_signals_mosaic.png")
        else:
            # Auto-détection depuis le checkpoint_path
            auto_exp_dirs = detect_experiment_directories_from_checkpoint(
                checkpoint_path
            )
            if auto_exp_dirs is not None:
                save_path = os.path.join(
                    auto_exp_dirs["figures"], "activity_signals_mosaic.png"
                )
            else:
                # Fallback vers le répertoire reports/figures
                save_path = "reports/figures/activity_signals_mosaic.png"

    # Analyse des signaux d'activité
    activity_results = analyze_activity_signals_mosaic(
        model=model,
        dataset=dataset,
        n_spectra=n_spectra,
        n_lines=n_lines,
        data_root_dir=data_root_dir,
        save_path=save_path,
        show_plot=show_plot,
        remove_outliers=remove_outliers,
        spectrum_indices=spectrum_indices,
    )

    return activity_results


def full_analysis_from_checkpoint(
    checkpoint_path="models/aestra_colab_config_final.pth",
    star_name="STAR1136",
    save_dir=None,  # Sera déterminé automatiquement depuis l'expérience
    show_plots=False,
    data_root_dir="data",  # Ajout pour la compatibilité avec les chemins de données
    remove_outliers=None,  # Nouvelle option pour supprimer les outliers
    decorrelate_rv=False,  # Nouvelle option pour décorréler les RV
    exp_dirs=None,  # Répertoires d'expérience pour l'organisation automatique
):
    """
    Fonction pour effectuer une analyse complète depuis un checkpoint.

    Args:
        checkpoint_path: Chemin vers le checkpoint du modèle
        star_name: Nom de l'étoile
        save_dir: Répertoire pour sauvegarder les figures (None pour auto-détection depuis l'expérience)
        show_plots: Afficher les figures ou non
        data_root_dir: Répertoire racine des données pour charger le dataset
        remove_outliers: Liste des indices à supprimer (défaut [334, 464])
        decorrelate_rv: Si True, applique la décorrélation des RV
        exp_dirs: Répertoires d'expérience (fournis automatiquement par train.py)

    Returns:
        dict: Dictionnaire contenant tous les résultats de l'analyse
    """

    print("=" * 60)
    print("ANALYSE COMPLÈTE AESTRA")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Étoile: {star_name}")
    if remove_outliers:
        print(f"Outliers à supprimer: {remove_outliers}")
    if decorrelate_rv:
        print("Décorrélation des RV activée")

    # Chargement du modèle et du dataset
    exp_data = load_experiment_checkpoint(checkpoint_path, data_root_dir=data_root_dir)
    model = exp_data["model"]
    dataset = exp_data["dataset"]
    cfg_name = exp_data["cfg_name"]

    print(f"\nModèle chargé: {cfg_name}")
    print(f"Dataset: {len(dataset)} spectres, {dataset.n_pixels} pixels")

    # Détermination automatique du répertoire de sauvegarde si non fourni
    if save_dir is None:
        if exp_dirs is not None:
            # Utiliser le répertoire figures de l'expérience
            save_dir = exp_dirs["figures"]
        else:
            # Auto-détection depuis le checkpoint_path
            auto_exp_dirs = detect_experiment_directories_from_checkpoint(
                checkpoint_path
            )
            if auto_exp_dirs is not None:
                save_dir = auto_exp_dirs["figures"]
            else:
                # Fallback vers le répertoire reports/figures
                save_dir = "reports/figures"

    print(f"Répertoire de sauvegarde: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)

    results = {}

    # 1. Analyse des distances latentes (Figure 3)
    print("\n1. Calcul des distances latentes...")
    delta_s_rand, delta_s_aug = compute_latent_distances(
        model=model, dataset=dataset, batch_size=4
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
        data_root_dir=data_root_dir,
        remove_outliers=remove_outliers,
        decorrelate_rv=decorrelate_rv,
        exp_dirs={"figures": save_dir} if exp_dirs is None else exp_dirs,
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
        decorrelate_rv=decorrelate_rv,
        exp_dirs={"figures": save_dir} if exp_dirs is None else exp_dirs,
    )
    results["periodogram"] = periodogram_analysis

    # 5. Analyse des périodogrammes des composantes latentes
    print("\n5. Analyse des périodogrammes des composantes latentes...")
    latent_periodogram_analysis = analyze_latent_periodograms_from_checkpoint(
        checkpoint_path=checkpoint_path,
        star_name=star_name,
        n_top_components=6,
        save_dir=save_dir,
        show_plots=show_plots,
        data_root_dir=data_root_dir,
        remove_outliers=remove_outliers,
        exp_dirs={"figures": save_dir} if exp_dirs is None else exp_dirs,
    )
    results["latent_periodogram"] = latent_periodogram_analysis

    # 6. Analyse des signaux d'activité pour les raies importantes
    print("\n6. Analyse des signaux d'activité pour les raies importantes...")
    activity_analysis = analyze_activity_signals_from_checkpoint(
        checkpoint_path=checkpoint_path,
        n_spectra=4,
        n_lines=4,
        save_path=os.path.join(save_dir, "activity_signals_mosaic.png"),
        show_plot=show_plots,
        data_root_dir=data_root_dir,
        remove_outliers=remove_outliers,
        exp_dirs={"figures": save_dir} if exp_dirs is None else exp_dirs,
    )
    results["activity_signals"] = activity_analysis

    # Résumé
    print("\n" + "=" * 60)
    print("RÉSUMÉ DE L'ANALYSE")
    print("=" * 60)
    print(f"Dimension latente: {results['latent_space']['latent_dim']}D")
    print(f"Nombre de spectres: {results['latent_space']['n_spectra']}")
    print(
        f"Plage RV originales: {results['latent_space']['rv_stats']['original']['min']:.2f} à {results['latent_space']['rv_stats']['original']['max']:.2f}"
    )
    if results["latent_space"]["decorrelate_applied"]:
        print(
            f"Plage RV décorrélées: {results['latent_space']['rv_stats']['corrected']['min']:.2f} à {results['latent_space']['rv_stats']['corrected']['max']:.2f}"
        )
    print(
        f"Distance latente moyenne (paires aléatoires): {results['latent_distances']['mean_rand']:.3e}"
    )
    print(
        f"Distance latente moyenne (paires augmentées): {results['latent_distances']['mean_aug']:.3e}"
    )
    print(
        f"Meilleure période détectée (RV originales): {results['periodogram']['best_period']:.2f} jours"
    )
    if results["periodogram"]["decorrelate_applied"]:
        print(
            f"Meilleure période détectée (RV décorrélées): {results['periodogram']['best_period_corr']:.2f} jours"
        )
    if results["periodogram"]["known_periods"] is not None:
        print(f"Périodes connues: {results['periodogram']['known_periods']}")

    # Nouvelles informations sur les composantes latentes
    if "latent_periodogram" in results:
        print(
            f"Composantes latentes avec pics significatifs: {results['latent_periodogram']['components_with_peaks']}"
        )
        if results["latent_periodogram"]["most_significant_component"] is not None:
            print(
                f"Composante latente la plus significative: {results['latent_periodogram']['most_significant_component']}"
            )

    if results["activity_signals"] is not None:
        n_analyzed_spectra = len(results["activity_signals"]["selected_spectra"])
        n_analyzed_lines = len(results["activity_signals"]["selected_lines"])
        print(
            f"Signaux d'activité analysés: {n_analyzed_spectra} spectres, {n_analyzed_lines} raies"
        )
    print("=" * 60)

    return results


# =============================================================================
# MAIN EXECUTION - Exécution principale
# =============================================================================


def main(
    cfg_name=None,
    checkpoint_path=None,
    star_name="STAR1136",
    show_plots=True,
    remove_outliers=None,
    decorrelate_rv=True,
    data_root_dir="data",
    analysis_type="full",
):
    """
    Fonction principale d'analyse AESTRA compatible avec les notebooks.

    Args:
        cfg_name: Nom de la configuration (ex: "colab_config", "base_config")
        checkpoint_path: Chemin vers le checkpoint (optionnel, auto-détecté si cfg_name fourni)
        star_name: Nom de l'étoile à analyser
        show_plots: Afficher les plots interactifs
        remove_outliers: Liste des indices à supprimer (défaut [334, 464])
        decorrelate_rv: Appliquer la décorrélation des RV
        data_root_dir: Répertoire racine des données
        analysis_type: Type d'analyse ("full", "periodogram", "latent", "activity", "latent_periodogram")

    Returns:
        dict: Résultats de l'analyse
    """
    import os
    import sys

    # Configuration par défaut des outliers
    if remove_outliers is None:
        remove_outliers = [334, 464]

    # Gestion des arguments via argparse SEULEMENT si appelé en ligne de commande
    # (pas dans un notebook)
    if cfg_name is None and checkpoint_path is None and len(sys.argv) > 1:
        # Mode ligne de commande détecté
        import argparse

        parser = argparse.ArgumentParser(
            description="Analyse AESTRA avec auto-détection des checkpoints"
        )
        parser.add_argument(
            "--cfg_name",
            help="Nom de l'expérience (ex: colab_config)",
            default="base_config",
        )
        parser.add_argument(
            "--checkpoint_path", help="Chemin vers un checkpoint spécifique (optionnel)"
        )
        parser.add_argument(
            "--star_name", default="STAR1136", help="Nom de l'étoile à analyser"
        )
        parser.add_argument(
            "--show_plots", action="store_true", help="Afficher les plots interactifs"
        )
        parser.add_argument(
            "--no_remove_outliers",
            action="store_true",
            help="Ne pas supprimer les outliers par défaut",
        )
        parser.add_argument(
            "--no_decorrelate",
            action="store_true",
            help="Ne pas appliquer la décorrélation des RV",
        )
        parser.add_argument(
            "--data_root_dir", default="data", help="Répertoire racine des données"
        )
        parser.add_argument(
            "--analysis_type",
            choices=["full", "periodogram", "latent", "activity", "latent_periodogram"],
            default="full",
            help="Type d'analyse à effectuer",
        )

        args = parser.parse_args()
        cfg_name = args.cfg_name
        checkpoint_path = args.checkpoint_path or checkpoint_path
        star_name = args.star_name
        show_plots = args.show_plots
        remove_outliers = [] if args.no_remove_outliers else remove_outliers
        decorrelate_rv = not args.no_decorrelate
        data_root_dir = args.data_root_dir
        analysis_type = args.analysis_type

    print(f"🔍 AESTRA Analysis - Experiment: {cfg_name}")
    print(f"📊 Analysis type: {analysis_type}")
    print(f"⭐ Star: {star_name}")
    print(f"🚫 Remove outliers: {remove_outliers}")
    print(f"🔧 Decorrelate RV: {decorrelate_rv}")

    # Auto-détection du checkpoint si non fourni
    if checkpoint_path is None:
        # Essayer plusieurs chemins possibles
        possible_paths = [
            f"experiments/aestra_{cfg_name}_experiment/models/aestra_{cfg_name}_final.pth",
            f"experiments/aestra_{cfg_name}/models/aestra_{cfg_name}_final.pth",
            f"models/aestra_{cfg_name}_final.pth",
            f"models/aestra_{cfg_name}_phase_joint_epoch_200.pth",
            f"models/aestra_{cfg_name}_phase_joint_epoch_100.pth",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                checkpoint_path = path
                print(f"✅ Checkpoint auto-détecté: {checkpoint_path}")
                break

        if checkpoint_path is None:
            raise FileNotFoundError(
                f"Aucun checkpoint trouvé pour {cfg_name}. Chemins testés:\n"
                + "\n".join(f"  - {p}" for p in possible_paths)
            )
    else:
        print(f"📂 Checkpoint fourni: {checkpoint_path}")

    # Vérification de l'existence du checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint non trouvé: {checkpoint_path}")

    # Exécution de l'analyse selon le type demandé
    try:
        if analysis_type == "full":
            print("\n🚀 Lancement de l'analyse complète...")
            results = full_analysis_from_checkpoint(
                checkpoint_path=checkpoint_path,
                star_name=star_name,
                show_plots=show_plots,
                data_root_dir=data_root_dir,
                remove_outliers=remove_outliers,
                decorrelate_rv=decorrelate_rv,
            )

        elif analysis_type == "periodogram":
            print("\n📈 Analyse du périodogramme RV...")
            results = analyze_rv_periodogram_from_checkpoint(
                checkpoint_path=checkpoint_path,
                star_name=star_name,
                show_plot=show_plots,
                data_root_dir=data_root_dir,
                remove_outliers=remove_outliers,
                decorrelate_rv=decorrelate_rv,
            )

        elif analysis_type == "latent":
            print("\n🎯 Analyse de l'espace latent...")
            results = analyze_latent_space_from_checkpoint(
                checkpoint_path=checkpoint_path,
                show_plot=show_plots,
                data_root_dir=data_root_dir,
                remove_outliers=remove_outliers,
                decorrelate_rv=decorrelate_rv,
            )

        elif analysis_type == "activity":
            print("\n🌟 Analyse des signaux d'activité...")
            results = analyze_activity_signals_from_checkpoint(
                checkpoint_path=checkpoint_path,
                show_plot=show_plots,
                data_root_dir=data_root_dir,
                remove_outliers=remove_outliers,
            )

        elif analysis_type == "latent_periodogram":
            print("\n📊 Analyse des périodogrammes des composantes latentes...")
            results = analyze_latent_periodograms_from_checkpoint(
                checkpoint_path=checkpoint_path,
                star_name=star_name,
                show_plots=show_plots,
                data_root_dir=data_root_dir,
                remove_outliers=remove_outliers,
            )

        else:
            raise ValueError(f"Type d'analyse inconnu: {analysis_type}")

        print(f"\n✅ Analyse '{analysis_type}' terminée avec succès!")

        # Affichage du résumé selon le type d'analyse
        if analysis_type == "full" and "periodogram" in results:
            print(
                f"🎯 Meilleure période détectée: {results['periodogram']['best_period']:.2f} jours"
            )
            if "latent_space" in results:
                print(
                    f"📐 Dimension de l'espace latent: {results['latent_space']['latent_dim']}D"
                )
                print(
                    f"📊 Nombre de spectres analysés: {results['latent_space']['n_spectra']}"
                )

        elif analysis_type == "latent_periodogram":
            print(f"📐 Dimension latente: {results['latent_dim']}D")
            print(
                f"📊 Composantes avec pics significatifs: {results['components_with_peaks']}"
            )
            if results["most_significant_component"] is not None:
                print(
                    f"🎯 Composante la plus significative: {results['most_significant_component']}"
                )

        return results

    except Exception as e:
        print(f"❌ Erreur lors de l'analyse: {str(e)}")
        raise


if __name__ == "__main__":
    # Appel de la fonction main pour l'exécution en ligne de commande
    main(
        checkpoint_path="experiments/aestra_local_experiment/models/aestra_base_config_final.pth",
        show_plots=False,
    )


# =============================================================================
# USAGE EXAMPLES - Exemples d'utilisation
# =============================================================================

"""
Exemples d'utilisation des fonctions:

# 1. Analyse complète depuis un checkpoint avec suppression des outliers
results = full_analysis_from_checkpoint(
    checkpoint_path="models/aestra_colab_config_final.pth",
    star_name="STAR1136",
    save_dir="mon_analyse",
    show_plots=True,
    remove_outliers=[334, 464]  # Suppression des outliers par défaut
)

# 2. Analyse complète sans suppression d'outliers
results = full_analysis_from_checkpoint(
    checkpoint_path="models/aestra_colab_config_final.pth",
    star_name="STAR1136",
    save_dir="mon_analyse",
    show_plots=True,
    remove_outliers=[]  # Aucun outlier à supprimer
)

# 3. Analyse seulement du périodogramme avec outliers supprimés
periodogram_results = analyze_rv_periodogram_from_checkpoint(
    checkpoint_path="models/aestra_colab_config_final.pth",
    save_path="mon_periodogramme.png",
    show_plot=True,
    remove_outliers=[334, 464]
)

# 4. Analyse seulement de l'espace latent avec outliers supprimés
latent_results = analyze_latent_space_from_checkpoint(
    checkpoint_path="models/aestra_colab_config_final.pth",
    save_path="mon_espace_latent.png",
    show_plot=True,
    remove_outliers=[334, 464]
)

# 5. Analyse des signaux d'activité pour les raies importantes
activity_results = analyze_activity_signals_from_checkpoint(
    checkpoint_path="models/aestra_colab_config_final.pth",
    n_spectra=6,  # 6 spectres
    n_lines=3,    # 3 raies les plus importantes
    save_path="mon_analyse_activite.png",
    show_plot=True,
    remove_outliers=[334, 464],
    spectrum_indices=[10, 50, 100, 200, 300, 400]  # Indices spécifiques (optionnel)
)

# 6. Utilisation modulaire des fonctions individuelles avec outliers supprimés
exp_data = load_experiment_checkpoint("models/aestra_colab_config_final.pth")
model, dataset = exp_data["model"], exp_data["dataset"]

# 6. Utilisation modulaire des fonctions individuelles avec outliers supprimés
exp_data = load_experiment_checkpoint("models/aestra_colab_config_final.pth")
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

# Analyse des signaux d'activité avec paramètres personnalisés
activity_results = analyze_activity_signals_mosaic(
    model=model,
    dataset=dataset,
    n_spectra=3,
    n_lines=5,
    remove_outliers=[334, 464],
    line_window=1.5,  # Fenêtre de 1.5 Å autour de chaque raie
    spectrum_indices=[42, 123, 256]  # Spectres spécifiques
)

# 7. NOUVELLES FONCTIONS - Analyse des périodogrammes des composantes latentes

# Analyse complète des périodogrammes des composantes latentes
latent_periodogram_results = analyze_latent_periodograms_from_checkpoint(
    checkpoint_path="models/aestra_colab_config_final.pth",
    star_name="STAR1136",
    n_top_components=8,  # Analyser les 8 composantes les plus significatives
    show_plots=True,
    remove_outliers=[334, 464]
)

# Utilisation modulaire des nouvelles fonctions
import numpy as np
from astropy.timeseries import LombScargle

# Récupération des temps
n_specs = len(dataset)
valid_indices = [i for i in range(n_specs) if i not in [334, 464]]
all_times = dataset.jdb.cpu().numpy()
times = all_times[valid_indices]

# Calcul des périodogrammes pour toutes les composantes latentes
periodogram_data = compute_latent_periodogram(
    latent_vectors=latent_s,
    times=times,
    component_indices=None,  # Toutes les composantes
)

# Analyse des pics significatifs
known_periods = get_transit_periods("STAR1136")
analysis_results = analyze_latent_periodogram_peaks(
    periodogram_data=periodogram_data,
    known_periods=known_periods,
    threshold_ratio=0.1,  # Pics > 10% du maximum
    top_n_components=6
)

# Visualisation d'ensemble des composantes les plus significatives
plot_latent_periodogram_overview(
    periodogram_data=periodogram_data,
    analysis_results=analysis_results,
    n_top_components=6,
    save_path="reports/figures/latent_periodograms_overview.png",
    show_plot=True
)

# Analyse détaillée d'une composante spécifique (par exemple, la plus significative)
if analysis_results["most_significant_component"] is not None:
    most_significant = analysis_results["most_significant_component"]
    plot_latent_periodogram_zoom(
        periodogram_data=periodogram_data,
        analysis_results=analysis_results,
        component_idx=most_significant,
        known_periods=known_periods,
        save_path=f"reports/figures/latent_periodogram_zoom_comp_{most_significant}.png",
        show_plot=True
    )

# Périodes connues
known_periods = get_transit_periods("STAR1136")

# 8. Analyse spécialisée pour des composantes spécifiques
specific_components = [0, 1, 2, 5, 10]  # Analyser des composantes spécifiques
periodogram_data_specific = compute_latent_periodogram(
    latent_vectors=latent_s,
    times=times,
    component_indices=specific_components,
)

analysis_specific = analyze_latent_periodogram_peaks(
    periodogram_data=periodogram_data_specific,
    known_periods=known_periods,
    top_n_components=3
)

# Créer des zooms pour chaque composante d'intérêt
for comp_idx in specific_components[:3]:  # Top 3 seulement
    plot_latent_periodogram_zoom(
        periodogram_data=periodogram_data_specific,
        analysis_results=analysis_specific,
        component_idx=comp_idx,
        known_periods=known_periods,
        save_path=f"reports/figures/latent_zoom_comp_{comp_idx}.png",
        show_plot=False
    )

print("Résumé de l'analyse des composantes latentes:")
print(f"- Dimension latente totale: {latent_s.shape[1]}D")
print(f"- Composantes avec pics significatifs: {analysis_results['summary']['components_with_peaks']}")
print(f"- Composante la plus significative: {analysis_results['summary']['most_significant_component']}")
print(f"- Nombre moyen de pics par composante: {analysis_results['summary']['avg_peaks_per_component']:.1f}")
"""
