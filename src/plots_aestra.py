"""
Fonctions de plotting pour l'entraînement AESTRA.

Ce module contient les fonctions de visualisation optimisées pour AESTRA :
- Plotting des losses d'entraînement
- Visualisation des spectres selon le papier AESTRA
- Zoom ultra-précis pour l'analyse Doppler
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Optional, Tuple
from src.interpolate import shift_spectra_linear


def plot_losses(losses_history, exp_name, phase_name, epoch, plot_dir, console):
    """
    Crée une mosaïque avec tous les plots des losses sur une seule figure.

    Args:
        losses_history: Dict avec les listes de losses {'rv': [...], 'fid': [...], etc.}
        exp_name: Nom de l'expérience
        phase_name: Nom de la phase actuelle
        epoch: Epoch actuelle
        plot_dir: Répertoire de sauvegarde des plots
        console: Instance de console pour les logs (rich.console.Console)
    """
    os.makedirs(plot_dir, exist_ok=True)

    # Configurer la mosaïque (2x3 pour 6 plots)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        f"{exp_name} - {phase_name} - Training Losses (Epoch {epoch})",
        fontsize=16,
        fontweight="bold",
    )

    epochs = range(1, len(losses_history["rv"]) + 1)

    # Plot 1: RV Loss
    axes[0, 0].plot(epochs, losses_history["rv"], "b-", linewidth=2, label="RV Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("RV Loss")
    axes[0, 0].set_title("Radial Velocity Loss")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale("log")
    axes[0, 0].legend()

    # Plot 2: Fidelity Loss
    axes[0, 1].plot(
        epochs, losses_history["fid"], "r-", linewidth=2, label="Fidelity Loss"
    )
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Fidelity Loss")
    axes[0, 1].set_title("Reconstruction Fidelity Loss")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale("log")
    axes[0, 1].legend()

    # Plot 3: Regularization Loss
    axes[0, 2].plot(
        epochs, losses_history["reg"], "m-", linewidth=2, label="Regularization Loss"
    )
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("Regularization Loss")
    axes[0, 2].set_title("Ridge Regularization Loss")
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_yscale("log")
    axes[0, 2].legend()

    # Plot 4: Consistency Loss (C)
    axes[1, 0].plot(
        epochs, losses_history["c"], "g-", linewidth=2, label="Consistency Loss"
    )
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Consistency Loss")
    axes[1, 0].set_title("Latent Consistency Loss")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale("log")
    axes[1, 0].legend()

    # Plot 5: Learning Rate (Scheduler) - Log Scale
    if "lr" in losses_history and losses_history["lr"]:
        axes[1, 1].plot(
            epochs, losses_history["lr"], "orange", linewidth=2, label="Learning Rate"
        )
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Learning Rate")
        axes[1, 1].set_title("Learning Rate Schedule (Log Scale)")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale("log")
        axes[1, 1].legend()
    else:
        # Si pas de données de scheduler, afficher un message
        axes[1, 1].text(
            0.5,
            0.5,
            "No Learning Rate\nData Available",
            transform=axes[1, 1].transAxes,
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"),
        )
        axes[1, 1].set_title("Learning Rate Schedule (Log Scale)")

    # Plot 6: Learning Rate (Scheduler) - Linear Scale
    if "lr" in losses_history and losses_history["lr"]:
        axes[1, 2].plot(
            epochs, losses_history["lr"], "orange", linewidth=2, label="Learning Rate"
        )
        axes[1, 2].set_xlabel("Epoch")
        axes[1, 2].set_ylabel("Learning Rate")
        axes[1, 2].set_title("Learning Rate Schedule (Linear Scale)")
        axes[1, 2].grid(True, alpha=0.3)
        # Pas de log scale ici pour voir les détails
        axes[1, 2].legend()
    else:
        axes[1, 2].text(
            0.5,
            0.5,
            "No Learning Rate\nData Available",
            transform=axes[1, 2].transAxes,
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"),
        )
        axes[1, 2].set_title("Learning Rate Schedule (Linear Scale)")

    plt.tight_layout()

    # Sauvegarde
    filename = f"{exp_name}_{phase_name}_losses_mosaic_epoch_{epoch}.png"
    filepath = os.path.join(plot_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    console.log(f"📊 Mosaic plot saved: {filename}")


def plot_aestra_analysis(
    batch: tuple,
    model: torch.nn.Module,
    exp_name: str,
    phase_name: str,
    epoch: int,
    plot_dir: str,
    sample_idx: Optional[int] = None,
    zoom_line: bool = False,
) -> None:
    """
    Plot complet d'analyse AESTRA selon le papier de référence.

    Affiche les spectres clés mentionnés dans le papier :
    - y_obs : spectre observé
    - y_aug : spectre augmenté (avec décalage Doppler artificiel)
    - r_obs, r_aug : spectres résiduels (après soustraction template)
    - b_rest : template au repos (rest-frame)
    - y_rest : modèle au repos complet
    - y_act : spectre d'activité
    - y_obs_prime : reconstruction finale

    Args:
        batch: Batch de données (y_obs, y_aug, v_offset, wavegrid)
        model: Modèle AESTRA
        exp_name: Nom de l'expérience
        phase_name: Phase d'entraînement
        epoch: Époch actuelle
        plot_dir: Répertoire de sauvegarde
        device: Device (cpu/cuda)
        sample_idx: Index de l'échantillon (None pour aléatoire)
    """
    os.makedirs(plot_dir, exist_ok=True)

    # Extraction des données du batch
    batch_yobs, batch_yaug, batch_voffset, batch_wavegrid = batch
    batch_size = batch_yobs.shape[0]

    g2mask = np.loadtxt("data/rv_datachallenge/masks/G2_mask.txt")
    line_positions, line_weights = g2mask[:, 0], g2mask[:, 1]
    wavegrid = batch_wavegrid[0].detach().cpu().numpy()
    line_weights = line_weights[
        (line_positions >= wavegrid.min()) & (line_positions <= wavegrid.max())
    ]
    line_positions = line_positions[
        (line_positions >= wavegrid.min()) & (line_positions <= wavegrid.max())
    ]
    most_weighted_line = line_positions[np.argmax(line_weights)]
    halfwin = 0.18  # Fenêtre de zoom de 0.5 Å

    # Sélection d'un échantillon
    if sample_idx is None:
        sample_idx = np.random.randint(0, batch_size)

    # Forward pass du modèle pour obtenir les composantes AESTRA
    model.eval()
    with torch.no_grad():
        # Spectres résiduels (après soustraction des templates)
        batch_robs = batch_yobs - model.b_obs.unsqueeze(0)
        batch_raug = batch_yaug - model.b_obs.unsqueeze(0)

        # Encodage + Décodage pour obtenir les paramètres d'activité et le spectre d'activité
        batch_yact, batch_s = model.spender(batch_robs)
        batch_yact_aug, batch_s_aug = model.spender(batch_raug)

        batch_yrest = batch_yact + model.b_rest.unsqueeze(0)

        # Estimation RV et reconstruction finale
        batch_vencode = model.rvestimator(batch_robs)

        # Reconstruction finale (y_obs_prime)
        # Appliquer le décalage Doppler estimé au modèle au repos
        batch_yobsprime = shift_spectra_linear(
            batch_yrest, batch_wavegrid, batch_vencode
        )

    # Données pour l'échantillon sélectionné
    y_obs = batch_yobs[sample_idx].detach().cpu().numpy()
    y_aug = batch_yaug[sample_idx].detach().cpu().numpy()
    wavegrid = batch_wavegrid[sample_idx].detach().cpu().numpy()
    v_offset = batch_voffset[sample_idx].detach().cpu().numpy()
    r_obs = batch_robs[sample_idx].detach().cpu().numpy()
    r_aug = batch_raug[sample_idx].detach().cpu().numpy()
    b_obs = model.b_obs.detach().cpu().numpy()
    b_rest = model.b_rest.detach().cpu().numpy()
    y_rest = batch_yact[sample_idx].detach().cpu().numpy()
    y_act = batch_yact[sample_idx].detach().cpu().numpy()
    y_obs_prime = batch_yobsprime[sample_idx].detach().cpu().numpy()
    s_obs = batch_s[sample_idx].detach().cpu().numpy()
    s_aug = batch_s_aug[sample_idx].detach().cpu().numpy()
    v_encode = batch_vencode[sample_idx].detach().cpu().numpy()

    # Création du plot d'analyse complet
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    fig.suptitle(
        f"AESTRA Analysis - {exp_name} - {phase_name} - Epoch {epoch}\n"
        f"Sample {sample_idx} | True RV offset: {v_offset:.3f} m/s",
        fontsize=14,
        fontweight="bold",
    )

    # Plot 1: Spectres observés (y_obs vs y_aug)
    axes[0, 0].plot(
        wavegrid, y_obs, "b-", linewidth=1.5, alpha=0.8, label="y_obs (observed)"
    )
    axes[0, 0].plot(
        wavegrid, y_aug, "r-", linewidth=1.5, alpha=0.8, label="y_aug (augmented)"
    )
    axes[0, 0].set_xlabel("Wavelength (Å)")
    axes[0, 0].set_ylabel("Normalized Flux")
    axes[0, 0].set_title("Observed Spectra")
    axes[0, 0].legend()
    if zoom_line:
        axes[0, 0].set_xlim(most_weighted_line - halfwin, most_weighted_line + halfwin)
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Spectres résiduels (r_obs vs r_aug)
    axes[0, 1].plot(
        wavegrid, r_obs, "b-", linewidth=1.5, alpha=0.8, label="r_obs (residual)"
    )
    axes[0, 1].plot(
        wavegrid, r_aug, "r-", linewidth=1.5, alpha=0.8, label="r_aug (residual)"
    )
    axes[0, 1].axhline(y=0, color="k", linestyle="--", alpha=0.5)
    axes[0, 1].set_xlabel("Wavelength (Å)")
    axes[0, 1].set_ylabel("Residual Flux")
    axes[0, 1].set_title("Residual Spectra (after template subtraction)")
    axes[0, 1].legend()
    if zoom_line:
        axes[0, 1].set_xlim(most_weighted_line - halfwin, most_weighted_line + halfwin)
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Templates (b_obs vs b_rest)
    axes[1, 0].plot(
        wavegrid,
        b_obs,
        "g-",
        linewidth=2,
        alpha=0.8,
        label="b_obs (observed template)",
    )
    axes[1, 0].plot(
        wavegrid, b_rest, "m-", linewidth=2, alpha=0.8, label="b_rest (rest template)"
    )
    axes[1, 0].set_xlabel("Wavelength (Å)")
    axes[1, 0].set_ylabel("Normalized Flux")
    axes[1, 0].set_title("Templates")
    axes[1, 0].legend()
    if zoom_line:
        axes[1, 0].set_xlim(most_weighted_line - halfwin, most_weighted_line + halfwin)
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Spectre d'activité (y_act)
    axes[1, 1].plot(
        wavegrid, y_act, "orange", linewidth=2, alpha=0.8, label="y_act (activity)"
    )
    axes[1, 1].axhline(y=0, color="k", linestyle="--", alpha=0.5)
    axes[1, 1].set_xlabel("Wavelength (Å)")
    axes[1, 1].set_ylabel("Activity Flux")
    axes[1, 1].set_title("Activity Spectrum (decoded from latent)")
    axes[1, 1].legend()
    if zoom_line:
        axes[1, 1].set_xlim(most_weighted_line - halfwin, most_weighted_line + halfwin)
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 5: Modèle au repos (y_rest = y_act + b_rest)
    axes[2, 0].plot(
        wavegrid, y_rest, "purple", linewidth=2, alpha=0.8, label="y_rest (rest model)"
    )
    axes[2, 0].plot(wavegrid, b_rest, "m--", linewidth=1, alpha=0.6, label="b_rest")
    axes[2, 0].plot(wavegrid, y_act, "orange", linewidth=1, alpha=0.6, label="y_act")
    axes[2, 0].set_xlabel("Wavelength (Å)")
    axes[2, 0].set_ylabel("Normalized Flux")
    axes[2, 0].set_title("Rest-frame Model (y_rest = y_act + b_rest)")
    axes[2, 0].legend()
    if zoom_line:
        axes[2, 0].set_xlim(most_weighted_line - halfwin, most_weighted_line + halfwin)
    axes[2, 0].grid(True, alpha=0.3)

    # Plot 6: Reconstruction finale vs observé
    axes[2, 1].plot(
        wavegrid, y_obs, "b-", linewidth=2, alpha=0.8, label="y_obs (observed)"
    )
    axes[2, 1].plot(
        wavegrid,
        y_obs_prime,
        "r--",
        linewidth=2,
        alpha=0.8,
        label="y'_obs (reconstructed)",
    )
    axes[2, 1].set_xlabel("Wavelength (Å)")
    axes[2, 1].set_ylabel("Normalized Flux")
    axes[2, 1].set_title("Final Reconstruction")
    axes[2, 1].legend()
    if zoom_line:
        axes[2, 1].set_xlim(most_weighted_line - halfwin, most_weighted_line + halfwin)
    axes[2, 1].grid(True, alpha=0.3)

    # Plot 7: Résidus de reconstruction
    residual = y_obs - y_obs_prime
    axes[3, 0].plot(
        wavegrid, residual, "g-", linewidth=1.5, alpha=0.8, label="Residual"
    )
    axes[3, 0].axhline(y=0, color="k", linestyle="--", alpha=0.5)
    axes[3, 0].set_xlabel("Wavelength (Å)")
    axes[3, 0].set_ylabel("Flux Difference")
    axes[3, 0].set_title("Reconstruction Residual (y_obs - y'_obs)")
    axes[3, 0].legend()
    if zoom_line:
        axes[3, 0].set_xlim(most_weighted_line - halfwin, most_weighted_line + halfwin)
    axes[3, 0].grid(True, alpha=0.3)

    # Plot 8: Informations sur l'analyse
    axes[3, 1].axis("off")
    info_text = f"""AESTRA Analysis Summary:
    
• Sample: {sample_idx}/{batch_size - 1}
• True RV offset: {v_offset:.3f} m/s
• Estimated RV: {v_encode.item():.3f} m/s
• RV Error: {abs(v_offset - v_encode.item()):.3f} m/s

• Latent parameters (s): {s_obs}
• Augmented latent parameters (s_aug): {s_aug}
• s_obs - s_aug difference: {np.mean(np.abs(s_obs - s_aug)):.3E}
• Template b_obs range: [{model.b_obs.min():.3f}, {model.b_obs.max():.3f}]
• Template b_rest range: [{b_rest.min():.3f}, {b_rest.max():.3f}]
• Activity y_act range: [{y_act.min():.3f}, {y_act.max():.3f}]

• Reconstruction RMS: {np.sqrt(np.mean(residual**2)):.6f}
• Wavelength range: {wavegrid.min():.1f} - {wavegrid.max():.1f} Å
• Spectral resolution: {np.mean(np.diff(wavegrid)) * 1000:.1f} mÅ/pixel"""

    axes[3, 1].text(
        0.05,
        0.95,
        info_text,
        transform=axes[3, 1].transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
    )

    plt.tight_layout()

    # Sauvegarde
    filename = f"{exp_name}_{phase_name}_aestra_analysis_epoch_{epoch}.png"
    filepath = os.path.join(plot_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    # ⚠️ CRITIQUE: Nettoyage explicite des variables PyTorch pour libérer la mémoire GPU
    del batch_yobs, batch_yaug, batch_voffset, batch_wavegrid
    del batch_robs, batch_raug, batch_yact, batch_yact_aug, batch_s, batch_s_aug
    del batch_yrest, batch_vencode, batch_yobsprime
    
    # Force le garbage collection
    import gc
    gc.collect()
    
    # Libère le cache GPU si disponible
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"📊 AESTRA analysis saved: {filename}")


def plot_ultra_precise_doppler(
    batch: tuple,
    model: torch.nn.Module,
    exp_name: str,
    phase_name: str,
    epoch: int,
    plot_dir: str,
    device: str = "cpu",
    ultra_zoom_window: float = 0.2,  # Angstroms - PLUS petit que 0.5Å
) -> None:
    """
    Zoom ultra-précis encore plus serré pour visualiser les décalages Doppler.

    Args:
        ultra_zoom_window: Fenêtre de zoom en Angstroms (0.2Å par défaut, très serré)
    """
    from src.spectral_lines import find_best_lines_for_doppler

    os.makedirs(plot_dir, exist_ok=True)

    # Extraction des données
    batch_yobs, batch_yaug, batch_voffset, batch_wavegrid = batch
    sample_idx = np.random.randint(0, batch_yobs.shape[0])

    y_obs = batch_yobs[sample_idx].detach().cpu().numpy()
    y_aug = batch_yaug[sample_idx].detach().cpu().numpy()
    wavegrid = batch_wavegrid[sample_idx].detach().cpu().numpy()
    v_offset = batch_voffset[sample_idx].detach().cpu().numpy()

    # Trouver la meilleure raie
    wave_min, wave_max = wavegrid.min(), wavegrid.max()
    try:
        mask_filepath = "data/rv_datachallenge/masks/G2_mask.txt"
        selected_lines = find_best_lines_for_doppler(
            mask_filepath, (wave_min, wave_max), 1
        )
        if selected_lines:
            best_line = selected_lines[0]
        else:
            best_line = wavegrid[len(wavegrid) // 2]
    except Exception:
        best_line = wavegrid[len(wavegrid) // 2]

    # Zoom ultra-précis
    line_mask = (wavegrid >= best_line - ultra_zoom_window) & (
        wavegrid <= best_line + ultra_zoom_window
    )

    if not np.any(line_mask):
        print(f"⚠️  No data in ultra zoom window around {best_line:.3f} Å")
        return

    wave_zoom = wavegrid[line_mask]
    yobs_zoom = y_obs[line_mask]
    yaug_zoom = y_aug[line_mask]

    # Calculer les décalages théoriques
    c = 299792.458  # km/s
    expected_shift = best_line * (v_offset / 1000.0) / c  # en Å
    expected_wave = best_line + expected_shift

    # Créer le plot ultra-précis
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    fig.suptitle(
        f"{exp_name} - {phase_name} - Ultra-Precise Doppler Analysis (Epoch {epoch})",
        fontsize=14,
        fontweight="bold",
    )

    # Plots des spectres avec plus de détail
    ax.plot(
        wave_zoom, yobs_zoom, "b-", linewidth=3, alpha=0.9, label="Observed spectrum"
    )
    ax.plot(
        wave_zoom, yaug_zoom, "r-", linewidth=3, alpha=0.9, label="Augmented spectrum"
    )

    # Marquer la position de repos de la raie
    ax.axvline(
        best_line,
        color="gray",
        linestyle="--",
        alpha=0.8,
        linewidth=2,
        label=f"Rest wavelength: {best_line:.4f} Å",
    )

    # Marquer le décalage attendu
    if wave_zoom.min() <= expected_wave <= wave_zoom.max():
        ax.axvline(
            expected_wave,
            color="red",
            linestyle=":",
            alpha=0.9,
            linewidth=3,
            label=f"Expected shift: {expected_shift * 1000:.3f} mÅ ({v_offset:.2f} m/s)",
        )

    # Configuration des axes
    ax.set_xlabel("Wavelength (Å)", fontsize=12)
    ax.set_ylabel("Normalized Flux", fontsize=12)
    ax.set_title(
        f"Ultra-precise view: {best_line:.4f} Å (±{ultra_zoom_window:.1f} Å)",
        fontsize=12,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    # Informations détaillées
    pixel_resolution = np.mean(np.diff(wave_zoom)) * 1000  # en mÅ
    n_pixels = len(wave_zoom)

    analysis_text = f"""Ultra-Precise Doppler Analysis:

• Line center: {best_line:.4f} Å
• Zoom window: ±{ultra_zoom_window:.1f} Å ({n_pixels} pixels)
• Pixel resolution: {pixel_resolution:.2f} mÅ/pixel
• True RV offset: {v_offset:.3f} m/s
• Expected shift: {expected_shift * 1000:.4f} mÅ
• Shift in pixels: {expected_shift * 1000 / pixel_resolution:.2f} pixels

• Theoretical detectability:
  - Photon noise limited precision
  - Requires multiple exposures for averaging
  - Activity variations may dominate"""

    ax.text(
        0.02,
        0.98,
        analysis_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.7", facecolor="white", alpha=0.95),
    )

    plt.tight_layout()

    # Sauvegarde
    filename = f"{exp_name}_{phase_name}_ultra_doppler_epoch_{epoch}.png"
    filepath = os.path.join(plot_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    # ⚠️ CRITIQUE: Nettoyage de la mémoire GPU
    del batch_yobs, batch_yaug, batch_voffset, batch_wavegrid
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"🔬 Ultra-precise Doppler analysis saved: {filename}")


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


def plot_activity_perturbation(
    y_act_original,
    y_act_perturbed_list,
    wavelength,
    save_path=None,
    show_plot=False,
    wave_range=(5000, 5010),
):
    """
    Reproduit la Figure 2: Visualisation de l'effet de la perturbation du vecteur latent.

    Args:
        y_act_original: Spectre d'activité original (courbe noire)
        y_act_perturbed_list: Liste des spectres d'activité perturbés
        wavelength: Grille de longueurs d'onde
        save_path: Chemin de sauvegarde (optionnel)
        show_plot: Afficher le plot (False par défaut)
        wave_range: Gamme de longueurs d'onde à afficher (tuple)
    """
    # Couleurs pour les perturbations (comme dans la Figure 2)
    colors = ["purple", "cyan", "orange"]
    labels = ["Perturb s₁", "Perturb s₂", "Perturb s₃"]

    # Filtre pour la gamme de longueurs d'onde
    wave_mask = (wavelength >= wave_range[0]) & (wavelength <= wave_range[1])
    wave_filtered = wavelength[wave_mask]

    # Création de la figure avec subplots pour chaque perturbation
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(
        "Visualization of the effect of perturbing the latent vector",
        fontsize=14,
        fontweight="bold",
    )

    for i, (y_perturbed, color, label) in enumerate(
        zip(y_act_perturbed_list, colors, labels)
    ):
        ax = axes[i]

        # Filtrage des spectres pour la gamme de longueurs d'onde
        y_original_filtered = y_act_original[wave_mask]
        y_perturbed_filtered = y_perturbed[wave_mask]

        # Plot du spectre original (courbe noire)
        ax.plot(wave_filtered, y_original_filtered, "k-", linewidth=1.5, alpha=0.8)

        # Plot du spectre perturbé (courbe colorée)
        ax.plot(
            wave_filtered,
            y_perturbed_filtered,
            color=color,
            linewidth=1.5,
            alpha=0.9,
            label=label,
        )

        # Configuration des axes
        ax.set_ylabel("y_act", fontsize=11)
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)

        # Ajustement des limites y pour bien voir les différences
        y_min = min(y_original_filtered.min(), y_perturbed_filtered.min())
        y_max = max(y_original_filtered.max(), y_perturbed_filtered.max())
        margin = (y_max - y_min) * 0.1
        ax.set_ylim(y_min - margin, y_max + margin)

    # Configuration de l'axe x pour le dernier subplot
    axes[-1].set_xlabel("Restframe wavelength (Å)", fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure 2 sauvegardée: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()
