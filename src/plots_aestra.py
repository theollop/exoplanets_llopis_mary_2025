"""
Fonctions de plotting pour l'entraînement AESTRA.

Ce module contient les fonctions de visualisation optimisées pour AESTRA :
- Plotting des losses d'entraînement
- Visualisation des spectres selon le papier AESTRA
- Zoom ultra-précis pour l'analyse Doppler
- Plot 3D de l'espace latent
- Périodogramme des vitesses radiales
"""

import os
import gc
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Optional
from src.interpolate import shift_spectra_linear
from src.dataset import SpectrumDataset


def create_phase_plot_dir(plot_dir: str, phase_name: str) -> str:
    """
    Crée un sous-dossier pour une phase spécifique dans le répertoire de plots.

    Args:
        plot_dir: Répertoire de base des plots (ex: "experiments/exp1/figures")
        phase_name: Nom de la phase (ex: "rvonly", "joint")

    Returns:
        str: Chemin du sous-dossier de la phase
    """
    phase_dir = os.path.join(plot_dir, phase_name)
    os.makedirs(phase_dir, exist_ok=True)
    return phase_dir


def create_typed_plot_dir(plot_dir: str, phase_name: str, plot_type: str) -> str:
    """
    Crée un sous-dossier organisé par type de plot dans une phase spécifique.

    Args:
        plot_dir: Répertoire de base des plots (ex: "experiments/exp1/figures")
        phase_name: Nom de la phase (ex: "rvonly", "joint")
        plot_type: Type de plot (ex: "losses", "rv_predictions", "analysis", "activity")

    Returns:
        str: Chemin du sous-dossier organisé
    """
    typed_dir = os.path.join(plot_dir, phase_name, plot_type)
    os.makedirs(typed_dir, exist_ok=True)
    return typed_dir


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
    # Créer le sous-dossier organisé par type pour la phase
    typed_plot_dir = create_typed_plot_dir(plot_dir, phase_name, "losses")

    # Configurer la mosaïque (2x3 pour 6 plots)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        f"Training Losses - Epoch {epoch}",
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

    # Nom de fichier simplifié
    filename = f"losses_epoch_{epoch}.png"
    filepath = os.path.join(typed_plot_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    console.log(f"📊 Losses plot saved: {phase_name}/losses/{filename}")


def plot_rv_predictions_dataset(
    dataset,
    model: torch.nn.Module,
    exp_name: str,
    phase_name: str,
    epoch: int,
    plot_dir: str,
    batch_size: int = 32,
) -> None:
    """
    Calcule et trace les prédictions RV (r_obs = y_obs - b_obs) pour TOUT le dataset.

    Args:
        dataset: SpectrumDataset (utilise dataset.spectra et éventuellement dataset.time_values)
        model: Modèle AESTRA (utilise model.rvestimator et model.b_obs)
        exp_name: Nom de l'expérience
        phase_name: Phase d'entraînement
        epoch: Époch actuelle (pour le nom de fichier)
        plot_dir: Répertoire de sauvegarde
        batch_size: Taille des chunks pour éviter d'épuiser la mémoire
    """
    # Créer le sous-dossier organisé par type pour la phase
    typed_plot_dir = create_typed_plot_dir(plot_dir, phase_name, "rv_predictions")

    N = len(dataset)
    device = model.b_obs.device

    # Préserver l'état d'entraînement du modèle
    was_training = model.training
    model.eval()

    preds = []
    with torch.no_grad():
        for i in range(0, N, batch_size):
            j = min(i + batch_size, N)
            y_batch = dataset.spectra[i:j].to(device)
            r_batch = y_batch - model.b_obs.unsqueeze(0)
            v_batch = model.rvestimator(r_batch)
            preds.append(v_batch.detach().cpu())

    # Restaurer l'état précédent
    if was_training:
        model.train()
    else:
        model.eval()

    v_pred = torch.cat(preds, dim=0).numpy()
    idx = np.arange(N)

    # Axe temporel si disponible
    t = None
    try:
        t = dataset.time_values.detach().cpu().numpy()
    except Exception:
        t = None

    import matplotlib.pyplot as plt

    if t is not None and t.shape[0] == N:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        axes[0].plot(idx, v_pred, "b.-", alpha=0.85)
        axes[0].set_title("RV predictions vs index")
        axes[0].set_xlabel("Index")
        axes[0].set_ylabel("v_pred (m/s)")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(t, v_pred, "g.-", alpha=0.85)
        axes[1].set_title("RV predictions vs time")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("v_pred (m/s)")
        axes[1].grid(True, alpha=0.3)

        axes[2].hist(v_pred, bins=40, color="tab:blue", alpha=0.8)
        axes[2].set_title("Distribution of RV predictions")
        axes[2].set_xlabel("v_pred (m/s)")
        axes[2].set_ylabel("Count")
        axes[2].grid(True, alpha=0.3)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(idx, v_pred, "b.-", alpha=0.85)
        axes[0].set_title("RV predictions on full dataset")
        axes[0].set_xlabel("Index")
        axes[0].set_ylabel("v_pred (m/s)")
        axes[0].grid(True, alpha=0.3)

        axes[1].hist(v_pred, bins=40, color="tab:blue", alpha=0.8)
        axes[1].set_title("Distribution of RV predictions")
        axes[1].set_xlabel("v_pred (m/s)")
        axes[1].set_ylabel("Count")
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    # Nom de fichier simplifié
    filename = f"rv_predictions_full_epoch_{epoch}.png"
    filepath = os.path.join(typed_plot_dir, filename)
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close()


def plot_rv_predictions(
    batch: tuple,
    model: torch.nn.Module,
    exp_name: str,
    phase_name: str,
    epoch: int,
    plot_dir: str,
) -> None:
    """
    Plot des prédictions de vitesses radiales du RVEstimator sur les entrées r_obs = y_obs - b_obs.

    Args:
        batch: Batch de données (y_obs, y_aug, v_offset, wavegrid)
        model: Modèle AESTRA (utilise model.rvestimator et model.b_obs)
        exp_name: Nom de l'expérience
        phase_name: Phase d'entraînement
        epoch: Époch actuelle (pour le nom de fichier)
        plot_dir: Répertoire de sauvegarde
    """
    # Créer le sous-dossier organisé par type pour la phase
    typed_plot_dir = create_typed_plot_dir(plot_dir, phase_name, "rv_predictions")

    batch_yobs, _, _, _ = batch

    # Préserver l'état d'entraînement du modèle
    was_training = model.training

    with torch.no_grad():
        batch_robs = batch_yobs - model.b_obs.unsqueeze(0)
        v_pred = model.rvestimator(batch_robs).detach().cpu().numpy()

    # Restaurer l'état précédent
    if was_training:
        model.train()
    else:
        model.eval()

    idx = np.arange(len(v_pred))

    import matplotlib.pyplot as plt  # local import to avoid issues if backend changes

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Courbe des prédictions (par ordre de batch)
    axes[0].plot(idx, v_pred, "b.-", alpha=0.85)
    axes[0].set_title("RV predictions on r_obs = y_obs - b_obs")
    axes[0].set_xlabel("Batch index")
    axes[0].set_ylabel("v_pred (m/s)")
    axes[0].grid(True, alpha=0.3)

    # Histogramme des prédictions
    axes[1].hist(v_pred, bins=30, color="tab:blue", alpha=0.75)
    axes[1].set_title("Distribution of RV predictions")
    axes[1].set_xlabel("v_pred (m/s)")
    axes[1].set_ylabel("Count")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    # Nom de fichier simplifié
    filename = f"rv_predictions_batch_epoch_{epoch}.png"
    filepath = os.path.join(typed_plot_dir, filename)
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close()

    # Pas de console ici pour éviter une dépendance circulaire avec train.py


def plot_aestra_analysis(
    batch: tuple,
    dataset: SpectrumDataset,
    model: torch.nn.Module,
    exp_name: str,
    phase_name: str,
    epoch: int,
    plot_dir: str,
    sample_idx: Optional[int] = None,
    zoom_line: bool = False,
    data_root_dir: str = "data",
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
        sample_idx: Index de l'échantillon (None pour aléatoire)
        zoom_line: Activer le zoom sur une raie spectrale
        data_root_dir: Répertoire racine des données (par défaut "data")
    """
    # Créer le sous-dossier organisé par type pour la phase
    typed_plot_dir = create_typed_plot_dir(plot_dir, phase_name, "analysis")

    # Extraction des données du batch
    batch_yobs, batch_yaug, batch_voffset, batch_wavegrid = batch
    batch_size = batch_yobs.shape[0]

    g2mask = np.loadtxt(f"{data_root_dir}/rv_datachallenge/masks/G2_mask.txt")
    line_positions, line_weights = g2mask[:, 0], g2mask[:, 1]
    wavegrid = batch_wavegrid[0].detach().cpu().numpy()
    line_weights = line_weights[
        (line_positions >= wavegrid.min()) & (line_positions <= wavegrid.max())
    ]
    line_positions = line_positions[
        (line_positions >= wavegrid.min()) & (line_positions <= wavegrid.max())
    ]
    # Sélectionner une ligne parmi les 10 lignes les plus weighted
    top_indices = np.argsort(line_weights)[-10:]
    # selected_line = np.random.choice(line_positions[top_indices]) # Choix random
    selected_line = line_positions[top_indices][
        1
    ]  # Choix de la première ligne des 10 les plus weighted
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
    # Use the correct rest-frame model for the sample
    y_rest = batch_yrest[sample_idx].detach().cpu().numpy()
    y_act = batch_yact[sample_idx].detach().cpu().numpy()
    y_obs_prime = batch_yobsprime[sample_idx].detach().cpu().numpy()
    s_obs = batch_s[sample_idx].detach().cpu().numpy()
    s_aug = batch_s_aug[sample_idx].detach().cpu().numpy()
    v_encode = batch_vencode[sample_idx].detach().cpu().numpy()

    yact_true = (
        dataset.activity[sample_idx].detach().cpu().numpy()
        if hasattr(dataset, "activity")
        else None
    )

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
        axes[0, 0].set_xlim(selected_line - halfwin, selected_line + halfwin)
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
        axes[0, 1].set_xlim(selected_line - halfwin, selected_line + halfwin)
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
        axes[1, 0].set_xlim(selected_line - halfwin, selected_line + halfwin)
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Spectre d'activité (y_act)
    axes[1, 1].plot(
        wavegrid, y_act, "orange", linewidth=2, alpha=0.8, label="y_act (activity)"
    )
    if yact_true is not None:
        # Si un template de référence est fourni et la RV vraie, tracer l'activité vraie
        axes[1, 1].plot(
            wavegrid,
            yact_true,
            "k",
            linewidth=1.5,
            alpha=0.6,
            label="True Activity",
        )
    axes[1, 1].axhline(y=0, color="k", linestyle="--", alpha=0.5)
    axes[1, 1].set_xlabel("Wavelength (Å)")
    axes[1, 1].set_ylabel("Activity Flux")
    axes[1, 1].set_title("Activity Spectrum (decoded from latent)")
    axes[1, 1].legend()
    if zoom_line:
        axes[1, 1].set_xlim(selected_line - halfwin, selected_line + halfwin)
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
        axes[2, 0].set_xlim(selected_line - halfwin, selected_line + halfwin)
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
        axes[2, 1].set_xlim(selected_line - halfwin, selected_line + halfwin)
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
        axes[3, 0].set_xlim(selected_line - halfwin, selected_line + halfwin)
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

    # Nom de fichier simplifié
    filename = f"aestra_analysis_epoch_{epoch}.png"
    filepath = os.path.join(typed_plot_dir, filename)
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
    data_root_dir: str = "data",
) -> None:
    """
    Zoom ultra-précis encore plus serré pour visualiser les décalages Doppler.

    Args:
        ultra_zoom_window: Fenêtre de zoom en Angstroms (0.2Å par défaut, très serré)
        data_root_dir: Répertoire racine des données (par défaut "data")
    """
    from src.spectral_lines import find_best_lines_for_doppler

    # Créer le sous-dossier organisé par type pour la phase
    typed_plot_dir = create_typed_plot_dir(plot_dir, phase_name, "ultra_doppler")

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
        mask_filepath = f"{data_root_dir}/rv_datachallenge/masks/G2_mask.txt"
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

    # Nom de fichier simplifié
    filename = f"ultra_doppler_epoch_{epoch}.png"
    filepath = os.path.join(typed_plot_dir, filename)
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


def plot_latent_space_3d(
    latent_s,
    rv_values,
    save_path=None,  # Sera déterminé automatiquement depuis l'expérience
    show_plot=False,
    decorrelated=False,
):
    """
    Crée un plot 3D de l'espace latent avec projections 2D, coloré selon les valeurs RV.
    Ne fonctionne que si les vecteurs latents sont 3D ou plus.

    Args:
        latent_s: Array numpy des vecteurs latents (N, D)
        rv_values: Array numpy des valeurs RV correspondantes (N,)
        save_path: Chemin pour sauvegarder la figure (None pour utiliser le répertoire par défaut)
        show_plot: Afficher la figure ou non

    Returns:
        bool: True si le plot a été créé, False si l'espace latent n'est pas 3D
    """
    print(f"Dimension de l'espace latent: {latent_s.shape[1]}")

    # Vérifier si l'espace latent est au moins 3D
    if latent_s.shape[1] < 3:
        print(f"⚠️  L'espace latent n'est que {latent_s.shape[1]}D, plot 3D impossible")
        return False

    # Détermination automatique du chemin de sauvegarde si non fourni
    if save_path is None:
        save_path = "reports/figures/latent_space_3d.png"

    # Prendre les 3 premières dimensions
    s1, s2, s3 = latent_s[:, 0], latent_s[:, 1], latent_s[:, 2]

    # Utilisation directe des valeurs RV (les outliers sont déjà supprimés en amont)
    print(f"RV range: [{np.min(rv_values):.3f}, {np.max(rv_values):.3f}] m/s")
    print(f"Nombre de spectres: {len(rv_values)}")

    # Création de la figure avec 4 sous-plots (1 en 3D + 3 projections 2D)
    fig = plt.figure(figsize=(20, 15))

    # Plot 3D principal (occupant la partie gauche)
    ax_3d = fig.add_subplot(2, 3, (1, 4), projection="3d")

    # Scatter plot 3D coloré selon les valeurs RV
    scatter_3d = ax_3d.scatter(
        s1, s2, s3, c=rv_values, cmap="viridis", s=20, alpha=0.7, edgecolors="none"
    )

    ax_3d.set_xlabel("S₁")
    ax_3d.set_ylabel("S₂")
    ax_3d.set_zlabel("S₃")
    ax_3d.set_title(
        f"Espace latent 3D coloré par V_encode [m/s] {'DECORRELATED' if decorrelated else ''}"
    )

    # Colorbar pour le plot 3D
    cbar_3d = plt.colorbar(scatter_3d, ax=ax_3d, shrink=0.6)
    cbar_3d.set_label("V_encode [m/s]")

    # Fonction pour calculer et tracer la corrélation
    def add_correlation_analysis(ax, x_data, y_data, x_label, y_label):
        """Ajoute la corrélation et la droite de régression à un subplot."""
        # Calcul de la corrélation de Pearson
        correlation = np.corrcoef(x_data, y_data)[0, 1]

        # Régression linéaire (polyfit de degré 1)
        slope, intercept = np.polyfit(x_data, y_data, 1)

        # Points pour tracer la droite de régression
        x_range = np.linspace(x_data.min(), x_data.max(), 100)
        y_fit = slope * x_range + intercept

        # Tracer la droite de régression en rouge pointillé
        ax.plot(
            x_range,
            y_fit,
            "r--",
            linewidth=2,
            alpha=0.8,
            label=f"R={correlation:.3f}, slope={slope:.3f}",
        )

        # Mise à jour du titre avec la corrélation
        current_title = ax.get_title()
        ax.set_title(f"{current_title}\nR={correlation:.3f}")

        # Ajout de la légende
        ax.legend(fontsize=8, loc="best")

        return correlation, slope

    # Projections 2D avec corrélations
    # Projection S1 vs S2
    ax_12 = fig.add_subplot(2, 3, 2)
    ax_12.scatter(
        s1, s2, c=rv_values, cmap="viridis", s=15, alpha=0.7, edgecolors="none"
    )
    ax_12.set_xlabel("S₁")
    ax_12.set_ylabel("S₂")
    ax_12.set_title("Projection S₁-S₂")
    ax_12.grid(True, alpha=0.3)
    corr_12, slope_12 = add_correlation_analysis(ax_12, s1, s2, "S₁", "S₂")

    # Projection S1 vs S3
    ax_13 = fig.add_subplot(2, 3, 3)
    ax_13.scatter(
        s1, s3, c=rv_values, cmap="viridis", s=15, alpha=0.7, edgecolors="none"
    )
    ax_13.set_xlabel("S₁")
    ax_13.set_ylabel("S₃")
    ax_13.set_title("Projection S₁-S₃")
    ax_13.grid(True, alpha=0.3)
    corr_13, slope_13 = add_correlation_analysis(ax_13, s1, s3, "S₁", "S₃")

    # Projection S2 vs S3
    ax_23 = fig.add_subplot(2, 3, 6)
    ax_23.scatter(
        s2, s3, c=rv_values, cmap="viridis", s=15, alpha=0.7, edgecolors="none"
    )
    ax_23.set_xlabel("S₂")
    ax_23.set_ylabel("S₃")
    ax_23.set_title("Projection S₂-S₃")
    ax_23.grid(True, alpha=0.3)
    corr_23, slope_23 = add_correlation_analysis(ax_23, s2, s3, "S₂", "S₃")

    # Histogramme des valeurs RV
    ax_hist = fig.add_subplot(2, 3, 5)

    # Histogramme des valeurs RV
    ax_hist.hist(
        rv_values,
        bins=50,
        alpha=0.8,
        color="skyblue",
        edgecolor="black",
        label="Distribution RV",
    )

    ax_hist.set_xlabel("V_encode [m/s]")
    ax_hist.set_ylabel("Fréquence")
    ax_hist.set_title("Distribution des vitesses radiales")
    ax_hist.grid(True, alpha=0.3)
    ax_hist.legend(fontsize=8)

    # Calcul des corrélations avec les valeurs RV
    corr_s1_rv = np.corrcoef(s1, rv_values)[0, 1]
    corr_s2_rv = np.corrcoef(s2, rv_values)[0, 1]
    corr_s3_rv = np.corrcoef(s3, rv_values)[0, 1]

    # Statistiques textuelles avec corrélations
    stats_text = f"""Statistiques:
N spectres: {len(rv_values)}
RV:
  Min: {np.min(rv_values):.3f} m/s
  Max: {np.max(rv_values):.3f} m/s
  Mean: {np.mean(rv_values):.3f} m/s
  Std: {np.std(rv_values):.3f} m/s
Dim latente: {latent_s.shape[1]}D

Corrélations entre dimensions:
  S₁-S₂: R={corr_12:.3f}
  S₁-S₃: R={corr_13:.3f}
  S₂-S₃: R={corr_23:.3f}

Corrélations avec RV:
  S₁-RV: R={corr_s1_rv:.3f}
  S₂-RV: R={corr_s2_rv:.3f}
  S₃-RV: R={corr_s3_rv:.3f}"""

    ax_hist.text(
        0.05,
        0.95,
        stats_text,
        transform=ax_hist.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()

    # Sauvegarde
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot 3D de l'espace latent sauvegardé: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_rv_periodogram(
    periods,
    power,
    rv_values,
    times,
    known_periods=None,
    save_path=None,  # Sera déterminé automatiquement depuis l'expérience
    show_plot=False,
    periods_corr=None,
    power_corr=None,
    rv_corrected=None,
    decorrelate_applied=False,
):
    """
    Trace le périodogramme des vitesses radiales avec des zooms sur les périodes d'intérêt.
    Supporte la comparaison entre RV originales et décorrélées.

    Args:
        periods: Périodes en jours (RV originales)
        power: Puissance du périodogramme (RV originales)
        rv_values: Valeurs de vitesses radiales originales
        times: Temps JDB
        known_periods: Liste des périodes connues des planètes (optionnel)
        save_path: Chemin pour sauvegarder la figure (None pour utiliser le répertoire par défaut)
        show_plot: Afficher la figure ou non
        periods_corr: Périodes pour RV décorrélées (optionnel)
        power_corr: Puissance pour RV décorrélées (optionnel)
        rv_corrected: RV décorrélées (optionnel)
        decorrelate_applied: Si True, affiche la comparaison entre original et décorrélé
    """

    # Détermination automatique du chemin de sauvegarde si non fourni
    if save_path is None:
        save_path = "reports/figures/rv_periodogram_with_known_periods.png"

    # Création de la figure avec plusieurs sous-graphiques
    if decorrelate_applied and power_corr is not None:
        # Mode comparaison avec décorrélation et zooms sur périodes connues
        fig = plt.figure(figsize=(20, 16))

        # 1. Périodogrammes superposés (haut, large)
        ax1 = plt.subplot(4, 3, (1, 3))
        ax1.semilogx(periods, power, "b-", linewidth=1.0, label="RV originales")
        ax1.semilogx(
            periods_corr, power_corr, "r-", linewidth=1.0, label="RV décorrélées"
        )
        ax1.set_xlabel("Période (jours)")
        ax1.set_ylabel("Puissance LS")
        ax1.set_title("Comparaison des périodogrammes - RV originales vs décorrélées")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Marquer les périodes connues
        if known_periods is not None and len(known_periods) > 0:
            for i, period in enumerate(known_periods):
                if periods.min() <= period <= periods.max():
                    ax1.axvline(
                        period,
                        color="green",
                        linestyle="--",
                        alpha=0.8,
                        linewidth=2,
                        label=f"Planète {i + 1}: {period:.1f}j" if i < 3 else "",
                    )
            if len(known_periods) <= 3:
                ax1.legend()

        # 2. Zooms sur les périodes connues (ligne du milieu)
        if known_periods is not None and len(known_periods) > 0:
            for i, period in enumerate(known_periods[:3]):  # Max 3 zooms
                ax_zoom = plt.subplot(4, 3, 4 + i)

                # Fenêtre de zoom : ±20% autour de la période
                zoom_min = period * 0.8
                zoom_max = period * 1.2
                zoom_mask = (periods >= zoom_min) & (periods <= zoom_max)
                zoom_mask_corr = (periods_corr >= zoom_min) & (periods_corr <= zoom_max)

                if np.any(zoom_mask) and np.any(zoom_mask_corr):
                    ax_zoom.plot(
                        periods[zoom_mask],
                        power[zoom_mask],
                        "b-",
                        linewidth=1.5,
                        label="Original",
                    )
                    ax_zoom.plot(
                        periods_corr[zoom_mask_corr],
                        power_corr[zoom_mask_corr],
                        "r-",
                        linewidth=1.5,
                        label="Décorrélé",
                    )
                    ax_zoom.axvline(
                        period, color="green", linestyle="--", alpha=0.8, linewidth=2
                    )
                    ax_zoom.set_xlabel("Période (jours)")
                    ax_zoom.set_ylabel("Puissance LS")
                    ax_zoom.set_title(f"Zoom Planète {i + 1}: {period:.1f}j")
                    ax_zoom.grid(True, alpha=0.3)
                    if i == 0:
                        ax_zoom.legend(fontsize=8)

        # 3. RV originales vs temps (bas gauche)
        ax2 = plt.subplot(4, 2, 7)
        ax2.plot(times, rv_values, "b-", linewidth=0.8, alpha=0.8)
        ax2.scatter(
            times[::20], rv_values[::20], c="blue", s=10, alpha=0.6
        )  # Points espacés
        ax2.set_xlabel("Temps (JDB)")
        ax2.set_ylabel("RV originales (m/s)")
        ax2.set_title(f"RV originales vs temps\n(std = {np.std(rv_values):.4f} m/s)")
        ax2.grid(True, alpha=0.3)

        # 4. RV décorrélées vs temps (bas droite)
        ax3 = plt.subplot(4, 2, 8)
        ax3.plot(times, rv_corrected, "r-", linewidth=0.8, alpha=0.8)
        ax3.scatter(
            times[::20], rv_corrected[::20], c="red", s=10, alpha=0.6
        )  # Points espacés
        ax3.set_xlabel("Temps (JDB)")
        ax3.set_ylabel("RV décorrélées (m/s)")
        ax3.set_title(
            f"RV décorrélées vs temps\n(std = {np.std(rv_corrected):.4f} m/s)"
        )
        ax3.grid(True, alpha=0.3)

    elif known_periods is not None and len(known_periods) > 0:
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
                    ax_zoom.set_ylabel("Puissance")
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


def plot_ccf_analysis(
    v_grid: np.ndarray,
    ccf: np.ndarray,
    analysis_results: dict,
    spectrum_idx: int = 0,
    save_path: Optional[str] = None,
    show_plot: bool = True,
    all_ccfs: Optional[np.ndarray] = None,  # (N, M) pour halo min–max
    halo_alpha: float = 0.15,  # transparence du halo
):
    """
    Visualise une CCF avec son analyse complète (fit gaussien, FWHM, bissector span).

    Args:
        v_grid: Grille de vitesses en m/s
        ccf: Profil CCF à visualiser (1D) ou matrice (N, M). Si 2D, la première ligne est tracée et l'ensemble sert au halo
        analysis_results: Résultats de l'analyse CCF (dict avec rv, depth, fwhm, span, etc.)
        spectrum_idx: Index du spectre analysé (pour le titre)
        save_path: Chemin de sauvegarde du plot (optionnel)
        show_plot: Si True, affiche le plot
        all_ccfs: Optionnel, tableau/list de toutes les CCFs (N, M) pour tracer un halo (enveloppe min–max) sous la CCF principale
        halo_alpha: Transparence du halo (0–1)
    """
    from src.ccf import gaussian

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f"Analyse CCF complète - Spectre {spectrum_idx}", fontsize=14, fontweight="bold"
    )

    # Plot 1: CCF et fit gaussien (haut gauche)
    ax1 = axes[0, 0]

    # Préparer CCF principale et éventuel ensemble pour halo
    ccf_main = ccf
    ccf_stack = None

    # Si l'utilisateur passe ccf en 2D, utiliser la 1ère comme principale et l'ensemble pour halo
    if isinstance(ccf, np.ndarray) and ccf.ndim == 2:
        if ccf.shape[1] == len(v_grid):
            ccf_stack = ccf
            ccf_main = ccf[0]
        else:
            # Dimensions incompatibles, on ignore le halo
            ccf_main = ccf[0]
            ccf_stack = None

    # Paramètre dédié all_ccfs a priorité si fourni
    if all_ccfs is not None:
        try:
            ccf_stack = np.asarray(all_ccfs)
            if ccf_stack.ndim == 1:
                ccf_stack = ccf_stack[None, :]
            # Vérifier compatibilité
            if ccf_stack.shape[1] != len(v_grid):
                ccf_stack = None  # incompatibilité → pas de halo
        except Exception:
            ccf_stack = None

    # Tracer le halo (enveloppe min–max) avant la courbe principale pour qu'il soit dessous
    if ccf_stack is not None and ccf_stack.shape[0] > 1:
        ccf_min = np.nanmin(ccf_stack, axis=0)
        ccf_max = np.nanmax(ccf_stack, axis=0)
        ax1.fill_between(
            v_grid,
            ccf_min,
            ccf_max,
            color="b",
            alpha=halo_alpha,
            linewidth=0,
            zorder=0,
            label="Halo CCFs (min–max)",
        )

    # Tracer la CCF principale
    ax1.plot(
        v_grid, ccf_main, "b-", linewidth=1.5, label="CCF observée", alpha=0.7, zorder=1
    )

    # Fit gaussien si disponible
    if "popt" in analysis_results and not np.any(np.isnan(analysis_results["popt"])):
        v_fine = np.linspace(v_grid.min(), v_grid.max(), 1000)
        ccf_fit = gaussian(v_fine, *analysis_results["popt"])
        ax1.plot(v_fine, ccf_fit, "r-", linewidth=2, label="Fit gaussien", zorder=2)

        # Marquer le centre de la gaussienne (RV)
        rv = analysis_results["rv"]
        if not np.isnan(rv):
            rv_idx = np.argmin(np.abs(v_grid - rv))
            ax1.axvline(
                rv,
                color="red",
                linestyle="--",
                alpha=0.8,
                label=f"RV = {rv:.1f} m/s",
                zorder=2,
            )
            # Sécurité si la CCF principale était 2D/halo
            y_min_marker = (
                ccf_main[rv_idx] if np.ndim(ccf_main) == 1 else ccf[0, rv_idx]
            )
            ax1.plot(
                rv, y_min_marker, "ro", markersize=8, label="Minimum CCF", zorder=3
            )

        # Marquer la FWHM
        fwhm = analysis_results["fwhm"]
        if not np.isnan(fwhm):
            half_depth = (
                analysis_results["continuum"] + analysis_results["amplitude"] / 2
            )
            ax1.axhline(
                half_depth,
                color="orange",
                linestyle=":",
                alpha=0.8,
                label=f"FWHM = {fwhm:.1f} m/s",
            )
            ax1.axvspan(rv - fwhm / 2, rv + fwhm / 2, alpha=0.2, color="orange")

    ax1.set_xlabel("Vitesse radiale (m/s)")
    ax1.set_ylabel("CCF")
    ax1.set_title("CCF et fit gaussien")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Bissector (haut droite)
    ax2 = axes[0, 1]

    # Calculer et tracer le bissector si le fit est disponible
    if "popt" in analysis_results and not np.any(np.isnan(analysis_results["popt"])):
        try:
            # Paramètres du fit
            c, k, x0, fwhm = analysis_results["popt"]

            # Calcul du bissector (adapté de calculate_bissector_span)
            nstep = 100
            margin = 5
            len_depth = nstep - 2 * margin + 1

            # CCF normalisée
            norm_CCF = -c / k * (1 - ccf / c)
            depth_levels = np.array([(i + margin) / nstep for i in range(len_depth)])

            # Calcul simplifié du bissector
            ind_max = np.argmax(norm_CCF)
            bis = np.zeros(len_depth)

            for i in range(len_depth):
                # Recherche des indices gauche et droite pour chaque niveau de profondeur
                current_depth = depth_levels[i]

                # Trouver les points où la CCF normalisée = current_depth
                left_indices = np.where((norm_CCF[:ind_max] <= current_depth))[0]
                right_indices = (
                    np.where((norm_CCF[ind_max:] <= current_depth))[0] + ind_max
                )

                if len(left_indices) > 0 and len(right_indices) > 0:
                    # Interpolation linéaire pour une position plus précise
                    if len(left_indices) > 1:
                        idx_left = left_indices[-1]
                        if idx_left > 0:
                            # Interpolation entre les deux points
                            alpha = (current_depth - norm_CCF[idx_left - 1]) / (
                                norm_CCF[idx_left] - norm_CCF[idx_left - 1]
                            )
                            v_left = v_grid[idx_left - 1] + alpha * (
                                v_grid[idx_left] - v_grid[idx_left - 1]
                            )
                        else:
                            v_left = v_grid[idx_left]
                    else:
                        v_left = v_grid[left_indices[0]]

                    if len(right_indices) > 1:
                        idx_right = right_indices[0]
                        if idx_right < len(v_grid) - 1:
                            # Interpolation entre les deux points
                            alpha = (current_depth - norm_CCF[idx_right]) / (
                                norm_CCF[idx_right + 1] - norm_CCF[idx_right]
                            )
                            v_right = v_grid[idx_right] + alpha * (
                                v_grid[idx_right + 1] - v_grid[idx_right]
                            )
                        else:
                            v_right = v_grid[idx_right]
                    else:
                        v_right = v_grid[right_indices[0]]

                    # Position du bissector = milieu entre gauche et droite
                    bis[i] = (v_left + v_right) / 2
                else:
                    bis[i] = np.nan

            # Tracer le bissector
            valid_mask = ~np.isnan(bis)
            if np.any(valid_mask):
                ax2.plot(
                    bis[valid_mask],
                    depth_levels[valid_mask],
                    "g-",
                    linewidth=2,
                    label="Bissector",
                )

                # Marquer les zones pour le calcul du span
                top_mask = (depth_levels >= 0.1) & (depth_levels <= 0.4) & valid_mask
                bottom_mask = (depth_levels >= 0.6) & (depth_levels <= 0.9) & valid_mask

                if np.any(top_mask):
                    ax2.plot(
                        bis[top_mask],
                        depth_levels[top_mask],
                        "ro",
                        markersize=4,
                        alpha=0.7,
                        label="Zone haute (10-40%)",
                    )
                    ax2.axhspan(0.1, 0.4, alpha=0.2, color="red")

                if np.any(bottom_mask):
                    ax2.plot(
                        bis[bottom_mask],
                        depth_levels[bottom_mask],
                        "bo",
                        markersize=4,
                        alpha=0.7,
                        label="Zone basse (60-90%)",
                    )
                    ax2.axhspan(0.6, 0.9, alpha=0.2, color="blue")

                # Afficher le span calculé
                span = analysis_results.get("span", np.nan)
                if not np.isnan(span):
                    # Calculer les moyennes pour visualiser le span
                    if np.any(top_mask) and np.any(bottom_mask):
                        bis_top_mean = np.mean(bis[top_mask])
                        bis_bottom_mean = np.mean(bis[bottom_mask])

                        # Ligne horizontale pour visualiser le span
                        ax2.annotate(
                            "",
                            xy=(bis_top_mean, 0.25),
                            xytext=(bis_bottom_mean, 0.75),
                            arrowprops=dict(arrowstyle="<->", color="purple", lw=2),
                        )
                        ax2.text(
                            0.02,
                            0.95,
                            f"Span = {span:.2f} m/s",
                            transform=ax2.transAxes,
                            bbox=dict(
                                boxstyle="round,pad=0.3", facecolor="purple", alpha=0.3
                            ),
                        )

        except Exception as e:
            ax2.text(
                0.5,
                0.5,
                f"Erreur calcul bissector:\n{str(e)}",
                transform=ax2.transAxes,
                ha="center",
                va="center",
                color="red",
            )
    else:
        ax2.text(
            0.5,
            0.5,
            "Fit gaussien non disponible",
            transform=ax2.transAxes,
            ha="center",
            va="center",
            color="red",
        )

    ax2.set_xlabel("Vitesse radiale (m/s)")
    ax2.set_ylabel("Profondeur normalisée")
    ax2.set_title("Bissector et calcul du span")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    # Plot 3: Résidus du fit (bas gauche)
    ax3 = axes[1, 0]
    if "popt" in analysis_results and not np.any(np.isnan(analysis_results["popt"])):
        ccf_model = gaussian(v_grid, *analysis_results["popt"])
        residuals = ccf - ccf_model
        ax3.plot(v_grid, residuals, "g-", linewidth=1, label="Résidus (CCF - modèle)")
        ax3.axhline(0, color="black", linestyle="--", alpha=0.5)

        # RMS des résidus
        rms = np.sqrt(np.mean(residuals**2))
        ax3.text(
            0.02,
            0.95,
            f"RMS résidus = {rms:.4f}",
            transform=ax3.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"),
        )
    else:
        ax3.text(
            0.5,
            0.5,
            "Pas de fit disponible",
            transform=ax3.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color="red",
        )

    ax3.set_xlabel("Vitesse radiale (m/s)")
    ax3.set_ylabel("Résidus")
    ax3.set_title("Résidus du fit gaussien")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Plot 4: Paramètres et statistiques (bas droite)
    ax4 = axes[1, 1]
    ax4.axis("off")

    # Ajouter les paramètres d'analyse
    params_text = ["Paramètres CCF:\n"]
    if not np.isnan(analysis_results.get("rv", np.nan)):
        params_text.append(f"• RV: {analysis_results['rv']:.2f} m/s")
    if not np.isnan(analysis_results.get("depth", np.nan)):
        params_text.append(f"• Profondeur: {analysis_results['depth']:.4f}")
    if not np.isnan(analysis_results.get("fwhm", np.nan)):
        params_text.append(f"• FWHM: {analysis_results['fwhm']:.2f} m/s")
    if not np.isnan(analysis_results.get("span", np.nan)):
        params_text.append(f"• Bissector Span: {analysis_results['span']:.2f} m/s")
    if not np.isnan(analysis_results.get("continuum", np.nan)):
        params_text.append(f"• Continuum: {analysis_results['continuum']:.4f}")
    if not np.isnan(analysis_results.get("amplitude", np.nan)):
        params_text.append(f"• Amplitude: {analysis_results['amplitude']:.4f}")

    params_text.append("\nInterpretation:")
    span_val = analysis_results.get("span", np.nan)
    if not np.isnan(span_val):
        if abs(span_val) < 10:
            params_text.append("• Span faible → Étoile calme")
        elif abs(span_val) < 50:
            params_text.append("• Span modéré → Activité stellaire")
        else:
            params_text.append("• Span élevé → Forte activité")

    depth_val = analysis_results.get("depth", np.nan)
    if not np.isnan(depth_val):
        if depth_val > 0.05:
            params_text.append("• Profondeur élevée → Bon signal")
        elif depth_val > 0.01:
            params_text.append("• Profondeur modérée")
        else:
            params_text.append("• Profondeur faible → Signal bruité")

    ax4.text(
        0.05,
        0.95,
        "\n".join(params_text),
        transform=ax4.transAxes,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
    )

    plt.tight_layout()

    # Sauvegarde
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot CCF sauvegardé: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_activity(
    batch: tuple,
    dataset: SpectrumDataset,
    model: torch.nn.Module,
    exp_name: str,
    phase_name: str,
    epoch: int,
    plot_dir: str,
    sample_idx: Optional[int] = None,
    data_root_dir: str = "data",
) -> None:
    """
    Plot de comparaison entre l'activité vraie et l'activité prédite par AESTRA.

    Structure du plot :
    - 1ère ligne : spectre d'activité complet (vraie vs prédite)
    - 2ème ligne : zoom sur les 3 raies les plus importantes (weightées)

    Args:
        batch: Batch de données (y_obs, y_aug, v_offset, wavegrid)
        dataset: Dataset contenant l'activité vraie
        model: Modèle AESTRA
        exp_name: Nom de l'expérience
        phase_name: Phase d'entraînement
        epoch: Époch actuelle
        plot_dir: Répertoire de sauvegarde
        sample_idx: Index de l'échantillon (None pour aléatoire)
        data_root_dir: Répertoire racine des données (par défaut "data")
    """
    # Créer le sous-dossier organisé par type pour la phase
    typed_plot_dir = create_typed_plot_dir(plot_dir, phase_name, "activity")

    # Extraction des données du batch
    batch_yobs, batch_yaug, batch_voffset, batch_wavegrid = batch
    batch_size = batch_yobs.shape[0]

    # Chargement du masque G2 pour les raies importantes
    g2mask = np.loadtxt(f"{data_root_dir}/rv_datachallenge/masks/G2_mask.txt")
    line_positions, line_weights = g2mask[:, 0], g2mask[:, 1]
    wavegrid = batch_wavegrid[0].detach().cpu().numpy()

    # Filtrer les raies dans la plage spectrale
    mask_in_range = (line_positions >= wavegrid.min()) & (
        line_positions <= wavegrid.max()
    )
    line_weights = line_weights[mask_in_range]
    line_positions = line_positions[mask_in_range]

    # Sélectionner les 3 raies les plus importantes
    top_indices = np.argsort(line_weights)[-3:]  # Les 3 plus fortes
    selected_lines = line_positions[top_indices]
    halfwin = 0.18  # Fenêtre de zoom de 0.18 Å comme dans plot_aestra_analysis

    # Sélection d'un échantillon
    if sample_idx is None:
        sample_idx = np.random.randint(0, batch_size)

    # Vérifier que le dataset a bien l'activité vraie
    if not hasattr(dataset, "activity"):
        print(
            "⚠️ Warning: Dataset doesn't have 'activity' attribute. Cannot plot activity comparison."
        )
        return

    # Forward pass du modèle pour obtenir l'activité prédite
    model.eval()
    with torch.no_grad():
        # Spectres résiduels (après soustraction des templates)
        batch_robs = batch_yobs - model.b_obs.unsqueeze(0)

        # Encodage + Décodage pour obtenir le spectre d'activité prédit
        batch_yact, batch_s = model.spender(batch_robs)

    # Données pour l'échantillon sélectionné
    wavegrid = batch_wavegrid[sample_idx].detach().cpu().numpy()
    y_act_pred = batch_yact[sample_idx].detach().cpu().numpy()  # Activité prédite
    y_act_true = dataset.activity[sample_idx].detach().cpu().numpy()  # Activité vraie

    # Création du plot avec 4 subplots (1 en haut + 3 en bas)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Activity Comparison - {exp_name} - {phase_name} - Epoch {epoch}\n"
        f"Sample {sample_idx} | True vs Predicted Activity",
        fontsize=14,
        fontweight="bold",
    )

    # Plot 1 : Spectre d'activité complet (occupe les 3 colonnes de la première ligne)
    # Fusionner les 3 subplots de la première ligne
    gs = fig.add_gridspec(2, 3)
    ax_full = fig.add_subplot(gs[0, :])  # Première ligne complète

    # Supprimer les axes individuels de la première ligne
    for i in range(3):
        axes[0, i].remove()

    ax_full.plot(
        wavegrid, y_act_true, "k-", linewidth=2, alpha=0.8, label="True Activity"
    )
    ax_full.plot(
        wavegrid,
        y_act_pred,
        "orange",
        linewidth=2,
        alpha=0.8,
        label="Predicted Activity",
    )
    ax_full.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax_full.set_xlabel("Wavelength (Å)")
    ax_full.set_ylabel("Activity Flux")
    ax_full.set_title("Full Spectrum Activity Comparison")
    ax_full.legend()
    ax_full.grid(True, alpha=0.3)

    # Plots 2-4 : Zoom sur les 3 raies les plus importantes
    for i, line_pos in enumerate(selected_lines):
        ax = axes[1, i]

        # Créer le masque de zoom
        zoom_mask = (wavegrid >= line_pos - halfwin) & (wavegrid <= line_pos + halfwin)
        wave_zoom = wavegrid[zoom_mask]
        true_zoom = y_act_true[zoom_mask]
        pred_zoom = y_act_pred[zoom_mask]

        ax.plot(
            wave_zoom, true_zoom, "k-", linewidth=2, alpha=0.8, label="True Activity"
        )
        ax.plot(
            wave_zoom,
            pred_zoom,
            "orange",
            linewidth=2,
            alpha=0.8,
            label="Predicted Activity",
        )
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.axvline(
            x=line_pos,
            color="red",
            linestyle=":",
            alpha=0.7,
            label=f"Line @ {line_pos:.2f}Å",
        )

        ax.set_xlabel("Wavelength (Å)")
        ax.set_ylabel("Activity Flux")
        ax.set_title(
            f"Line {i + 1}: {line_pos:.2f}Å (Weight: {line_weights[top_indices[i]]:.3f})"
        )
        ax.set_xlim(line_pos - halfwin, line_pos + halfwin)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Nom de fichier simplifié
    filename = f"activity_comparison_epoch_{epoch}.png"
    filepath = os.path.join(typed_plot_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    # Nettoyage mémoire
    del batch_yobs, batch_yaug, batch_voffset, batch_wavegrid
    del batch_robs, batch_yact, batch_s
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==== ANALYSIS PLOTS ====


def plot_periodogram_analysis(
    periods,
    power,
    metrics,
    P_inj=None,
    fap_threshold=0.01,
    exclude_width_frac=0.05,
    peak_prominence=None,
    title="Lomb–Scargle Periodogram",
    save_path=None,
    show_plot=False,
):
    """Plot periodogram with FAP threshold, exclusion band, and metrics."""
    from scipy.signal import find_peaks

    max_power = np.max(power)
    fap_level = max_power * fap_threshold

    if P_inj is not None:
        mask_excl = np.abs(periods - P_inj) <= exclude_width_frac * P_inj
    else:
        mask_excl = np.zeros_like(periods, dtype=bool)

    if peak_prominence is None:
        peak_prominence = 0.5 * np.std(power)

    p_out = power[~mask_excl]
    per_out = periods[~mask_excl]
    peaks_out, _ = find_peaks(p_out, prominence=peak_prominence)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.semilogx(periods, power, lw=1.6)
    ax.set_xlabel("Période [jours]")
    ax.set_ylabel("Puissance Lomb–Scargle")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)

    ax.axhline(
        fap_level, ls="--", lw=1.2, label=f"Seuil FAP = {int(fap_threshold * 100)}%"
    )

    if P_inj is not None and P_inj > 0:
        ax.axvspan(
            P_inj * (1 - exclude_width_frac),
            P_inj * (1 + exclude_width_frac),
            alpha=0.15,
            label=f"Bande autour de $P_{{inj}}$ = {P_inj:.4g} j",
        )

        if metrics.get("P_detected") is not None:
            P_detected = metrics["P_detected"]
            idx = np.argmin(np.abs(periods - P_detected))
            ax.plot(
                periods[idx],
                power[idx],
                marker="o",
                ms=7,
                mec="k",
                mfc="none",
                label=f"Pic détecté @ {P_detected:.4g} j",
            )
            ax.axvline(P_inj, color="k", lw=1.0, alpha=0.5)

    if peaks_out.size:
        ax.plot(
            per_out[peaks_out], p_out[peaks_out], "x", ms=6, label="Pics (hors planète)"
        )

    lines = []
    if metrics.get("fap_at_Pinj") is not None:
        lines.append(f"FAP @ P_inj: {metrics['fap_at_Pinj']:.3g}")
    if metrics.get("power_ratio") is not None:
        lines.append(f"Power ratio: {metrics['power_ratio']:.3g}")
    if metrics.get("n_sig_peaks_outside") is not None:
        lines.append(f"# pics FAP<1% (hors): {metrics['n_sig_peaks_outside']}")
    if metrics.get("delta_P") is not None:
        lines.append(f"ΔP: {metrics['delta_P']:.3g} j")
    if lines:
        txt = "\n".join(lines)
        ax.text(
            0.02,
            0.98,
            txt,
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7", alpha=0.9),
            fontsize=10,
        )

    ax.legend(loc="best", frameon=True)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.close()
    elif show_plot:
        plt.show()
    else:
        plt.close()


def plot_mcmc_posteriors(samples, truths=None, save_path=None):
    """Plot MCMC posterior distributions."""
    labels = ["P [d]", "K [m/s]", "phi [rad]"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for i, (ax, lab) in enumerate(zip(axes, labels)):
        ax.hist(samples[:, i], bins=50, color="#4C72B0", alpha=0.7, density=True)
        ax.set_xlabel(lab)
        if truths is not None:
            v = truths[["P", "K", "phi"][i]]
            ax.axvline(v, color="k", ls="--", lw=1)
        ax.grid(True, alpha=0.2)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()


def plot_yact_perturbed(all_yact_perturbed, wavegrid, save_path=None):
    """Plot perturbed activations for each latent dimension."""
    latent_dim, n_spectra, n_pixels = all_yact_perturbed.shape

    fig, axes = plt.subplots(latent_dim, 1, figsize=(12, 3 * latent_dim))
    if latent_dim == 1:
        axes = [axes]

    for dim in range(latent_dim):
        ax = axes[dim]

        n_plot = min(10, n_spectra)
        for i in range(n_plot):
            ax.plot(wavegrid, all_yact_perturbed[dim, i], alpha=0.3, lw=0.8)

        mean_perturbed = np.mean(all_yact_perturbed[dim], axis=0)
        ax.plot(
            wavegrid,
            mean_perturbed,
            "k-",
            lw=2,
            alpha=0.8,
            label=f"Mean (dim {dim + 1})",
        )

        ax.set_xlabel("Wavelength [Å]")
        ax.set_ylabel("Perturbed Activation")
        ax.set_title(f"Perturbed y_act for latent dimension {dim + 1}")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()


def plot_latent_analysis_for_series(all_s, y_series, label, out_root_dir, correlations):
    """Plot latent analysis (3D/pairwise + histogram) for a velocity series."""
    from matplotlib.gridspec import GridSpec

    y = np.asarray(y_series).reshape(-1)
    S = all_s.shape[1]
    os.makedirs(out_root_dir, exist_ok=True)

    lat_vel = correlations.get("latent_vs_velocity")
    act_vel = correlations.get("activity_vs_velocity")

    if lat_vel is not None and lat_vel.size:
        top_k = int(np.argmax(np.abs(lat_vel)))
        top_val = float(lat_vel[top_k])
        mean_abs = float(np.mean(np.abs(lat_vel)))
        lat_summary = f"max|corr(y,s_k)|=|{top_val:.2f}| (k={top_k + 1})\nmean|corr|={mean_abs:.2f}"
    else:
        lat_summary = ""

    act_summary = (
        f"corr(y,FWHM)={act_vel['fwhm']:.2f}\n"
        f"corr(y,depth)={act_vel['depth']:.2f}\n"
        f"corr(y,span)={act_vel['span']:.2f}"
    )
    annotation = lat_summary + ("\n" if lat_summary else "") + act_summary

    if S == 3:
        fig = plt.figure(figsize=(14, 6))
        gs = GridSpec(2, 3, figure=fig, width_ratios=[1.6, 1, 1], height_ratios=[1, 1])

        ax3d = fig.add_subplot(gs[:, 0], projection="3d")
        p = ax3d.scatter(
            all_s[:, 0], all_s[:, 1], all_s[:, 2], c=y, cmap="viridis", s=8, alpha=0.9
        )
        ax3d.set_xlabel("s1")
        ax3d.set_ylabel("s2")
        ax3d.set_zlabel("s3")
        fig.colorbar(p, ax=ax3d, shrink=0.7, label=f"{label} [m/s]")
        ax3d.set_title(f"Latent space (colored by {label})")

        ax12 = fig.add_subplot(gs[0, 1])
        ax13 = fig.add_subplot(gs[0, 2])
        ax23 = fig.add_subplot(gs[1, 1])
        axh = fig.add_subplot(gs[1, 2])

        ax12.scatter(all_s[:, 0], all_s[:, 1], c=y, cmap="viridis", s=6, alpha=0.8)
        ax12.set_xlabel("s1")
        ax12.set_ylabel("s2")
        ax12.grid(True, alpha=0.3)

        ax13.scatter(all_s[:, 0], all_s[:, 2], c=y, cmap="viridis", s=6, alpha=0.8)
        ax13.set_xlabel("s1")
        ax13.set_ylabel("s3")
        ax13.grid(True, alpha=0.3)

        ax23.scatter(all_s[:, 1], all_s[:, 2], c=y, cmap="viridis", s=6, alpha=0.8)
        ax23.set_xlabel("s2")
        ax23.set_ylabel("s3")
        ax23.grid(True, alpha=0.3)

        axh.hist(y, bins=40, color="#4C72B0", alpha=0.8, density=False)
        axh.set_xlabel(f"{label} [m/s]")
        axh.set_ylabel("N")
        axh.grid(True, alpha=0.3)
        axh.text(
            0.02,
            0.98,
            annotation,
            transform=axh.transAxes,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7", alpha=0.9),
            fontsize=9,
        )

        fig.tight_layout()
        out_path = os.path.join(out_root_dir, f"latent_analysis_{label}.png")
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

    else:
        # Pairwise plots
        pair_dir = os.path.join(out_root_dir, f"latent_pairs_{label}")
        os.makedirs(pair_dir, exist_ok=True)
        for i in range(S):
            for j in range(i + 1, S):
                fig, ax = plt.subplots(figsize=(5, 4))
                sc = ax.scatter(
                    all_s[:, i], all_s[:, j], c=y, cmap="viridis", s=6, alpha=0.8
                )
                ax.set_xlabel(f"s{i + 1}")
                ax.set_ylabel(f"s{j + 1}")
                ax.grid(True, alpha=0.3)
                cbar = plt.colorbar(sc, ax=ax)
                cbar.set_label(f"{label} [m/s]")
                fig.tight_layout()
                fig.savefig(
                    os.path.join(
                        pair_dir, f"latent_pair_s{i + 1}_s{j + 1}_{label}.png"
                    ),
                    dpi=200,
                )
                plt.close(fig)

        # Histogram
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(y, bins=40, color="#4C72B0", alpha=0.8)
        ax.set_xlabel(f"{label} [m/s]")
        ax.set_ylabel("N")
        ax.grid(True, alpha=0.3)
        ax.text(
            0.02,
            0.98,
            annotation,
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7", alpha=0.9),
            fontsize=9,
        )
        fig.tight_layout()
        fig.savefig(os.path.join(out_root_dir, f"hist_{label}.png"), dpi=200)
        plt.close(fig)

    print(f"📊 Activity comparison plot saved: {filename}")
