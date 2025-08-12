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
from scipy.signal import find_peaks


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


def plot_losses(losses_history, phase_name, epoch, plot_dir, console):
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
    if "rv_val" in losses_history:
        axes[0, 0].plot(epochs, losses_history["rv_val"], "b--", linewidth=2, label="RV Val Loss")
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
    if "fid_val" in losses_history:
        axes[0, 1].plot(
            epochs, losses_history["fid_val"], "r--", linewidth=2, label="Fidelity Val Loss"
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
    if "reg_val" in losses_history:
        axes[0, 2].plot(
            epochs, losses_history["reg_val"], "m--", linewidth=2, label="Regularization Val Loss"
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
    if "c_val" in losses_history:
        axes[1, 0].plot(
            epochs, losses_history["c_val"], "g--", linewidth=2, label="Consistency Val Loss"
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
    filename = f"rv_predictions_epoch_{epoch}.png"
    filepath = os.path.join(typed_plot_dir, filename)
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close()


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
    filename = f"activity_epoch_{epoch}.png"
    filepath = os.path.join(typed_plot_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    # Nettoyage mémoire
    del batch_yobs, batch_yaug, batch_voffset, batch_wavegrid
    del batch_robs, batch_yact, batch_s
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==== Plots de predict.py ====


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
    Visualisation de l'effet de la perturbation du vecteur latent pour TOUTES les dimensions fournies.

    Args:
        y_act_original: Spectre d'activité original (1D)
        y_act_perturbed_list: Liste des spectres d'activité perturbés (liste de 1D)
        wavelength: Grille de longueurs d'onde (1D)
        save_path: Chemin de sauvegarde (optionnel)
        show_plot: Afficher le plot (False par défaut)
        wave_range: Gamme de longueurs d'onde à afficher (tuple)
    """
    n_dims = len(y_act_perturbed_list)
    if n_dims == 0:
        return

    # Palette de couleurs
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, n_dims)))

    # Filtre pour la gamme de longueurs d'onde
    wave_mask = (wavelength >= wave_range[0]) & (wavelength <= wave_range[1])
    wave_filtered = wavelength[wave_mask]
    y_original_filtered = y_act_original[wave_mask]

    # Création de la figure avec subplots pour chaque perturbation
    fig, axes = plt.subplots(n_dims, 1, figsize=(10, max(6, 2 * n_dims)), sharex=True)
    if n_dims == 1:
        axes = [axes]

    fig.suptitle(
        "Visualization of the effect of perturbing the latent vector",
        fontsize=14,
        fontweight="bold",
    )

    for i, y_perturbed in enumerate(y_act_perturbed_list):
        ax = axes[i]
        color = colors[i % len(colors)]
        label = f"Perturb s_{i + 1}"

        y_perturbed_filtered = y_perturbed[wave_mask]

        # Plot du spectre original (courbe noire)
        ax.plot(wave_filtered, y_original_filtered, "k-", linewidth=1.2, alpha=0.8)

        # Plot du spectre perturbé (courbe colorée)
        ax.plot(
            wave_filtered,
            y_perturbed_filtered,
            color=color,
            linewidth=1.2,
            alpha=0.95,
            label=label,
        )

        # Configuration des axes
        ax.set_ylabel("y_act", fontsize=11)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

        # Ajustement des limites y pour bien voir les différences
        y_min = min(y_original_filtered.min(), y_perturbed_filtered.min())
        y_max = max(y_original_filtered.max(), y_perturbed_filtered.max())
        margin = (y_max - y_min) * 0.1 if y_max > y_min else 1e-3
        ax.set_ylim(y_min - margin, y_max + margin)

    # Configuration de l'axe x pour le dernier subplot
    axes[-1].set_xlabel("Restframe wavelength (Å)", fontsize=11)

    plt.tight_layout(rect=[0, 0.02, 1, 0.98])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure sauvegardée: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_latent_marginal_distributions(all_s, save_path=None, show_plot=False):
    """Plot histograms of each latent coordinate distribution."""
    n, d = all_s.shape
    n_cols = min(4, d)
    n_rows = int(np.ceil(d / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols + 1, 2.4 * n_rows))
    axes = np.array(axes).reshape(n_rows, n_cols)

    for i in range(d):
        r, c = divmod(i, n_cols)
        ax = axes[r, c]
        ax.hist(all_s[:, i], bins=40, color="steelblue", alpha=0.85, edgecolor="k")
        ax.set_title(f"s_{i + 1}", fontsize=10)
        ax.grid(True, alpha=0.25)

    # Hide unused axes
    for j in range(d, n_rows * n_cols):
        r, c = divmod(j, n_cols)
        axes[r, c].axis("off")

    fig.suptitle("Latent coordinates distributions", fontsize=14)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        if show_plot:
            plt.show()
        else:
            plt.close()


def plot_latent_space_3d(
    latent_s,
    rv_values,
    save_path=None,
    show_plot=False,
    decorrelated=False,
):
    """Latent space visualization.

    - If D == 3: draw 3D scatter + 2D projections + RV histogram (no regression fits).
    - If D > 3: skip 3D; draw only 2D projections (S1-S2, S1-S3, S2-S3) + RV histogram.
    - If D < 3: return False.
    """
    D = latent_s.shape[1]
    if D < 3:
        print(f"⚠️  L'espace latent n'est que {D}D, projections insuffisantes")
        return False

    if save_path is None:
        save_path = "reports/figures/latent_space.png"

    s1, s2, s3 = latent_s[:, 0], latent_s[:, 1], latent_s[:, 2]

    print(f"RV range: [{np.min(rv_values):.3f}, {np.max(rv_values):.3f}] m/s")
    print(f"Nombre de spectres: {len(rv_values)}")

    # Figure layout differs depending on dimensionality
    if D == 3:
        fig = plt.figure(figsize=(20, 15))
        # 3D scatter
        ax_3d = fig.add_subplot(2, 3, (1, 4), projection="3d")
        scatter_3d = ax_3d.scatter(
            s1, s2, s3, c=rv_values, cmap="viridis", s=20, alpha=0.7, edgecolors="none"
        )
        ax_3d.set_xlabel("S₁")
        ax_3d.set_ylabel("S₂")
        ax_3d.set_zlabel("S₃")
        ax_3d.set_title(
            f"Espace latent 3D coloré par V_encode [m/s] {'DECORRELATED' if decorrelated else ''}"
        )
        cbar_3d = plt.colorbar(scatter_3d, ax=ax_3d, shrink=0.6)
        cbar_3d.set_label("V_encode [m/s]")

        # 2D projections
        ax_12 = fig.add_subplot(2, 3, 2)
        ax_12.scatter(
            s1, s2, c=rv_values, cmap="viridis", s=15, alpha=0.7, edgecolors="none"
        )
        ax_12.set_xlabel("S₁")
        ax_12.set_ylabel("S₂")
        ax_12.set_title("Projection S₁-S₂")
        ax_12.grid(True, alpha=0.3)

        ax_13 = fig.add_subplot(2, 3, 3)
        ax_13.scatter(
            s1, s3, c=rv_values, cmap="viridis", s=15, alpha=0.7, edgecolors="none"
        )
        ax_13.set_xlabel("S₁")
        ax_13.set_ylabel("S₃")
        ax_13.set_title("Projection S₁-S₃")
        ax_13.grid(True, alpha=0.3)

        ax_23 = fig.add_subplot(2, 3, 6)
        ax_23.scatter(
            s2, s3, c=rv_values, cmap="viridis", s=15, alpha=0.7, edgecolors="none"
        )
        ax_23.set_xlabel("S₂")
        ax_23.set_ylabel("S₃")
        ax_23.set_title("Projection S₂-S₃")
        ax_23.grid(True, alpha=0.3)

        # RV histogram
        ax_hist = fig.add_subplot(2, 3, 5)
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

    else:  # D > 3
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        ax_12, ax_13, ax_23, ax_hist = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

        # 2D projections only
        ax_12.scatter(
            s1, s2, c=rv_values, cmap="viridis", s=15, alpha=0.7, edgecolors="none"
        )
        ax_12.set_xlabel("S₁")
        ax_12.set_ylabel("S₂")
        ax_12.set_title("Projection S₁-S₂")
        ax_12.grid(True, alpha=0.3)

        ax_13.scatter(
            s1, s3, c=rv_values, cmap="viridis", s=15, alpha=0.7, edgecolors="none"
        )
        ax_13.set_xlabel("S₁")
        ax_13.set_ylabel("S₃")
        ax_13.set_title("Projection S₁-S₃")
        ax_13.grid(True, alpha=0.3)

        ax_23.scatter(
            s2, s3, c=rv_values, cmap="viridis", s=15, alpha=0.7, edgecolors="none"
        )
        ax_23.set_xlabel("S₂")
        ax_23.set_ylabel("S₃")
        ax_23.set_title("Projection S₂-S₃")
        ax_23.grid(True, alpha=0.3)

        # RV histogram
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

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Visualisation de l'espace latent sauvegardée: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return True


def plot_correlation_matrix(
    v_apparent,
    v_correct,
    v_traditionnal,
    v_ref,
    depth,
    span,
    fwhm,
    latent_vectors,
    save_path=None,
    show_plot=False,
):
    """Plot correlation matrix between RV methods and activity indicators/latent dimensions."""

    # Prepare data
    n_latent = latent_vectors.shape[1]

    # X-axis variables (indicators + latent dims)
    x_vars = {
        "FWHM": fwhm,
        "Span": span,
        "Depth": depth,
    }
    for i in range(n_latent):
        x_vars[f"s_{i + 1}"] = latent_vectors[:, i]

    # Y-axis variables (RV methods)
    y_vars = {
        "v_apparent": v_apparent,
        "v_correct": v_correct,
        "v_traditionnal": v_traditionnal,
        "v_ref": v_ref,
    }

    # Compute correlations
    x_names = list(x_vars.keys())
    y_names = list(y_vars.keys())
    corr_matrix = np.zeros((len(y_names), len(x_names)))

    for i, y_name in enumerate(y_names):
        for j, x_name in enumerate(x_names):
            corr_matrix[i, j] = np.corrcoef(y_vars[y_name], x_vars[x_name])[0, 1]

    # Plot
    fig, ax = plt.subplots(figsize=(max(8, len(x_names) * 0.8), 6))
    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    # Labels
    ax.set_xticks(range(len(x_names)))
    ax.set_xticklabels(x_names, rotation=45, ha="right")
    ax.set_yticks(range(len(y_names)))
    ax.set_yticklabels(y_names)

    # Add correlation values
    for i in range(len(y_names)):
        for j in range(len(x_names)):
            ax.text(
                j,
                i,
                f"{corr_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=10,
            )

    # Colorbar and title
    plt.colorbar(im, ax=ax, label="Corrélation")
    ax.set_title("Matrice de corrélation : Méthodes RV vs Indicateurs d'activité")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        if show_plot:
            plt.show()
        else:
            plt.close()


def plot_periodogram(
    periods,
    power,
    metrics=None,
    P_inj=None,
    fap_threshold=0.01,
    exclude_width_frac=0.05,
    peak_prominence=None,
    title="Lomb–Scargle Periodogram",
    save_path=None,
    show_plot=False,
    xlim=None,
):
    """
    Plot periodogram with detection metrics and annotations.
    Supports multiple injected periods.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Base periodogram plot
    ax.semilogx(periods, power, lw=1.6, color="blue")
    ax.set_xlabel("Période [jours]")
    ax.set_ylabel("Puissance Lomb–Scargle")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)

    # Safe max/min power
    power_clean = power[np.isfinite(power)]
    if power_clean.size == 0:
        max_power = 1.0
    else:
        max_power = (
            float(np.max(power_clean)) if np.isfinite(np.max(power_clean)) else 1.0
        )
    if max_power <= 0:
        max_power = 1.0

    # FAP horizontal level
    fap_level = max_power * fap_threshold
    if np.isfinite(fap_level) and fap_level > 0:
        ax.axhline(
            fap_level,
            ls="--",
            lw=1.2,
            color="red",
            label=f"Seuil FAP = {int(fap_threshold * 100)}%",
        )

    # Handle multiple injected periods and draw bands + markers
    P_inj_list = []
    metrics_list = []
    if P_inj is not None:
        P_inj_list = P_inj if isinstance(P_inj, (list, tuple, np.ndarray)) else [P_inj]
        if metrics is not None:
            metrics_list = metrics if isinstance(metrics, list) else [metrics]
            if len(metrics_list) != len(P_inj_list):
                metrics_list = (
                    [metrics_list[0]] * len(P_inj_list)
                    if metrics_list
                    else [None] * len(P_inj_list)
                )
        else:
            metrics_list = [None] * len(P_inj_list)

        colors = plt.cm.Set1(np.linspace(0, 1, max(len(P_inj_list), 3)))

        for i, (P_planet, planet_metrics) in enumerate(zip(P_inj_list, metrics_list)):
            if P_planet is None or P_planet <= 0:
                continue
            color = colors[i % len(colors)]

            # Exclusion band around injected period
            ax.axvspan(
                P_planet * (1 - exclude_width_frac),
                P_planet * (1 + exclude_width_frac),
                alpha=0.15,
                color=color,
                label=f"Fenêtre planète {i + 1}: P={P_planet:.3f} j",
            )

            # Reference line at injected period
            ax.axvline(P_planet, color=color, lw=1.2, alpha=0.7)

            # Red circle marker for detected peak if available
            if planet_metrics and planet_metrics.get("P_detected") is not None:
                P_detected = planet_metrics["P_detected"]
                idx = int(np.argmin(np.abs(periods - P_detected)))
                if idx >= 0 and idx < len(power) and np.isfinite(power[idx]):
                    ax.plot(
                        periods[idx],
                        power[idx],
                        marker="o",
                        ms=8,
                        mec="red",
                        mfc="none",
                        mew=2,
                        label=f"Pic détecté {i + 1}: {P_detected:.3f} j",
                    )

    # Mark significant peaks outside all planet windows (if any injected)
    combined_mask = np.zeros_like(periods, dtype=bool)
    if P_inj_list:
        for P_planet in P_inj_list:
            combined_mask |= np.abs(periods - P_planet) <= exclude_width_frac * P_planet

    if peak_prominence is None:
        peak_prominence = (
            0.5 * np.std(power[np.isfinite(power)])
            if np.any(np.isfinite(power))
            else 0.0
        )

    if np.any(~combined_mask):
        p_out = power[~combined_mask]
        per_out = periods[~combined_mask]
        if np.any(np.isfinite(p_out)):
            peaks_out, _ = find_peaks(p_out, prominence=peak_prominence)
            if peaks_out.size > 0:
                ax.plot(
                    per_out[peaks_out],
                    p_out[peaks_out],
                    "x",
                    ms=6,
                    color="orange",
                    label="Pics significatifs",
                )

    # Add metrics text box: one line per injected period
    if P_inj_list and metrics_list:
        lines = []
        for i, (P_planet, m) in enumerate(zip(P_inj_list, metrics_list)):
            if not m:
                continue
            parts = [f"P={P_planet:.3g}j"]
            if m.get("delta_P") is not None:
                parts.append(f"ΔP={m['delta_P']:.3g}j")
            if m.get("fap_at_Pinj") is not None:
                parts.append(f"FAP={m['fap_at_Pinj']:.2g}")
            if m.get("n_sig_peaks_outside") is not None:
                parts.append(f"Nsig={m['n_sig_peaks_outside']}")
            if m.get("power_ratio") is not None:
                parts.append(f"Ratio={m['power_ratio']:.2g}")
            lines.append("  ".join(parts))
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

    if xlim is not None:
        ax.set_xlim(*xlim)

    ax.legend(loc="best", frameon=True)
    plt.tight_layout()

    # Save or show
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        if show_plot:
            plt.show()
        else:
            plt.close()
