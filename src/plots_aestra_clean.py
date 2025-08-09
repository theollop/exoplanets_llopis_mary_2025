"""
Fonctions de plotting pour l'entraînement AESTRA.

Ce module contient les fonctions de visualisation optimisées pour AESTRA :
- Plotting des losses d'entraînement
- Visualisation des spectres selon le papier AESTRA
- Zoom ultra-précis pour l'analyse Doppler
- Plot 3D de l'espace latent
- Périodogramme des vitesses radiales
- Analyse des activations perturbées
- Distribution des distances latentes
- Analyse MCMC
"""

import os
import gc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from typing import Optional
from src.interpolate import shift_spectra_linear
from src.dataset import SpectrumDataset

# Lazy imports for optional dependencies
try:
    from scipy.signal import find_peaks
    from astropy.timeseries import LombScargle
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False

try:
    import corner
    HAS_CORNER = True
except ImportError:
    HAS_CORNER = False


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


def save_plot(
    fig,
    filename: str,
    tight_layout: bool = True,
    dpi: int = 150,
    verbose: bool = True,
):
    """
    Sauvegarde une figure avec optimisations GPU.
    """
    if tight_layout:
        try:
            fig.tight_layout()
        except Exception:
            pass

    fig.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # Nettoyage GPU après chaque plot
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if verbose:
        print(f"✅ Plot saved: {filename}")


def plot_losses_phase(
    losses_train: dict,
    losses_val: dict,
    epoch_start: int,
    epoch_end: int,
    filename: str,
    title: str = "Training and Validation Losses",
):
    """
    Plot optimisé des losses d'entraînement pour une phase spécifique.

    Args:
        losses_train: Dictionnaire des losses d'entraînement
        losses_val: Dictionnaire des losses de validation
        epoch_start: Époque de début de la phase
        epoch_end: Époque de fin de la phase
        filename: Nom du fichier de sauvegarde
        title: Titre du graphique
    """
    # Configuration dynamique selon le nombre de losses
    loss_keys = set(losses_train.keys()) | set(losses_val.keys())
    n_losses = len(loss_keys)

    # Configuration adaptative de la grille
    if n_losses <= 2:
        fig, axes = plt.subplots(1, n_losses, figsize=(6 * n_losses, 5))
    elif n_losses <= 4:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    else:
        cols = 3
        rows = (n_losses + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))

    # Ensure axes is always a list
    if n_losses == 1:
        axes = [axes]
    elif n_losses > 1:
        axes = axes.flatten()

    epochs = range(epoch_start, epoch_end + 1)

    # Plot chaque loss
    for i, loss_key in enumerate(sorted(loss_keys)):
        ax = axes[i]

        if loss_key in losses_train:
            train_values = losses_train[loss_key][epoch_start : epoch_end + 1]
            ax.plot(epochs, train_values, label=f"Train {loss_key}", linewidth=2)

        if loss_key in losses_val:
            val_values = losses_val[loss_key][epoch_start : epoch_end + 1]
            ax.plot(
                epochs,
                val_values,
                label=f"Val {loss_key}",
                linestyle="--",
                linewidth=2,
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"{loss_key.title()} Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Cacher les axes supplémentaires si nécessaire
    if n_losses < len(axes):
        for j in range(n_losses, len(axes)):
            axes[j].set_visible(False)

    fig.suptitle(title, fontsize=16, y=0.98)

    save_plot(fig, filename, verbose=True)


def plot_rv_predictions_aestra(
    rv_true: np.ndarray,
    rv_pred: np.ndarray,
    time_values: np.ndarray,
    filename: str,
    title: str = "Radial Velocity Predictions",
    show_residuals: bool = True,
):
    """
    Plot optimisé des prédictions de vitesses radiales selon le style AESTRA.

    Args:
        rv_true: Vitesses radiales vraies
        rv_pred: Vitesses radiales prédites
        time_values: Valeurs temporelles
        filename: Nom du fichier de sauvegarde
        title: Titre du graphique
        show_residuals: Afficher les résidus
    """
    if show_residuals:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))

    # Plot principal
    ax1.plot(time_values, rv_true, "o-", label="True RV", markersize=3, alpha=0.8)
    ax1.plot(time_values, rv_pred, "s-", label="Predicted RV", markersize=3, alpha=0.8)
    ax1.set_ylabel("RV [m/s]")
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Calcul des métriques
    mse = np.mean((rv_true - rv_pred) ** 2)
    mae = np.mean(np.abs(rv_true - rv_pred))
    std_res = np.std(rv_true - rv_pred)

    metrics_text = f"MSE: {mse:.2f} | MAE: {mae:.2f} | σ_res: {std_res:.2f} m/s"
    ax1.text(
        0.02,
        0.98,
        metrics_text,
        transform=ax1.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        fontsize=10,
    )

    if show_residuals:
        # Plot des résidus
        residuals = rv_true - rv_pred
        ax2.plot(time_values, residuals, "o-", markersize=2, alpha=0.7, color="red")
        ax2.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        ax2.set_xlabel("Time [days]")
        ax2.set_ylabel("Residuals [m/s]")
        ax2.grid(True, alpha=0.3)

    save_plot(fig, filename, verbose=True)


def plot_spectra_aestra_style(
    spectra: np.ndarray,
    wavegrid: np.ndarray,
    indices: list,
    filename: str,
    title: str = "Stellar Spectra Analysis",
    zoom_range: tuple = None,
):
    """
    Plot optimisé des spectres selon le style du papier AESTRA.

    Args:
        spectra: Array des spectres [N, W]
        wavegrid: Grille de longueurs d'onde [W]
        indices: Indices des spectres à afficher
        filename: Nom du fichier de sauvegarde
        title: Titre du graphique
        zoom_range: Tuple (lambda_min, lambda_max) pour le zoom
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Palette de couleurs optimisée
    colors = plt.cm.Set1(np.linspace(0, 1, len(indices)))

    # Plot complet (panneau supérieur)
    ax1 = axes[0]
    for i, idx in enumerate(indices):
        ax1.plot(
            wavegrid,
            spectra[idx],
            color=colors[i],
            alpha=0.8,
            linewidth=1.5,
            label=f"Spectrum {idx}",
        )

    ax1.set_xlabel("Wavelength [Å]")
    ax1.set_ylabel("Normalized Flux")
    ax1.set_title(f"{title} - Full Range")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot zoomé (panneau inférieur)
    ax2 = axes[1]
    if zoom_range:
        mask = (wavegrid >= zoom_range[0]) & (wavegrid <= zoom_range[1])
        wave_zoom = wavegrid[mask]

        for i, idx in enumerate(indices):
            flux_zoom = spectra[idx][mask]
            ax2.plot(
                wave_zoom,
                flux_zoom,
                color=colors[i],
                alpha=0.8,
                linewidth=2,
                label=f"Spectrum {idx}",
            )

        ax2.set_xlabel("Wavelength [Å]")
        ax2.set_ylabel("Normalized Flux")
        ax2.set_title(f"Zoomed View: {zoom_range[0]:.1f} - {zoom_range[1]:.1f} Å")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(
            0.5,
            0.5,
            "No zoom range specified",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=12,
        )

    save_plot(fig, filename, verbose=True)


def plot_latent_space_3d(
    latent_vectors: np.ndarray,
    rv_values: np.ndarray,
    filename: str,
    title: str = "3D Latent Space Visualization",
):
    """
    Visualisation 3D de l'espace latent avec coloration par vitesse radiale.

    Args:
        latent_vectors: Vecteurs latents [N, D] (utilise les 3 premières dimensions)
        rv_values: Vitesses radiales pour la coloration [N]
        filename: Nom du fichier de sauvegarde
        title: Titre du graphique
    """
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Utiliser les 3 premières dimensions
    s1 = latent_vectors[:, 0] if latent_vectors.shape[1] > 0 else np.zeros(len(rv_values))
    s2 = latent_vectors[:, 1] if latent_vectors.shape[1] > 1 else np.zeros(len(rv_values))
    s3 = latent_vectors[:, 2] if latent_vectors.shape[1] > 2 else np.zeros(len(rv_values))

    # Scatter plot avec coloration
    scatter = ax.scatter(s1, s2, s3, c=rv_values, cmap="viridis", s=20, alpha=0.6)

    ax.set_xlabel("Latent Dimension 1")
    ax.set_ylabel("Latent Dimension 2")
    ax.set_zlabel("Latent Dimension 3")
    ax.set_title(title)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
    cbar.set_label("Radial Velocity [m/s]")

    save_plot(fig, filename, verbose=True)


def plot_periodogram_analysis(periods, power, metrics, P_inj=None, title="Lomb-Scargle Periodogram", save_path=None):
    """
    Plot complet d'analyse de périodogramme avec métriques de détection.

    Args:
        periods: Array des périodes
        power: Array des puissances du périodogramme
        metrics: Dictionnaire des métriques de détection
        P_inj: Période injectée (optionnel)
        title: Titre du graphique
        save_path: Chemin pour sauvegarder la figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot principal du périodogramme
    ax.loglog(periods, power, 'b-', alpha=0.8, linewidth=1)
    ax.set_xlabel('Period [days]')
    ax.set_ylabel('Power')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Marquer la période injectée
    if P_inj is not None and P_inj > 0:
        ax.axvline(P_inj, color='red', linestyle='--', linewidth=2, 
                  label=f'Injected P = {P_inj:.1f} d')

    # Marquer la période détectée si disponible
    P_detected = metrics.get('P_detected')
    if P_detected is not None:
        ax.axvline(P_detected, color='orange', linestyle=':', linewidth=2,
                  label=f'Detected P = {P_detected:.1f} d')

    # Ajouter les métriques comme texte
    text_lines = []
    if metrics.get('fap_at_Pinj') is not None:
        text_lines.append(f"FAP at P_inj: {metrics['fap_at_Pinj']:.2e}")
    if metrics.get('power_ratio') is not None:
        text_lines.append(f"Power ratio: {metrics['power_ratio']:.2f}")
    if metrics.get('n_sig_peaks_outside') is not None:
        text_lines.append(f"Sig. peaks outside: {metrics['n_sig_peaks_outside']}")
    if metrics.get('delta_P') is not None:
        text_lines.append(f"ΔP: {metrics['delta_P']:.2f} d")

    if text_lines:
        text_str = '\n'.join(text_lines)
        ax.text(0.02, 0.98, text_str, transform=ax.transAxes, 
               verticalalignment='top', horizontalalignment='left',
               bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8),
               fontsize=10)

    ax.legend()

    if save_path:
        save_plot(fig, save_path, verbose=True)
    else:
        plt.show()


def plot_mcmc_posteriors(samples, truths=None, save_path=None, labels=None):
    """
    Plot des distributions postérieures MCMC.

    Args:
        samples: Array des échantillons MCMC [N_samples, N_params]
        truths: Valeurs vraies des paramètres (optionnel)
        save_path: Chemin pour sauvegarder la figure
        labels: Labels des paramètres
    """
    if not HAS_CORNER:
        print("Corner package not available. Skipping MCMC posterior plot.")
        return

    if labels is None:
        labels = ['P [days]', 'K [m/s]', 'φ [rad]']

    fig = corner.corner(
        samples,
        labels=labels,
        truths=truths,
        truth_color='red',
        show_titles=True,
        title_kwargs={"fontsize": 12},
        label_kwargs={"fontsize": 12}
    )

    if save_path:
        save_plot(fig, save_path, verbose=True)
    else:
        plt.show()


def plot_latent_distance_distribution(delta_s_rand, delta_s_aug, save_path=None, show_plot=False):
    """
    Plot de la distribution des distances latentes pour pairs aléatoires vs augmentées.

    Args:
        delta_s_rand: Distances latentes pour les paires aléatoires
        delta_s_aug: Distances latentes pour les paires augmentées
        save_path: Chemin pour sauvegarder la figure (optionnel)
        show_plot: Afficher le plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calcul des statistiques pour les légendes
    mean_rand = np.mean(delta_s_rand)
    mean_aug = np.mean(delta_s_aug)

    # Détermination automatique de la plage des valeurs
    all_values = np.concatenate([delta_s_rand, delta_s_aug])
    min_val = np.min(all_values[all_values > 0])  # Éviter les valeurs nulles
    max_val = np.max(all_values)

    # Extension de la plage pour une meilleure visualisation
    x_min = min_val * 0.5
    x_max = max_val * 2.0

    # Création des histogrammes avec bins adaptés
    bins = np.logspace(np.log10(x_min), np.log10(x_max), 50)

    # Histogrammes
    ax.hist(delta_s_rand, bins=bins, alpha=0.7, density=True,
           label=f'Random pairs (μ={mean_rand:.3f})', color='blue')
    ax.hist(delta_s_aug, bins=bins, alpha=0.7, density=True,
           label=f'Augmented pairs (μ={mean_aug:.3f})', color='red')

    # Lignes verticales pour les moyennes
    ax.axvline(mean_rand, color='blue', linestyle='--', linewidth=2, alpha=0.8)
    ax.axvline(mean_aug, color='red', linestyle='--', linewidth=2, alpha=0.8)

    # Configuration de l'échelle logarithmique
    ax.set_xscale('log')
    ax.set_xlabel('Latent Distance ||s_i - s_j||')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Latent Distances')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Ajout d'informations statistiques
    info_text = f'Ratio μ_rand/μ_aug: {mean_rand/mean_aug:.2f}'
    ax.text(0.7, 0.8, info_text, transform=ax.transAxes,
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
           fontsize=10)

    if save_path:
        save_plot(fig, save_path, verbose=True)
    
    if show_plot:
        plt.show()


def plot_yact_perturbed(all_yact_perturbed, wavegrid, save_path=None, show_plot=False):
    """
    Plot des activations perturbées pour chaque dimension latente.

    Args:
        all_yact_perturbed: Array des activations perturbées [latent_dim, N_samples, W]
        wavegrid: Grille de longueurs d'onde [W]
        save_path: Chemin pour sauvegarder la figure
        show_plot: Afficher le plot
    """
    latent_dim = all_yact_perturbed.shape[0]
    
    # Configuration adaptative de la grille
    if latent_dim <= 2:
        nrows, ncols = 1, latent_dim
    elif latent_dim <= 4:
        nrows, ncols = 2, 2
    elif latent_dim <= 6:
        nrows, ncols = 2, 3
    else:
        ncols = 3
        nrows = (latent_dim + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    
    if latent_dim == 1:
        axes = [axes]
    elif nrows == 1:
        axes = axes
    else:
        axes = axes.flatten()

    # Couleurs pour différents échantillons
    colors = plt.cm.Set1(np.linspace(0, 1, min(10, all_yact_perturbed.shape[1])))

    for dim in range(latent_dim):
        ax = axes[dim]
        
        # Plot quelques échantillons représentatifs
        n_samples_to_plot = min(5, all_yact_perturbed.shape[1])
        sample_indices = np.linspace(0, all_yact_perturbed.shape[1]-1, n_samples_to_plot, dtype=int)
        
        for i, sample_idx in enumerate(sample_indices):
            ax.plot(wavegrid, all_yact_perturbed[dim, sample_idx], 
                   color=colors[i], alpha=0.7, linewidth=1,
                   label=f'Sample {sample_idx}' if dim == 0 else '')
        
        ax.set_xlabel('Wavelength [Å]')
        ax.set_ylabel('Perturbed Activation')
        ax.set_title(f'Dimension {dim+1} Perturbation')
        ax.grid(True, alpha=0.3)
        
        if dim == 0:  # Légende seulement sur le premier subplot
            ax.legend(fontsize=8)

    # Cacher les axes supplémentaires si nécessaire
    if latent_dim < len(axes):
        for j in range(latent_dim, len(axes)):
            axes[j].set_visible(False)

    fig.suptitle('Latent Space Perturbation Analysis', fontsize=16)

    if save_path:
        save_plot(fig, save_path, verbose=True)
    
    if show_plot:
        plt.show()


def plot_latent_analysis_for_series(all_s, y_series, label, out_root_dir, correlations):
    """
    Analyse complète de l'espace latent pour une série de VR donnée.

    Args:
        all_s: Vecteurs latents [N, S]
        y_series: Série de vitesses radiales [N]
        label: Label de la série (ex: 'v_correct', 'v_apparent')
        out_root_dir: Répertoire de sortie de base
        correlations: Dictionnaire des corrélations calculées
    """
    y = np.asarray(y_series).reshape(-1)
    S = all_s.shape[1]
    
    # Créer le répertoire de sortie
    os.makedirs(out_root_dir, exist_ok=True)
    
    # Statistiques pour annotation
    latent_vs_velocity = correlations.get('latent_vs_velocity', [])
    activity_vs_velocity = correlations.get('activity_vs_velocity', {})
    
    annotation_lines = [f"Series: {label}"]
    if latent_vs_velocity:
        max_corr_idx = np.argmax(np.abs(latent_vs_velocity))
        max_corr_val = latent_vs_velocity[max_corr_idx]
        annotation_lines.append(f"Max |corr|: s{max_corr_idx+1} = {max_corr_val:.3f}")
    
    for act_name, corr_val in activity_vs_velocity.items():
        if corr_val is not None:
            annotation_lines.append(f"Corr y-{act_name}: {corr_val:.3f}")
    
    annotation = "\n".join(annotation_lines)

    if S <= 5:  # Summary plot pour dimensions faibles
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(3, S, height_ratios=[2, 2, 1], hspace=0.3, wspace=0.3)

        # Rangée 1: Scatter plots latent vs y
        for k in range(S):
            ax = fig.add_subplot(gs[0, k])
            ax.scatter(all_s[:, k], y, alpha=0.6, s=8, c='blue')
            ax.set_xlabel(f's{k+1}')
            ax.set_ylabel(f'{label} [m/s]' if k == 0 else '')
            ax.grid(True, alpha=0.3)
            if latent_vs_velocity and k < len(latent_vs_velocity):
                corr_val = latent_vs_velocity[k]
                ax.set_title(f'r = {corr_val:.3f}')

        # Rangée 2: Histogrammes des dimensions latentes
        for k in range(S):
            ax = fig.add_subplot(gs[1, k])
            ax.hist(all_s[:, k], bins=30, alpha=0.7, color='green')
            ax.set_xlabel(f's{k+1}')
            ax.set_ylabel('Count' if k == 0 else '')
            ax.grid(True, alpha=0.3)

        # Rangée 3: Histogramme de la série RV (étendu sur toutes les colonnes)
        axh = fig.add_subplot(gs[2, :])
        axh.hist(y, bins=40, alpha=0.7, color='orange')
        axh.set_xlabel(f'{label} [m/s]')
        axh.set_ylabel('Count')
        axh.grid(True, alpha=0.3)
        
        # Annotation
        axh.text(0.02, 0.98, annotation, transform=axh.transAxes, va='top', ha='left',
                bbox=dict(boxstyle="round,pad=0.4", fc='white', ec='0.7', alpha=0.9),
                fontsize=9)

        fig.tight_layout()
        out_path = os.path.join(out_root_dir, f'latent_analysis_{label}.png')
        save_plot(fig, out_path, verbose=True)
        
    else:  # Pairwise plots pour dimensions élevées
        pair_dir = os.path.join(out_root_dir, f'latent_pairs_{label}')
        os.makedirs(pair_dir, exist_ok=True)
        
        for i in range(S):
            for j in range(i+1, S):
                fig, ax = plt.subplots(figsize=(5, 4))
                sc = ax.scatter(all_s[:, i], all_s[:, j], c=y, cmap='viridis', s=6, alpha=0.8)
                ax.set_xlabel(f's{i+1}')
                ax.set_ylabel(f's{j+1}')
                ax.grid(True, alpha=0.3)
                cbar = plt.colorbar(sc, ax=ax)
                cbar.set_label(f'{label} [m/s]')
                fig.tight_layout()
                save_plot(fig, os.path.join(pair_dir, f'latent_pair_s{i+1}_s{j+1}_{label}.png'), verbose=True)

        # Histogramme de la série
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(y, bins=40, color='#4C72B0', alpha=0.8)
        ax.set_xlabel(f'{label} [m/s]')
        ax.set_ylabel('N')
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.98, annotation, transform=ax.transAxes, va='top', ha='left',
               bbox=dict(boxstyle="round,pad=0.4", fc='white', ec='0.7', alpha=0.9),
               fontsize=9)
        fig.tight_layout()
        save_plot(fig, os.path.join(out_root_dir, f'hist_{label}.png'), verbose=True)
