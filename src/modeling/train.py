#!/usr/bin/env python3
"""
Module d'entraînement professionnel pour AESTRA avec gestion de config YAML et checkpoints.

Usage principal:
    from src.modeling.train import main

    # Nouvelle expérience
    main(config_path="src/modeling/configs/base_config.yaml")

    # Reprendre depuis checkpoint
    main(checkpoint_path="experiments/exp1/models/model_joint_epoch_100.pth")

    # Reprendre avec nouvelle config
    main(
        config_path="src/modeling/configs/base_config.yaml",
        checkpoint_path="experiments/exp1/models/model_joint_epoch_100.pth"
    )

    # Reprendre depuis le dernier checkpoint d'une expérience
    main(exp_path="experiments/exp1")

Note: La config utilisée est TOUJOURS celle spécifiée en argument, pas celle du checkpoint.
      Cela permet de faire des modifications à la config même lors de la reprise d'un checkpoint.
"""

import os
import yaml
import torch
import csv
from datetime import datetime
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler as DeprecatedGradScaler  # compat
from rich.console import Console
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from src.modeling.models import AESTRA, save_checkpoint
from src.dataset import SpectrumDataset, generate_collate_fn
from src.utils import get_class, clear_gpu_memory, get_gpu_memory_info
from src.plots_aestra import (
    plot_losses,
    plot_aestra_analysis,
    plot_rv_predictions_dataset,
    plot_activity,
)

console = Console()


def setup_experiment_directories(config, config_path=None):
    """
    Crée la structure de dossiers pour une expérience d'entraînement.

    Args:
        config: Configuration de l'expérience
        config_path: Chemin du fichier de config (pour extraction automatique du nom)

    Returns:
        dict: Dictionnaire avec les chemins des dossiers créés
    """
    # Déterminer le nom de l'expérience
    experiment_name = config.get("experiment_name")

    if not experiment_name:
        # Extraire le nom depuis le dataset_filepath
        dataset_filepath = config.get("dataset_filepath", "")
        if dataset_filepath:
            # Extraire juste le nom de fichier sans extension
            dataset_filename = os.path.splitext(os.path.basename(dataset_filepath))[0]
            experiment_name = dataset_filename
        else:
            # Fallback : utiliser le nom du fichier de config si disponible
            if config_path:
                config_filename = os.path.splitext(os.path.basename(config_path))[0]
                experiment_name = f"experiment_{config_filename}"
            else:
                experiment_name = "experiment_default"

    output_root = config.get("output_root_dir", "experiments")

    # Dossier principal de l'expérience
    exp_dir = os.path.join(output_root, experiment_name)

    # Structure des sous-dossiers (nouvelle organisation)
    subdirs = {
        "experiment_dir": exp_dir,
        "models_dir": os.path.join(exp_dir, "models"),
        "figures_dir": os.path.join(exp_dir, "figures"),
        "logs_dir": os.path.join(exp_dir, "logs"),
    }

    # Créer tous les dossiers
    for dir_path in subdirs.values():
        os.makedirs(dir_path, exist_ok=True)

    # Sauvegarder la configuration sous le nom standard "config.yaml"
    config_save_path = os.path.join(exp_dir, "config.yaml")
    with open(config_save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    console.print(f"📁 Structure d'expérience créée dans: {exp_dir}")
    console.print(f"📋 Configuration sauvegardée: {config_save_path}")

    return subdirs, experiment_name


def save_experiment_checkpoint(
    model,
    optimizer,
    scheduler,
    dataset,
    config,
    exp_name,
    epoch,
    phase_name,
    scaler=None,
    path=None,
    exp_dirs=None,
):
    """
    Sauvegarde complète d'une expérience avec config et dataset.

    Args:
        model: Le modèle AESTRA
        optimizer: L'optimiseur
        scheduler: Le scheduler (peut être None)
        dataset: Le dataset utilisé
        config: La configuration complète
        exp_name: Nom de l'expérience (ex: "exp0")
        epoch: Numéro d'epoch actuel
        phase_name: Nom de la phase actuelle
        scaler: Le GradScaler pour mixed precision (peut être None)
        path: Chemin de sauvegarde (optionnel)
        exp_dirs: Dictionnaire des dossiers d'expérience
    """
    if path is None:
        if exp_dirs is not None:
            # Nouvelle convention: {model_name}_{phase}_epoch_{epoch}.pth
            filename = f"model_{phase_name}_epoch_{epoch}.pth"
            path = os.path.join(exp_dirs["models_dir"], filename)
        else:
            # Fallback vers l'ancien système
            path = f"models/model_{phase_name}_epoch_{epoch}.pth"

    # Sauvegarde du checkpoint standard
    if optimizer is not None:
        save_checkpoint(model, optimizer, path, scheduler)
    else:
        # Sauvegarde minimale pour modèle final (sans optimizer/scheduler)
        ckpt = {
            "model_state_dict": model.state_dict(),
            "model_phase": model.phase,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(ckpt, path)

    # Ajout des métadonnées de l'expérience
    ckpt = torch.load(path)
    ckpt.update(
        {
            "exp_name": exp_name,
            "epoch": epoch,
            "current_phase": phase_name,  # Phase actuelle
            "config": config,
            "dataset_metadata": dataset.to_dict(),
        }
    )

    # Sauvegarde de l'état du scaler si la mixed precision est activée
    if scaler is not None:
        ckpt["scaler_state_dict"] = scaler.state_dict()

    torch.save(ckpt, path)
    console.log(f"💾 Experiment checkpoint saved: {path}")


def load_experiment_checkpoint(path, device="cuda"):
    """
    Charge un checkpoint d'expérience complet.

    Returns:
        dict: Contient model, optimizer, scheduler, dataset, config, exp_name, epoch, scaler_state_dict
    """
    console.log(f"📂 Loading experiment checkpoint: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device)

    # Reconstruction du dataset
    dataset_metadata = ckpt["dataset_metadata"]
    # data_root_dir override supprimé: on utilise le chemin complet enregistré
    dataset = SpectrumDataset(**dataset_metadata)

    # Reconstruction du modèle
    config = ckpt["config"]
    model = AESTRA(
        n_pixels=dataset.n_pixels,
        S=config["latent_dim"],
        sigma_v=config["sigma_v"],
        sigma_c=config["sigma_c"],
        sigma_y=config["sigma_y"],
        k_reg_init=config["k_reg_init"],
        cycle_length=config["cycle_length"],
        b_obs=dataset.template,
        b_rest=dataset.spectra.mean(dim=0),
        device=device,
        dtype=getattr(torch, config.get("model_dtype", "float32")),
    )

    # Load state dict with compatibility handling
    model_state_dict = ckpt["model_state_dict"]

    # Get current model's expected keys
    current_model_keys = set(model.state_dict().keys())
    saved_model_keys = set(model_state_dict.keys())

    # Filter out unexpected keys (backward compatibility)
    unexpected_keys = saved_model_keys - current_model_keys
    if unexpected_keys:
        print(
            f"Warning: Filtering out unexpected keys from checkpoint: {unexpected_keys}"
        )
        filtered_state_dict = {
            k: v for k, v in model_state_dict.items() if k in current_model_keys
        }
    else:
        filtered_state_dict = model_state_dict

    # Load the filtered state dict
    model.load_state_dict(filtered_state_dict, strict=False)
    model.set_phase(ckpt.get("model_phase", "joint"))

    if torch.cuda.is_available():
        model = model.cuda()

    return {
        "model": model,
        "dataset": dataset,
        "config": config,
        "exp_name": ckpt.get(
            "exp_name", ckpt.get("cfg_name", "unknown")
        ),  # Compatibilité
        "epoch": ckpt["epoch"],
        "current_phase": ckpt.get("current_phase", "joint"),
        "checkpoint_data": ckpt,
        "scaler_state_dict": ckpt.get("scaler_state_dict", None),
    }


def load_config(config_path):
    """
    Charge un fichier de configuration depuis un chemin complet.

    Args:
        config_path: Chemin complet vers le fichier YAML

    Returns:
        dict: Configuration chargée
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    console.log(f"📋 Config loaded: {config_path}")
    return config


def save_losses_to_csv(losses_history, exp_name, phase_name, epoch, csv_dir, config):
    """
    Sauvegarde les losses dans un fichier CSV unique.

    Args:
        losses_history: Dict avec les listes de losses {'rv': [...], 'fid': [...], etc.}
        exp_name: Nom de l'expérience
        phase_name: Nom de la phase actuelle
        epoch: Epoch actuelle
        csv_dir: Répertoire de sauvegarde des CSV
        config: Configuration complète (pour les métadonnées)
    """
    if not config.get("save_losses_csv", False):
        return  # CSV désactivé

    os.makedirs(csv_dir, exist_ok=True)

    # Nom du fichier CSV unique
    csv_filename = "losses.csv"
    csv_path = os.path.join(csv_dir, csv_filename)

    # Vérifier si le fichier existe déjà pour savoir si on ajoute les headers
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "w" if not file_exists else "a", newline="") as csvfile:
        fieldnames = [
            "timestamp",
            "exp_name",
            "phase",
            "epoch",
            "rv_loss",
            "fid_loss",
            "c_loss",
            "reg_loss",
            "total_loss",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Headers seulement si nouveau fichier
        if not file_exists:
            writer.writeheader()

        # Écrire les données pour cette epoch seulement (la dernière)
        if losses_history["rv"]:  # S'assurer qu'il y a des données
            current_epoch = len(losses_history["rv"])
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            writer.writerow(
                {
                    "timestamp": timestamp,
                    "exp_name": exp_name,
                    "phase": phase_name,
                    "epoch": current_epoch,
                    "rv_loss": losses_history["rv"][-1],
                    "fid_loss": losses_history["fid"][-1],
                    "c_loss": losses_history["c"][-1],
                    "reg_loss": losses_history["reg"][-1],
                    "total_loss": losses_history["total"][-1],
                }
            )

    console.log(f"💾 Losses saved to CSV: {csv_filename}")


def create_optimizer_and_scheduler(model, phase_config):
    """Crée l'optimiseur et le scheduler depuis la config d'une phase."""
    # Création de l'optimiseur
    optimizer_class = get_class(phase_config["optimizer"])
    optimizer = optimizer_class(model.parameters(), **phase_config["optimizer_kwargs"])

    # Création du scheduler (optionnel)
    scheduler = None
    if "scheduler" in phase_config:
        scheduler_class = get_class(phase_config["scheduler"])
        scheduler = scheduler_class(optimizer, **phase_config["scheduler_kwargs"])

    return optimizer, scheduler


def create_grad_scaler(config):
    """Crée le GradScaler pour la mixed precision selon la configuration."""
    if not config.get("use_mixed_precision", False) or not config.get(
        "grad_scaler_enabled", False
    ):
        return None

    if not torch.cuda.is_available():
        console.log("⚠️  Mixed precision désactivée : CUDA non disponible")
        return None

    # Utiliser l'API moderne torch.amp si dispo, sinon fallback
    try:
        scaler = torch.amp.GradScaler(
            "cuda",
            init_scale=config.get("grad_scaler_init_scale", 65536.0),
            growth_factor=config.get("grad_scaler_growth_factor", 2.0),
            backoff_factor=config.get("grad_scaler_backoff_factor", 0.5),
            growth_interval=config.get("grad_scaler_growth_interval", 2000),
            enabled=True,
        )
    except Exception:
        scaler = DeprecatedGradScaler(
            init_scale=config.get("grad_scaler_init_scale", 65536.0),
            growth_factor=config.get("grad_scaler_growth_factor", 2.0),
            backoff_factor=config.get("grad_scaler_backoff_factor", 0.5),
            growth_interval=config.get("grad_scaler_growth_interval", 2000),
            enabled=True,
        )

    console.log("🚀 Mixed precision activée avec GradScaler")
    return scaler


def find_latest_checkpoint(exp_path):
    """
    Trouve le dernier checkpoint dans un dossier d'expérience.

    Args:
        exp_path: Chemin vers le dossier d'expérience

    Returns:
        str or None: Chemin vers le dernier checkpoint ou None si aucun trouvé
    """
    models_dir = os.path.join(exp_path, "models")
    if not os.path.exists(models_dir):
        return None

    # Chercher tous les fichiers .pth dans models/
    checkpoint_files = []
    for file in os.listdir(models_dir):
        if file.endswith(".pth") and "epoch_" in file:
            try:
                # Extraire le numéro d'epoch du nom de fichier
                # Format attendu: model_{phase}_epoch_{epoch}.pth
                parts = file.split("_")
                epoch_part = None
                for i, part in enumerate(parts):
                    if part == "epoch" and i + 1 < len(parts):
                        epoch_str = parts[i + 1].split(".")[0]  # Enlever .pth
                        epoch_num = int(epoch_str)
                        epoch_part = epoch_num
                        break

                if epoch_part is not None:
                    checkpoint_files.append(
                        (os.path.join(models_dir, file), epoch_part)
                    )
            except (ValueError, IndexError):
                continue  # Ignorer les fichiers avec un format inattendu

    if not checkpoint_files:
        return None

    # Retourner le checkpoint avec le numéro d'epoch le plus élevé
    latest_checkpoint = max(checkpoint_files, key=lambda x: x[1])
    return latest_checkpoint[0]


class EarlyStopping:
    """
    Early Stopping pour arrêter l'entraînement quand la métrique surveillée ne s'améliore plus.
    """

    def __init__(
        self,
        patience=10,
        min_delta=0.0,
        metric="total",
        mode="min",
        restore_best_weights=True,
    ):
        """
        Args:
            patience: Nombre d'epochs sans amélioration avant d'arrêter
            min_delta: Amélioration minimale pour être considérée comme significative
            metric: Métrique à surveiller ("total", "rv", "fid", "c", "reg")
            mode: "min" pour minimiser, "max" pour maximiser
            restore_best_weights: Restaurer les meilleurs poids à l'arrêt
        """
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.mode = mode
        self.restore_best_weights = restore_best_weights

        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.best_epoch = 0
        self.counter = 0
        self.best_weights = None
        self.stopped_epoch = 0

    def __call__(self, current_value, epoch, model=None):
        """
        Vérifie si l'Early Stopping doit être déclenché.

        Args:
            current_value: Valeur actuelle de la métrique
            epoch: Epoch actuel
            model: Modèle (pour sauvegarder les meilleurs poids)

        Returns:
            bool: True si l'entraînement doit s'arrêter
        """
        if self._is_improvement(current_value):
            self.best_value = current_value
            self.best_epoch = epoch
            self.counter = 0

            # Sauvegarder les meilleurs poids
            if self.restore_best_weights and model is not None:
                self.best_weights = {
                    k: v.clone() for k, v in model.state_dict().items()
                }

        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.stopped_epoch = epoch
            return True

        return False

    def _is_improvement(self, current_value):
        """Vérifie si la valeur actuelle est une amélioration."""
        if self.mode == "min":
            return current_value < (self.best_value - self.min_delta)
        else:
            return current_value > (self.best_value + self.min_delta)

    def restore_weights(self, model):
        """Restaure les meilleurs poids dans le modèle."""
        if self.best_weights is not None and model is not None:
            model.load_state_dict(self.best_weights)
            console.log(f"🔄 Meilleurs poids restaurés (epoch {self.best_epoch})")


def create_early_stopping(phase_config):
    """Crée un objet Early Stopping depuis la config d'une phase."""
    if "early_stopping" not in phase_config:
        return None

    es_config = phase_config["early_stopping"]

    early_stopping = EarlyStopping(
        patience=int(es_config.get("patience", 10)),
        min_delta=float(es_config.get("min_delta", 0.0)),
        metric=es_config.get("metric", "total"),
        mode=es_config.get("mode", "min"),
        restore_best_weights=es_config.get("restore_best_weights", True),
    )

    console.log(
        f"⏹️  Early Stopping activé: patience={early_stopping.patience}, "
        f"metric={early_stopping.metric}, mode={early_stopping.mode}"
    )

    return early_stopping


def train_phase(
    model,
    dataset,
    dataloader,
    phase_config,
    config,
    exp_name,
    start_epoch=0,
    exp_dirs=None,
):
    """Entraîne le modèle pour une phase donnée avec support de la mixed precision."""
    phase_name = phase_config["name"]
    n_epochs = phase_config["n_epochs"]

    console.rule(f"[bold green]Phase: {phase_name}[/]")

    # Configuration de la trainabilité des paramètres
    model.set_b_spectra_trainable(
        phase_config.get("b_obs_trainable", True),
        phase_config.get("b_rest_trainable", True),
    )

    # Création optimiseur et scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(model, phase_config)

    # Création du GradScaler pour mixed precision
    scaler = create_grad_scaler(config)
    use_mixed_precision = (
        config.get("use_mixed_precision", False) and scaler is not None
    )
    autocast_enabled = config.get("autocast_enabled", True) and use_mixed_precision

    # Création de l'Early Stopping
    early_stopping = create_early_stopping(phase_config)

    if use_mixed_precision:
        console.log(f"🔧 Mixed precision activée pour la phase '{phase_name}'")
    else:
        console.log(f"🔧 Précision standard (float32) pour la phase '{phase_name}'")

    # Historique des losses pour plotting
    losses_history = {"rv": [], "fid": [], "c": [], "reg": [], "total": [], "lr": []}

    # Table pour les losses
    table = Table(expand=True)
    table.add_column("Epoch", justify="right")
    table.add_column("RV", justify="right")
    table.add_column("FID", justify="right")
    table.add_column("C", justify="right")
    table.add_column("Reg", justify="right")
    table.add_column("Total Loss", justify="right")

    model.set_phase(phase_name)
    model.train()

    # Préparation device & transferts CPU->GPU par batch
    model_device = next(model.parameters()).device
    move_batches_to_device = bool(config.get("move_batches_to_device", True))
    non_blocking_transfer = bool(config.get("non_blocking_transfer", True))

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        epoch_task = progress.add_task("Epochs", total=n_epochs)

        for epoch in range(start_epoch, n_epochs):
            epoch_losses = {"rv": 0.0, "fid": 0.0, "c": 0.0, "reg": 0.0}

            for it, batch in enumerate(dataloader):
                # Transfert CPU->GPU des batches si demandé
                if move_batches_to_device and (model_device.type == "cuda"):
                    try:
                        batch = tuple(
                            t.to(model_device, non_blocking=non_blocking_transfer)
                            if isinstance(t, torch.Tensor)
                            else t
                            for t in batch
                        )
                    except Exception as e:
                        console.log(f"⚠️  Batch to({model_device}) failed: {e}")

                B = batch[0].shape[0]

                # ⚠️ CRITIQUE: Reset gradients à chaque batch
                optimizer.zero_grad()

                # Forward pass avec ou sans autocast selon la configuration
                if autocast_enabled:
                    # Utilise l'API moderne si dispo
                    try:
                        amp_ctx = torch.amp.autocast("cuda")
                    except Exception:
                        from torch.cuda.amp import autocast as legacy_autocast

                        amp_ctx = legacy_autocast()

                    with amp_ctx:
                        losses = model.get_losses(
                            batch=batch,
                            extrapolate="linear",
                            iteration_count=it,
                        )
                        # Calculer la loss totale pour ce batch
                        total_batch_loss = sum(losses.values())
                else:
                    losses = model.get_losses(
                        batch=batch,
                        extrapolate="linear",
                        iteration_count=it,
                    )
                    # Calculer la loss totale pour ce batch
                    total_batch_loss = sum(losses.values())

                # Backward pass avec ou sans scaler
                if use_mixed_precision and scaler is not None:
                    # Mixed precision backward pass
                    scaler.scale(total_batch_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard backward pass
                    total_batch_loss.backward()
                    optimizer.step()

                # Accumulation des losses (avec detach pour éviter les gradients)
                with torch.no_grad():
                    for key in epoch_losses:
                        epoch_losses[key] += float(losses[key].detach()) * B

            # Moyenne des losses
            for key in epoch_losses:
                epoch_losses[key] /= len(dataloader.dataset)

            # Les losses sont maintenant des floats, pas des tensors
            total_loss = sum(epoch_losses.values())

            if scheduler is not None:
                scheduler.step()

            # Sauvegarde des losses dans l'historique
            losses_history["rv"].append(epoch_losses["rv"])
            losses_history["fid"].append(epoch_losses["fid"])
            losses_history["c"].append(epoch_losses["c"])
            losses_history["reg"].append(epoch_losses["reg"])
            losses_history["total"].append(total_loss)
            losses_history["lr"].append(float(optimizer.param_groups[0]["lr"]))

            # Ajout d'une ligne dans la table
            table.add_row(
                f"{epoch + 1}/{n_epochs}",
                f"{epoch_losses['rv']:.4e}",
                f"{epoch_losses['fid']:.4e}",
                f"{epoch_losses['c']:.4e}",
                f"{epoch_losses['reg']:.4e}",
                f"{total_loss:.4e}",
            )

            # Affichage
            console.clear()
            console.print(table)

            # Vérification Early Stopping
            if early_stopping is not None:
                # Obtenir la métrique à surveiller
                metric_value = epoch_losses.get(early_stopping.metric, total_loss)

                if early_stopping(metric_value, epoch + 1, model):
                    console.log(f"⏹️  Early Stopping déclenché à l'epoch {epoch + 1}")
                    console.log(
                        f"📈 Pas d'amélioration depuis {early_stopping.patience} epochs"
                    )
                    console.log(
                        f"🏆 Meilleure valeur: {early_stopping.best_value:.6e} à l'epoch {early_stopping.best_epoch}"
                    )

                    # Restaurer les meilleurs poids si configuré
                    if early_stopping.restore_best_weights:
                        early_stopping.restore_weights(model)

                    # Sortir de la boucle d'epochs
                    break

            # Plotting Losses périodique
            plot_every = config.get(
                "plot_every", 0
            )  # Par défaut pas de plots (0 = désactivé)
            if plot_every > 0 and (epoch + 1) % plot_every == 0:
                plot_dir = (
                    exp_dirs["figures_dir"]
                    if exp_dirs
                    else config.get("plot_dir", "reports/figures")
                )
                plot_losses(losses_history, phase_name, epoch + 1, plot_dir, console)

            # Plotting RV predictions périodique (dataset complet)
            plot_rv_every = phase_config.get(
                "plot_rv_every", config.get("plot_rv_every", 0)
            )
            if plot_rv_every > 0 and (epoch + 1) % plot_rv_every == 0:
                rv_plot_dir = (
                    exp_dirs["figures_dir"]
                    if exp_dirs
                    else config.get("plot_dir", "reports/figures")
                )
                try:
                    plot_rv_predictions_dataset(
                        dataset,
                        model,
                        phase_name,
                        epoch + 1,
                        rv_plot_dir,
                        batch_size=config.get("batch_size", 32),
                    )
                    console.log(
                        f"📈 RV predictions (full dataset) plotted at epoch {epoch + 1} (saved in {rv_plot_dir})"
                    )
                except Exception as e:
                    console.log(f"⚠️  RV plotting failed: {e}")

            # Plotting Activity comparaison périodique
            plot_activity_every = phase_config.get(
                "plot_activity_every", config.get("plot_activity_every", 0)
            )
            if plot_activity_every > 0 and (epoch + 1) % plot_activity_every == 0:
                activity_plot_dir = (
                    exp_dirs["figures_dir"]
                    if exp_dirs
                    else config.get("plot_dir", "reports/figures")
                )
                try:
                    plot_activity(
                        batch,
                        dataset,
                        model,
                        exp_name,
                        phase_name,
                        epoch + 1,
                        activity_plot_dir,
                        data_root_dir=config.get("data_root_dir", "data"),
                    )
                    console.log(
                        f"📊 Activity comparison plotted at epoch {epoch + 1} (saved in {activity_plot_dir})"
                    )
                except Exception as e:
                    console.log(f"⚠️  Activity plotting failed: {e}")

            # Plots de spectres périodiques
            plot_spectra_every = phase_config.get("plot_spectra_every", 0)
            if plot_spectra_every > 0 and (epoch + 1) % plot_spectra_every == 0:
                spectra_plot_dir = (
                    exp_dirs["spectra_dir"]
                    if exp_dirs
                    else phase_config.get("spectra_plot_dir", "reports/spectra")
                )
                plot_aestra_analysis(
                    batch,
                    dataset,
                    model,
                    exp_name,
                    phase_name,
                    epoch + 1,
                    spectra_plot_dir,
                    zoom_line=True,
                    data_root_dir=config.get("data_root_dir", "data"),
                )

            # Sauvegarde CSV périodique
            csv_save_every = config.get("csv_save_every", 0)  # Par défaut pas de CSV
            if csv_save_every > 0 and (epoch + 1) % csv_save_every == 0:
                csv_dir = (
                    exp_dirs["logs_dir"]
                    if exp_dirs
                    else config.get("csv_dir", "reports/logs")
                )
                save_losses_to_csv(
                    losses_history, exp_name, phase_name, epoch + 1, csv_dir, config
                )

            # Sauvegarde périodique (tous les 50 epochs)
            if (epoch + 1) % config.get("checkpoint_every", 50) == 0:
                save_experiment_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    dataset,
                    config,
                    exp_name,
                    epoch + 1,
                    phase_name,
                    scaler,
                    exp_dirs=exp_dirs,
                )
                # ⚠️ CRITIQUE: Nettoyage de la mémoire GPU après sauvegarde
                clear_gpu_memory()

            # Nettoyage périodique de la mémoire (tous les 10 epochs)
            if (epoch + 1) % 10 == 0:
                clear_gpu_memory()
                memory_info = get_gpu_memory_info()
                if "error" not in memory_info:
                    console.log(
                        f"🔍 GPU Memory: {memory_info['used_mb']:.1f}/{memory_info['total_mb']:.1f} MB ({memory_info['usage_percent']:.1f}%)"
                    )

            progress.advance(epoch_task)

    console.log(f"✅ Phase '{phase_name}' terminée.")

    # Sauvegarde finale de la phase (TOUJOURS, même si early stopping)
    final_epoch = min(len(losses_history["total"]), n_epochs)
    save_experiment_checkpoint(
        model,
        optimizer,
        scheduler,
        dataset,
        config,
        exp_name,
        final_epoch,
        phase_name,
        scaler,
        exp_dirs=exp_dirs,
    )
    console.log(
        f"💾 Final checkpoint saved for phase '{phase_name}' at epoch {final_epoch}"
    )

    # Plot final de la phase
    plot_every = config.get("plot_every", 0)
    if plot_every > 0:
        plot_dir = (
            exp_dirs["figures_dir"]
            if exp_dirs
            else config.get("plot_dir", "reports/figures")
        )
        plot_losses(losses_history, phase_name, n_epochs, plot_dir, console)

    # Sauvegarde CSV finale de la phase
    if config.get("save_losses_csv", False):
        csv_dir = (
            exp_dirs["logs_dir"] if exp_dirs else config.get("csv_dir", "reports/logs")
        )
        save_losses_to_csv(
            losses_history, exp_name, phase_name, n_epochs, csv_dir, config
        )


def main(
    config_path: str = None,
    checkpoint_path: str = None,
    exp_path: str = None,
    device: str = "cuda",
):
    """
    Fonction principale d'entraînement AESTRA.

    Args:
        config_path: Chemin vers le fichier de configuration YAML (optionnel)
        checkpoint_path: Chemin vers un checkpoint pour reprendre l'entraînement (optionnel)
        exp_path: Chemin vers un dossier d'expérience pour reprendre depuis le dernier checkpoint (optionnel)
        device: Device à utiliser ("cuda" ou "cpu")
    """

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # Déterminer le mode d'opération
    if exp_path and not checkpoint_path:
        # Cas 4: Reprendre depuis le dernier checkpoint d'une expérience
        checkpoint_path = find_latest_checkpoint(exp_path)
        if checkpoint_path:
            console.log(f"🔍 Latest checkpoint found: {checkpoint_path}")
        else:
            raise FileNotFoundError(f"No checkpoint found in {exp_path}")

    # Chargement de la configuration
    config = None
    config_source = None

    if checkpoint_path:
        # Cas 2 ou 3: Reprendre depuis checkpoint
        console.log(f"🔄 Loading checkpoint: {checkpoint_path}")
        exp_data = load_experiment_checkpoint(checkpoint_path, device)

        if config_path:
            # Cas 3: Reprendre avec nouvelle config
            config = load_config(config_path)
            config_source = f"config from {config_path} (overriding checkpoint config)"
        else:
            # Cas 2: Reprendre avec config du checkpoint
            exp_dir = os.path.dirname(
                os.path.dirname(checkpoint_path)
            )  # Remonter depuis models/ vers exp/
            exp_config_path = os.path.join(exp_dir, "config.yaml")
            if os.path.exists(exp_config_path):
                config = load_config(exp_config_path)
                config_source = f"config from experiment directory {exp_config_path}"
            else:
                config = exp_data["config"]
                config_source = "config from checkpoint (fallback)"

        start_epoch = exp_data["epoch"]
        current_phase = exp_data["current_phase"]
        exp_name = exp_data["exp_name"]

    else:
        # Cas 1: Nouvelle expérience
        if not config_path:
            raise ValueError("config_path is required for new experiments")

        config = load_config(config_path)
        config_source = f"config from {config_path}"
        start_epoch = 0
        current_phase = None
        exp_name = None  # Sera déterminé par setup_experiment_directories

    console.log(f"✅ Configuration loaded: {config_source}")

    # Configuration de la structure de dossiers pour l'expérience
    exp_dirs, determined_exp_name = setup_experiment_directories(config, config_path)

    # Utiliser le nom d'expérience déterminé si on n'en a pas encore
    if exp_name is None:
        exp_name = determined_exp_name

    console.rule(f"[bold blue]AESTRA Training - Experiment: {exp_name}[/]")

    # Création du dataset et du modèle à partir de la config actuelle (toujours)
    console.log("🔧 Début de la création du dataset...")
    console.log(f"📁 dataset_filepath: {config.get('dataset_filepath')}")

    # Création du dataset (NPZ standardisé uniquement)
    try:
        # Contrôle CPU/GPU pour le dataset via la config
        dataset_cuda = bool(config.get("dataset_cuda", False))
        dataset = SpectrumDataset(
            dataset_filepath=config.get(
                "dataset_filepath",
                "data/npz_datasets/dataset_1000specs_5000_5050_Kp1e-1_P100.npz",
            ),
            data_dtype=getattr(torch, config.get("data_dtype", "float32")),
            cuda=dataset_cuda,
        )
        console.log(
            f"✅ Dataset créé avec succès (device={'GPU' if dataset_cuda and torch.cuda.is_available() else 'CPU'})"
        )
    except Exception as e:
        console.log(f"❌ Erreur lors de la création du dataset: {e}")
        raise

    console.log("🤖 Début de la création du modèle...")

    # Création du modèle
    try:
        model = AESTRA(
            n_pixels=dataset.n_pixels,
            S=config["latent_dim"],
            sigma_v=config["sigma_v"],
            sigma_c=config["sigma_c"],
            sigma_y=config["sigma_y"],
            k_reg_init=config["k_reg_init"],
            cycle_length=config["cycle_length"],
            b_obs=dataset.template,
            b_rest=dataset.spectra.mean(dim=0),
            device=device,
            dtype=getattr(torch, config.get("model_dtype", "float32")),
        )
        console.log("✅ Modèle créé avec succès")
    except Exception as e:
        console.log(f"❌ Erreur lors de la création du modèle: {e}")
        raise

    # Si on charge depuis un checkpoint, charger les poids du modèle
    if checkpoint_path:
        console.log("🔄 Loading model weights from checkpoint...")
        checkpoint_model_state = exp_data["checkpoint_data"]["model_state_dict"]

        # Charger les poids avec gestion de compatibilité
        current_model_keys = set(model.state_dict().keys())
        saved_model_keys = set(checkpoint_model_state.keys())

        # Filtrer les clés inattendues (compatibilité backward)
        unexpected_keys = saved_model_keys - current_model_keys
        if unexpected_keys:
            console.log(
                f"⚠️  Filtering unexpected keys from checkpoint: {list(unexpected_keys)[:5]}..."
            )
            filtered_state_dict = {
                k: v
                for k, v in checkpoint_model_state.items()
                if k in current_model_keys
            }
        else:
            filtered_state_dict = checkpoint_model_state

        model.load_state_dict(filtered_state_dict, strict=False)
        console.log("✅ Model weights loaded from checkpoint")

    if torch.cuda.is_available():
        model = model.cuda()
        console.log("✅ Modèle déplacé vers GPU")
    else:
        console.log("💻 Modèle utilise le CPU")

    if checkpoint_path:
        console.log(f"🔄 Resuming from epoch {start_epoch}, phase '{current_phase}'")
    else:
        console.log("🆕 Starting new training")

    # Création du DataLoader
    collate_fn = generate_collate_fn(
        dataset,
        M=config["M_aug"],
        vmin=config["vmin"],
        vmax=config["vmax"],
        interpolate=config["interpolate"],
        extrapolate=config["extrapolate"],
        out_dtype=getattr(torch, config["out_dtype"]),
    )

    # Paramètres DataLoader contrôlés par la config
    num_workers = int(config.get("num_workers", 0))
    pin_memory = bool(config.get("pin_memory", torch.cuda.is_available()))
    prefetch_factor = int(config.get("prefetch_factor", 2)) if num_workers > 0 else None
    persistent_workers = bool(config.get("persistent_workers", num_workers > 0))

    dataloader_kwargs = dict(
        batch_size=config["batch_size"],
        shuffle=config["shuffle"],
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )
    if prefetch_factor is not None:
        dataloader_kwargs["prefetch_factor"] = prefetch_factor

    dataloader = DataLoader(dataset, **dataloader_kwargs)
    console.log(
        f"📦 DataLoader: batch_size={config['batch_size']}, workers={num_workers}, pin_memory={pin_memory}, prefetch_factor={prefetch_factor}"
    )

    console.log(f"📊 Dataset: {len(dataset)} spectres, {dataset.n_pixels} pixels")
    console.log(f"🔧 Modèle: {sum(p.numel() for p in model.parameters())} paramètres")
    console.log(f"Batch size: {config['batch_size']}")

    # Entraînement par phases
    if current_phase is not None:
        # Reprendre depuis une phase spécifique
        phase_found = False
        for phase_config in config["phases"]:
            if phase_config["name"] == current_phase:
                phase_found = True
                console.log(
                    f"🔄 Resuming phase '{current_phase}' from epoch {start_epoch}"
                )
                train_phase(
                    model,
                    dataset,
                    dataloader,
                    phase_config,
                    config,
                    exp_name,
                    start_epoch,
                    exp_dirs,
                )
                # Continuer avec les phases suivantes s'il y en a
                current_idx = config["phases"].index(phase_config)
                for next_phase_config in config["phases"][current_idx + 1 :]:
                    train_phase(
                        model,
                        dataset,
                        dataloader,
                        next_phase_config,
                        config,
                        exp_name,
                        0,
                        exp_dirs,
                    )
                break

        if not phase_found:
            console.log(
                f"⚠️  Phase '{current_phase}' not found in config, starting from beginning"
            )
            for phase_config in config["phases"]:
                train_phase(
                    model,
                    dataset,
                    dataloader,
                    phase_config,
                    config,
                    exp_name,
                    0,
                    exp_dirs,
                )
    else:
        # Nouvel entraînement - toutes les phases depuis le début
        for phase_config in config["phases"]:
            train_phase(
                model,
                dataset,
                dataloader,
                phase_config,
                config,
                exp_name,
                0,
                exp_dirs,
            )

    # Sauvegarde finale globale avec nom conventionnel pour predict.py
    final_model_path = os.path.join(exp_dirs["models_dir"], "aestra_final.pth")
    save_experiment_checkpoint(
        model,
        None,  # Pas d'optimizer pour le modèle final
        None,  # Pas de scheduler pour le modèle final
        dataset,
        config,
        exp_name,
        0,  # Epoch final
        "final",
        None,  # Pas de scaler pour le modèle final
        path=final_model_path,
        exp_dirs=exp_dirs,
    )
    console.log(f"💾 Final model saved: {final_model_path}")

    console.rule("[bold green]🎉 Entraînement terminé ![/]")
    console.log(f"Tous les checkpoints sont sauvés dans: {exp_dirs['models_dir']}")
    console.log("Le dernier modèle correspond au dernier checkpoint epoch sauvé")


if __name__ == "__main__":
    main(
        config_path="src/modeling/configs/base_config.yaml",
    )
