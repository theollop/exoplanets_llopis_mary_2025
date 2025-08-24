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
import numpy as np
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
    plot_latent_distance_distribution,
    plot_latent_space_3d,
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


def load_experiment_checkpoint(path, device="cuda", dataset_filepath=None):
    """
    Charge un checkpoint d'expérience complet.

    Returns:
        dict: Contient model, optimizer, scheduler, dataset, config, exp_name, epoch, scaler_state_dict
    """
    console.log(f"📂 Loading experiment checkpoint: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Try loading normally first. Some checkpoints may contain tensors
    # backed by external storages (e.g. UntypedStorage tagged with an NPZ file)
    # which can raise a RuntimeError like:
    # "don't know how to restore data location of torch.storage.UntypedStorage (...)"
    # In that case, fall back to a storage-preserving map_location which lets
    # the unpickler return storages as-is.
    try:
        ckpt = torch.load(path, map_location=device)
    except RuntimeError as e:
        msg = str(e)
        if (
            "don't know how to restore data location of torch.storage.UntypedStorage"
            in msg
            or "UntypedStorage" in msg
            or "Tagged" in msg
        ):
            console.log(
                "⚠️ Fallback: torch.load failed to restore tagged storage locations, retrying with storage-preserving map_location"
            )
            # This map_location returns the storage object unchanged. It's the
            # recommended fallback when loading checkpoints that reference
            # external storage files (e.g. custom npz-backed storages).
            ckpt = torch.load(path, map_location=lambda storage, loc: storage)
        else:
            raise

    # Reconstruction du dataset
    dataset_metadata = ckpt["dataset_metadata"]
    # If the caller provided an explicit dataset_filepath, prefer it and
    # override the saved path in the checkpoint metadata. Previously the
    # code added a `data_root_dir` key which doesn't match the
    # SpectrumDataset API and caused a TypeError.
    if dataset_filepath is not None:
        dataset_metadata["dataset_filepath"] = dataset_filepath

    # Construct the SpectrumDataset from the (possibly overridden)
    # metadata saved in the checkpoint. SpectrumDataset expects a
    # `dataset_filepath` argument.
    dataset = SpectrumDataset(**dataset_metadata)

    # Reconstruction du modèle
    config = ckpt["config"]
    model = AESTRA(
        n_pixels=dataset.n_pixels,
        b_obs=dataset.spectra.mean(dim=0),
        b_rest=dataset.spectra.mean(dim=0),
        b_rest_equal_b_obs=config.get("b_rest_equal_b_obs", False),
        b_rest_true=dataset.template if config.get("loss_b_rest", False) else None,
        loss_activity=config.get("loss_activity", False),
        S=config.get("latent_dim", 3),
        sigma_v=config.get("sigma_v", 1.0),
        sigma_s=config.get("sigma_s", 1.0),
        sigma_y=config.get("sigma_y", 1.0),
        k_reg_init=config.get("k_reg_init", 1.0),
        cycle_length=config.get("cycle_length", 1000),
        dropout=config.get("dropout", 0.0),
        device=device,
        dtype=getattr(torch, config.get("model_dtype", "float32")),
        smooth_alpha=config.get("smooth_alpha", 0.0),
        smooth_order=config.get("smooth_order", 1),
        sigma_l=config.get("sigma_l", 0.0),
        sigma_corr=config.get("sigma_corr", 0.0),
        include_activity_proxies=config.get("include_activity_proxies", False),
        activity_proxies_dim=config.get("activity_proxies_dim", 0),
        proxies_proj_dim=config.get("proxies_proj_dim", 32),
        conditioning_mode=config.get("conditioning_mode", "concat"),
        alpha_act=config.get("alpha_act", 1.0),
        beta_brest=config.get("beta_brest", 1.0),
        consistency_mode=config.get("consistency_mode", "mse"),
        encode_in_rest_frame=config.get("encode_in_rest_frame", True),
        interp_method=config.get("interpolate", "linear"),
        loss_fid_enabled=config.get("loss_fid_enabled", True),
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
            "smooth_loss",
            "corr_loss",
            "template_loss",
            "activity_loss",
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
                    "smooth_loss": losses_history.get("smooth", [0])[-1],
                    "corr_loss": losses_history.get("corr", [0])[-1],
                    "template_loss": losses_history.get("template", [0])[-1],
                    "activity_loss": losses_history.get("activity", [0])[-1],
                    "total_loss": losses_history["total"][-1],
                }
            )

    console.log(f"💾 Losses saved to CSV: {csv_filename}")


def create_optimizer_and_scheduler(model, phase_config):
    """Crée l'optimiseur et le scheduler depuis la config d'une phase."""
    # Création de l'optimiseur
    optimizer_class = get_class(phase_config["optimizer"])
    # N'inclure que les paramètres entraînables (requires_grad=True)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optimizer_class(trainable_params, **phase_config["optimizer_kwargs"])

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


def get_bobs_brest_init(
    b_obs: str, b_rest: str, dataset: SpectrumDataset, device="cuda", dtype="float32"
):
    """
    Récupère et initialise b_obs et b_rest depuis la configuration.

    Args:
        b_obs: "mean" / "random" / "true_template"
        b_rest: "mean" / "random" / "true_template"

        mean : Moyenne du dataset
        random : Échantillonnage aléatoire
        true_template : Template réel du dataset

    Returns:
        tuple: (b_obs, b_rest) comme tensors
    """

    if dtype == "float32":
        dtype = torch.float32
    elif dtype == "float64":
        dtype = torch.float64
    elif dtype == "float16":
        dtype = torch.float16

    if b_obs == "mean":
        b_obs_tensor = dataset.spectra.mean(axis=0)
    elif b_obs == "random":
        b_obs_tensor = torch.randn_like(dataset.spectra[0], device=device, dtype=dtype)
    elif b_obs == "true_template":
        b_obs_tensor = dataset.template
        if dataset.template is None:
            print("Template is not available in the dataset - Fallback to mean")
            b_obs_tensor = dataset.spectra.mean(
                axis=0
            )  # Fallback to mean if no template

    elif "index" in b_obs:
        index = int(b_obs.split("_")[-1])
        b_obs_tensor = dataset.spectra[index]
    elif b_obs == "zero":
        b_obs_tensor = torch.zeros_like(dataset.spectra[0], device=device, dtype=dtype)
    else:
        raise ValueError(f"Unknown b_obs type: {b_obs}")

    if b_rest == "mean":
        b_rest_tensor = dataset.spectra.mean(axis=0)
    elif b_rest == "random":
        b_rest_tensor = torch.randn_like(dataset.spectra[0], device=device, dtype=dtype)
    elif b_rest == "true_template":
        b_rest_tensor = dataset.template
        if dataset.template is None:
            print("Template is not available in the dataset - Fallback to mean")
            b_rest_tensor = dataset.spectra.mean(
                axis=0
            )  # Fallback to mean if no template
    elif "index" in b_rest:
        index = int(b_rest.split("_")[-1])
        b_rest_tensor = dataset.spectra[index]

    elif b_rest == "zero":
        b_rest_tensor = torch.zeros_like(dataset.spectra[0], device=device, dtype=dtype)
    else:
        raise ValueError(f"Unknown b_rest type: {b_rest}")

    return b_obs_tensor, b_rest_tensor


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

    console.rule(f"[bold green]🚀 PHASE: {phase_name.upper()}[/]", style="bold green")
    
    # Affichage détaillé de la configuration de la phase
    phase_info_table = Table(title=f"Configuration Phase '{phase_name}'", expand=True)
    phase_info_table.add_column("Paramètre", style="cyan", width=25)
    phase_info_table.add_column("Valeur", style="white")
    
    # Informations de base
    phase_info_table.add_row("📊 Nombre d'epochs", f"{n_epochs}")
    phase_info_table.add_row("🎯 Epoch de départ", f"{start_epoch}")
    phase_info_table.add_row("🔄 Phase du modèle", f"{phase_name}")
    
    # Configuration de l'optimiseur
    optimizer_info = phase_config.get("optimizer", "N/A")
    optimizer_kwargs = phase_config.get("optimizer_kwargs", {})
    lr = optimizer_kwargs.get("lr", "N/A")
    phase_info_table.add_row("⚙️ Optimiseur", f"{optimizer_info}")
    phase_info_table.add_row("📈 Learning Rate", f"{lr}")
    
    # Configuration du scheduler si présent
    if "scheduler" in phase_config:
        scheduler_info = phase_config.get("scheduler", "N/A")
        scheduler_kwargs = phase_config.get("scheduler_kwargs", {})
        phase_info_table.add_row("📉 Scheduler", f"{scheduler_info}")
        for key, value in scheduler_kwargs.items():
            phase_info_table.add_row(f"  └─ {key}", f"{value}")
    else:
        phase_info_table.add_row("📉 Scheduler", "Aucun")
    
    # Paramètres entraînables
    trainable_params = phase_config.get("trainable_params", {})
    phase_info_table.add_row("🔧 Paramètres entraînables", "")
    for param_name, is_trainable in trainable_params.items():
        status = "✅ Oui" if is_trainable else "❌ Non"
        phase_info_table.add_row(f"  └─ {param_name}", status)
    
    # Early Stopping si configuré
    if "early_stopping" in phase_config:
        es_config = phase_config["early_stopping"]
        patience = es_config.get("patience", 10)
        metric = es_config.get("metric", "total")
        mode = es_config.get("mode", "min")
        phase_info_table.add_row("⏹️ Early Stopping", f"Patience: {patience}, Metric: {metric}, Mode: {mode}")
    else:
        phase_info_table.add_row("⏹️ Early Stopping", "Désactivé")
    
    # Configuration des plots périodiques
    plot_rv_every = phase_config.get("plot_rv_every", config.get("plot_rv_every", 0))
    plot_activity_every = phase_config.get("plot_activity_every", config.get("plot_activity_every", 0))
    plot_spectra_every = phase_config.get("plot_spectra_every", 0)
    
    if plot_rv_every > 0:
        phase_info_table.add_row("📈 Plot RV (epochs)", f"{plot_rv_every}")
    if plot_activity_every > 0:
        phase_info_table.add_row("📊 Plot Activity (epochs)", f"{plot_activity_every}")
    if plot_spectra_every > 0:
        phase_info_table.add_row("🌈 Plot Spectra (epochs)", f"{plot_spectra_every}")
    
    # Configuration mixed precision
    use_mixed_precision = config.get("use_mixed_precision", False) and config.get("grad_scaler_enabled", False)
    autocast_enabled = config.get("autocast_enabled", True) and use_mixed_precision
    phase_info_table.add_row("🚀 Mixed Precision", "✅ Activée" if use_mixed_precision else "❌ Désactivée")
    if use_mixed_precision:
        phase_info_table.add_row("🎯 Autocast", "✅ Activé" if autocast_enabled else "❌ Désactivé")
    
    console.print(phase_info_table)
    console.print()  # Ligne vide pour la lisibilité

    # Configuration de la trainabilité des paramètres
    model.set_trainable(**phase_config["trainable_params"])

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
    losses_history = {
        "rv": [],
        "fid": [],
        "c": [],
        "reg": [],
        "total": [],
        "lr": [],
        "corr": [],
    }

    # Configuration des colonnes du tableau (affichage compact de la dernière ligne uniquement)
    loss_table_columns = [
        "Epoch",
        "RV",
        "FID",
        "C",
        "Reg",
        "Smooth",
        "Template",
        "Activity",
        "Corr",
        "Total Loss",
    ]

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
            epoch_losses = {
                "rv": 0.0,
                "fid": 0.0,
                "c": 0.0,
                "reg": 0.0,
                "smooth": 0.0,
                "template": 0.0,
                "activity": 0.0,
                "corr": 0.0,
            }

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
                            get_aug_data=config.get("get_aug_data", True),
                        )
                        # Calculer la loss totale pour ce batch
                        total_batch_loss = sum(losses.values())
                else:
                    losses = model.get_losses(
                        batch=batch,
                        extrapolate="linear",
                        iteration_count=it,
                        get_aug_data=config.get("get_aug_data", True),
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
                        epoch_losses[key] += (
                            float(losses.get(key, torch.tensor(0)).detach()) * B
                        )

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
            losses_history.setdefault("smooth", []).append(epoch_losses["smooth"])
            losses_history.setdefault("corr", []).append(epoch_losses["corr"])
            losses_history.setdefault("template", []).append(epoch_losses["template"])
            losses_history.setdefault("activity", []).append(epoch_losses["activity"])
            losses_history["total"].append(total_loss)
            losses_history["lr"].append(float(optimizer.param_groups[0]["lr"]))

            # Ajout d'une ligne dans la table (n'affiche que les losses > 0)
            def fmt(v):
                try:
                    vv = float(v)
                except Exception:
                    return ""
                return f"{vv:.4e}" if vv > 0.0 else ""

            # Affichage: ne montrer que la dernière ligne sous forme d'un mini-tableau
            row_table = Table(expand=True)
            for col in loss_table_columns:
                row_table.add_column(col)
            row_table.add_row(
                f"{epoch + 1}/{n_epochs}",
                fmt(epoch_losses["rv"]),
                fmt(epoch_losses["fid"]),
                fmt(epoch_losses["c"]),
                fmt(epoch_losses["reg"]),
                fmt(epoch_losses["smooth"]),
                fmt(epoch_losses["template"]),  # OK
                fmt(epoch_losses["activity"]),  # OK
                fmt(epoch_losses["corr"]),  # OK
                fmt(total_loss),
            )
            console.print(row_table)

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

            # Latent plots (distance distribution and 3D space) at intervals
            latent_dist_every = phase_config.get(
                "plot_latent_distance_every",
                config.get("plot_latent_distance_every", 0),
            )
            latent_space_every = phase_config.get(
                "plot_latent_space_every",
                config.get("plot_latent_space_every", 0),
            )

            need_latent_eval = (
                latent_dist_every > 0 and (epoch + 1) % latent_dist_every == 0
            ) or (latent_space_every > 0 and (epoch + 1) % latent_space_every == 0)

            if need_latent_eval:
                # Build an eval dataloader (no shuffle) and compute latents + RVs once
                eval_dl = DataLoader(
                    dataset=dataset,
                    batch_size=config.get("batch_size", 32),
                    shuffle=False,
                    collate_fn=generate_collate_fn(dataset=dataset),
                    num_workers=0,
                )
                was_training = model.training
                model.eval()
                all_s_list, all_saug_list, all_vobs_list = [], [], []
                device = next(model.parameters()).device
                with torch.no_grad():
                    for _batch in eval_dl:
                        (
                            yobs,
                            yaug,
                            voffset_true,
                            wavegrid,
                            weights_fid,
                            indices,
                            yact_true,
                            activity_proxies_norm,
                            batch_yact_noised,
                        ) = _batch

                        # Move to model device for safety
                        def _to_dev(t):
                            return (
                                t.to(device, non_blocking=True)
                                if isinstance(t, torch.Tensor)
                                else t
                            )

                        yobs = _to_dev(yobs)
                        yaug = _to_dev(yaug)
                        wavegrid = _to_dev(wavegrid)
                        activity_proxies_norm = _to_dev(activity_proxies_norm)

                        # RV predictions (obs and aug)
                        vobs_pred, vaug_pred = model.get_rvestimator_pred(
                            batch_yobs=yobs, batch_yaug=yaug
                        )

                        # Spender forward to get latents (obs and aug)
                        (
                            yobs_prime,
                            yact,
                            yact_aug,
                            s,
                            saug,
                        ) = model.get_spender_pred(
                            batch_yobs=yobs,
                            batch_yaug=yaug,
                            batch_wavegrid=wavegrid,
                            batch_vobs_pred=vobs_pred,
                            batch_vaug_pred=vaug_pred,
                            get_aug_data=True,
                            batch_activity_proxies_norm=activity_proxies_norm,
                            include_activity_proxies=model.include_activity_proxies,
                        )

                        all_s_list.append(s.detach().cpu().numpy())
                        all_saug_list.append(saug.detach().cpu().numpy())
                        all_vobs_list.append(vobs_pred.detach().cpu().numpy())

                # Restore training state
                if was_training:
                    model.train()

                all_s = np.concatenate(all_s_list, axis=0)
                all_saug = np.concatenate(all_saug_list, axis=0)
                rv_values = np.concatenate(all_vobs_list, axis=0)

                # Plot latent distance distribution
                if latent_dist_every > 0 and (epoch + 1) % latent_dist_every == 0:
                    # Compute distances (random pairs vs augmented)
                    n = all_s.shape[0]
                    inds = np.array(
                        [np.random.choice(n, size=2, replace=False) for _ in range(n)]
                    )
                    delta_s_rand = np.linalg.norm(
                        all_s[inds[:, 0]] - all_s[inds[:, 1]], axis=1
                    )
                    delta_s_aug = np.linalg.norm(all_s - all_saug, axis=1)

                    fig_dir = (
                        exp_dirs["figures_dir"]
                        if exp_dirs
                        else config.get("plot_dir", "reports/figures")
                    )
                    out_dir = os.path.join(fig_dir, phase_name, "latent")
                    os.makedirs(out_dir, exist_ok=True)
                    save_path = os.path.join(
                        out_dir, f"latent_distance_epoch_{epoch + 1}.png"
                    )
                    try:
                        plot_latent_distance_distribution(
                            delta_s_rand=delta_s_rand,
                            delta_s_aug=delta_s_aug,
                            save_path=save_path,
                            show_plot=False,
                        )
                        console.log(
                            f"📈 Latent distance distribution plotted at epoch {epoch + 1} (saved in {out_dir})"
                        )
                    except Exception as e:
                        console.log(f"⚠️  Latent distance plotting failed: {e}")

                # Plot latent 3D space (or 2D projections when D>3)
                if latent_space_every > 0 and (epoch + 1) % latent_space_every == 0:
                    fig_dir = (
                        exp_dirs["figures_dir"]
                        if exp_dirs
                        else config.get("plot_dir", "reports/figures")
                    )
                    out_dir = os.path.join(fig_dir, phase_name, "latent")
                    os.makedirs(out_dir, exist_ok=True)
                    save_path = os.path.join(
                        out_dir, f"latent_space_epoch_{epoch + 1}.png"
                    )
                    try:
                        plot_latent_space_3d(
                            latent_s=all_s,
                            rv_values=rv_values,
                            save_path=save_path,
                            show_plot=False,
                        )
                        console.log(
                            f"🧭 Latent space plotted at epoch {epoch + 1} (saved in {out_dir})"
                        )
                    except Exception as e:
                        console.log(f"⚠️  Latent space plotting failed: {e}")

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

            # Nettoyage périodique de la mémoire (tous les 50 epochs)
            if (epoch + 1) % config.get("clear_memory_every", 50) == 0:
                clear_gpu_memory()
                memory_info = get_gpu_memory_info()
                if "error" not in memory_info:
                    console.log(
                        f"🔍 GPU Memory: {memory_info['used_mb']:.1f}/{memory_info['total_mb']:.1f} MB ({memory_info['usage_percent']:.1f}%)"
                    )

            progress.advance(epoch_task)

    # Résumé de fin de phase
    console.rule(f"[bold blue]📊 RÉSUMÉ PHASE '{phase_name.upper()}'[/]", style="bold blue")
    
    # Créer un tableau de résumé
    summary_table = Table(title=f"Résultats Phase '{phase_name}'", expand=True)
    summary_table.add_column("Métrique", style="cyan", width=20)
    summary_table.add_column("Valeur Finale", style="white")
    summary_table.add_column("Évolution", style="yellow")
    
    # Calculer les évolutions (première vs dernière epoch)
    if len(losses_history["total"]) > 1:
        total_evolution = ((losses_history["total"][-1] - losses_history["total"][0]) / losses_history["total"][0]) * 100
        rv_evolution = ((losses_history["rv"][-1] - losses_history["rv"][0]) / losses_history["rv"][0]) * 100 if losses_history["rv"][0] > 0 else 0
        fid_evolution = ((losses_history["fid"][-1] - losses_history["fid"][0]) / losses_history["fid"][0]) * 100 if losses_history["fid"][0] > 0 else 0
    else:
        total_evolution = rv_evolution = fid_evolution = 0
    
    # Formater les évolutions
    def format_evolution(evo):
        if abs(evo) < 0.01:
            return "→ Stable"
        elif evo < 0:
            return f"↓ {abs(evo):.1f}% (amélioration)"
        else:
            return f"↑ {evo:.1f}% (dégradation)"
    
    # Ajouter les métriques finales
    final_epoch = len(losses_history["total"])
    summary_table.add_row("🎯 Epochs complétées", f"{final_epoch}/{n_epochs}", "")
    summary_table.add_row("📊 Loss totale", f"{losses_history['total'][-1]:.4e}", format_evolution(total_evolution))
    summary_table.add_row("🎯 Loss RV", f"{losses_history['rv'][-1]:.4e}", format_evolution(rv_evolution))
    summary_table.add_row("🔍 Loss FID", f"{losses_history['fid'][-1]:.4e}", format_evolution(fid_evolution))
    summary_table.add_row("⚙️ Loss C", f"{losses_history['c'][-1]:.4e}", "")
    summary_table.add_row("📏 Loss Reg", f"{losses_history['reg'][-1]:.4e}", "")
    
    # Ajouter les losses optionnelles si elles existent et sont > 0
    if losses_history.get("smooth") and losses_history["smooth"][-1] > 0:
        summary_table.add_row("🌊 Loss Smooth", f"{losses_history['smooth'][-1]:.4e}", "")
    if losses_history.get("template") and losses_history["template"][-1] > 0:
        summary_table.add_row("📋 Loss Template", f"{losses_history['template'][-1]:.4e}", "")
    if losses_history.get("activity") and losses_history["activity"][-1] > 0:
        summary_table.add_row("⭐ Loss Activity", f"{losses_history['activity'][-1]:.4e}", "")
    if losses_history.get("corr") and losses_history["corr"][-1] > 0:
        summary_table.add_row("🔗 Loss Corr", f"{losses_history['corr'][-1]:.4e}", "")
    
    # Learning rate final
    if losses_history["lr"]:
        summary_table.add_row("📈 Learning Rate", f"{losses_history['lr'][-1]:.2e}", "")
    
    # Early stopping si activé
    if early_stopping is not None and early_stopping.stopped_epoch > 0:
        summary_table.add_row("⏹️ Early Stopping", f"Arrêté à l'epoch {early_stopping.stopped_epoch}", f"Meilleur: epoch {early_stopping.best_epoch}")
    
    console.print(summary_table)
    console.print()  # Ligne vide
    
    console.log(f"✅ Phase '{phase_name}' terminée avec succès !")
    
    if early_stopping is not None and early_stopping.stopped_epoch > 0:
        console.log(f"⏹️ Arrêt précoce déclenché - Meilleure performance à l'epoch {early_stopping.best_epoch}")
    else:
        console.log(f"🎯 Phase complète - {final_epoch} epochs exécutées")
        
    console.print()  # Ligne vide avant la sauvegarde

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
    dataset_filepath: str = None,
    output_root_dir: str = None,
    experiment_name: str = None,
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

    # Apply overrides from function parameters: prefer explicit function args over config file
    if dataset_filepath is not None:
        config["dataset_filepath"] = dataset_filepath
    if output_root_dir is not None:
        config["output_root_dir"] = output_root_dir
    if experiment_name is not None:
        config["experiment_name"] = experiment_name

    # Configuration de la structure de dossiers pour l'expérience
    exp_dirs, determined_exp_name = setup_experiment_directories(config, config_path)

    # Utiliser le nom d'expérience déterminé si on n'en a pas encore
    if exp_name is None:
        exp_name = determined_exp_name

    console.rule(f"[bold blue]🚀 AESTRA TRAINING - EXPERIMENT: {exp_name.upper()}[/]", style="bold blue")
    
    # Affichage de la configuration générale
    general_info_table = Table(title="Configuration Générale de l'Expérience", expand=True)
    general_info_table.add_column("Paramètre", style="cyan", width=25)
    general_info_table.add_column("Valeur", style="white")
    
    # Informations sur l'expérience
    general_info_table.add_row("🏷️ Nom d'expérience", f"{exp_name}")
    general_info_table.add_row("📁 Dossier de sortie", f"{exp_dirs['experiment_dir']}")
    general_info_table.add_row("📋 Source config", f"{config_source}")
    
    # Informations sur le dataset
    dataset_path = config.get('dataset_filepath', 'N/A')
    general_info_table.add_row("📊 Dataset", f"{os.path.basename(dataset_path)}")
    general_info_table.add_row("📁 Chemin dataset", f"{dataset_path}")
    
    # Configuration du modèle
    general_info_table.add_row("🧠 Modèle", "AESTRA")
    general_info_table.add_row("🎯 Dimension latente", f"{config.get('latent_dim', 'N/A')}")
    general_info_table.add_row("💾 Type de données", f"{config.get('model_dtype', 'float32')}")
    general_info_table.add_row("🖥️ Device", f"{device}")
    
    # Configuration d'entraînement
    general_info_table.add_row("🔄 Batch size", f"{config.get('batch_size', 'N/A')}")
    general_info_table.add_row("👥 Num workers", f"{config.get('num_workers', 0)}")
    general_info_table.add_row("📌 Pin memory", f"{'✅' if config.get('pin_memory', False) else '❌'}")
    
    # Phases d'entraînement
    phases_info = []
    total_epochs = 0
    for phase in config.get("phases", []):
        phase_name = phase.get("name", "Unknown")
        phase_epochs = phase.get("n_epochs", 0)
        total_epochs += phase_epochs
        phases_info.append(f"{phase_name} ({phase_epochs} epochs)")
    
    general_info_table.add_row("🎭 Phases", " → ".join(phases_info))
    general_info_table.add_row("⏱️ Total epochs", f"{total_epochs}")
    
    # Configuration de sauvegarde/plotting
    checkpoint_every = config.get("checkpoint_every", 50)
    plot_every = config.get("plot_every", 0)
    general_info_table.add_row("💾 Checkpoint every", f"{checkpoint_every} epochs")
    general_info_table.add_row("📈 Plot every", f"{plot_every} epochs" if plot_every > 0 else "Désactivé")
    
    # Reprise depuis checkpoint
    if checkpoint_path:
        general_info_table.add_row("� Reprise depuis", f"Epoch {start_epoch}, phase '{current_phase}'")
        general_info_table.add_row("📂 Checkpoint", f"{os.path.basename(checkpoint_path)}")
    else:
        general_info_table.add_row("🆕 Mode", "Nouvel entraînement")
    
    console.print(general_info_table)
    console.print()  # Ligne vide
    
    # Début de la création du dataset
    console.rule("[bold yellow]📊 CRÉATION DU DATASET[/]", style="bold yellow")
    console.log(f"📁 Chargement depuis: {config.get('dataset_filepath')}")
    console.log("🔧 Initialisation en cours...")

    # Création du dataset (NPZ standardisé uniquement)
    try:
        # Contrôle CPU/GPU pour le dataset via la config
        dataset_cuda = bool(config.get("dataset_cuda", False))
        console.log(f"📊 mask_weights_fid: {config.get('mask_weights_fid', False)}")
        dataset = SpectrumDataset(
            dataset_filepath=config.get(
                "dataset_filepath",
                "data/npz_datasets/dataset_1000specs_5000_5050_Kp1e-1_P100.npz",
            ),
            data_dtype=getattr(torch, config.get("data_dtype", "float32")),
            cuda=dataset_cuda,
            mask_weights_fid=config.get("mask_weights_fid", False),
        )
        console.log(
            f"✅ Dataset créé avec succès (device={'GPU' if dataset_cuda and torch.cuda.is_available() else 'CPU'})"
        )
        
        # Informations détaillées sur le dataset
        dataset_details_table = Table(title="Détails du Dataset", expand=True)
        dataset_details_table.add_column("Propriété", style="cyan", width=25)
        dataset_details_table.add_column("Valeur", style="white")
        
        dataset_details_table.add_row("📊 Nombre de spectres", f"{len(dataset)}")
        dataset_details_table.add_row("🌈 Nombre de pixels", f"{dataset.n_pixels}")
        dataset_details_table.add_row("💾 Type de données", f"{dataset.spectra.dtype}")
        dataset_details_table.add_row("🖥️ Device", f"{dataset.spectra.device}")
        dataset_details_table.add_row("📏 Forme des spectres", f"{dataset.spectra.shape}")
        
        # Métadonnées du dataset si disponibles
        if hasattr(dataset, 'metadata') and dataset.metadata:
            if dataset.metadata.get("activity_proxies_included", False):
                dataset_details_table.add_row("⭐ Activity proxies", "✅ Inclus")
                if hasattr(dataset, 'activity_proxies') and dataset.activity_proxies is not None:
                    dataset_details_table.add_row("📊 Proxies shape", f"{dataset.activity_proxies.shape}")
            else:
                dataset_details_table.add_row("⭐ Activity proxies", "❌ Non inclus")
            
            if dataset.template is not None:
                dataset_details_table.add_row("📋 Template", "✅ Disponible")
            else:
                dataset_details_table.add_row("📋 Template", "❌ Non disponible")
        
        console.print(dataset_details_table)
        console.print()  # Ligne vide
        
    except Exception as e:
        console.log(f"❌ Erreur lors de la création du dataset: {e}")
        raise

    console.rule("[bold yellow]🤖 CRÉATION DU MODÈLE[/]", style="bold yellow")

    # Création du modèle
    try:
        b_obs_init, b_rest_init = get_bobs_brest_init(
            b_obs=config.get("b_obs_init", "true_template"),
            b_rest=config.get("b_rest_init", "mean"),
            dataset=dataset,
            device=device,
            dtype=config.get("model_dtype", "float32"),
        )
        console.log(
            f"✅ Initialisation b_obs : {config.get('b_obs_init', 'true_template')} b_rest : {config.get('b_rest_init', 'mean')}"
        )

        if config.get("loss_b_rest", False):
            console.log("🔄 loss_brest is enabled, using dataset template for b_rest")
        if config.get("loss_activity", False):
            console.log("🔄 loss_activity is enabled")
        if config.get("b_rest_equal_b_obs", False):
            console.log("🔄 b_rest_equal_b_obs is enabled")
        if config.get("encode_in_rest_frame", True):
            console.log("🔄 encode_in_rest_frame is enabled")

        if config.get("interpolate", "linear"):
            console.log(f"🔄 interpolate method: {config.get('interpolate')}")

        if config.get("loss_fid_enabled", True):
            console.log("🔄 loss_fid_enabled is enabled")

        else:
            console.log("🔄 loss_fid_enabled is disabled")

        model = AESTRA(
            n_pixels=dataset.n_pixels,
            S=config["latent_dim"],
            sigma_v=config["sigma_v"],
            sigma_s=config["sigma_s"],
            sigma_y=config["sigma_y"],
            k_reg_init=config["k_reg_init"],
            cycle_length=config["cycle_length"],
            b_obs=b_obs_init,
            b_rest=b_rest_init,
            b_rest_true=dataset.template if config.get("loss_b_rest", False) else None,
            b_rest_equal_b_obs=config.get("b_rest_equal_b_obs", False),
            loss_activity=config.get("loss_activity", False),
            device=device,
            dtype=getattr(torch, config.get("model_dtype", "float32")),
            smooth_alpha=config.get("smooth_alpha", 0.0),
            smooth_order=config.get("smooth_order", 1),
            sigma_l=config.get("sigma_l", 1.0),
            sigma_corr=config.get("sigma_corr", 0.0),
            include_activity_proxies=config.get("include_activity_proxies", False),
            activity_proxies_dim=config.get("activity_proxies_dim", 0),
            proxies_proj_dim=config.get("proxies_proj_dim", 32),
            conditioning_mode=config.get("conditioning_mode", "concat"),
            alpha_act=config.get("alpha_act", 1.0),
            beta_brest=config.get("beta_brest", 1.0),
            consistency_mode=config.get("consistency_mode", "mse"),
            encode_in_rest_frame=config.get("encode_in_rest_frame", True),
            interp_method=config.get("interpolate", "linear"),
            loss_fid_enabled=config.get("loss_fid_enabled", True),
        )
        console.log(
            f"✅ Modèle créé avec succès (include_activity_proxies={model.include_activity_proxies})"
        )
        
        # Informations détaillées sur le modèle
        model_details_table = Table(title="Détails du Modèle AESTRA", expand=True)
        model_details_table.add_column("Paramètre", style="cyan", width=25)
        model_details_table.add_column("Valeur", style="white")
        
        # Paramètres principaux
        model_details_table.add_row("🎯 Dimension latente (S)", f"{config['latent_dim']}")
        model_details_table.add_row("🌈 Nombre de pixels", f"{dataset.n_pixels}")
        model_details_table.add_row("📊 Sigma V", f"{config['sigma_v']}")
        model_details_table.add_row("📊 Sigma S", f"{config['sigma_s']}")
        model_details_table.add_row("📊 Sigma Y", f"{config['sigma_y']}")
        model_details_table.add_row("⚖️ K reg init", f"{config['k_reg_init']}")
        model_details_table.add_row("🔄 Cycle length", f"{config['cycle_length']}")
        
        # Configuration b_obs et b_rest
        b_obs_type = config.get("b_obs_init", "true_template")
        b_rest_type = config.get("b_rest_init", "mean")
        model_details_table.add_row("🎯 B_obs init", f"{b_obs_type}")
        model_details_table.add_row("🎯 B_rest init", f"{b_rest_type}")
        model_details_table.add_row("🔗 B_rest = B_obs", f"{'✅' if config.get('b_rest_equal_b_obs', False) else '❌'}")
        
        # Losses activées
        model_details_table.add_row("📊 Loss B_rest", f"{'✅' if config.get('loss_b_rest', False) else '❌'}")
        model_details_table.add_row("⭐ Loss Activity", f"{'✅' if config.get('loss_activity', False) else '❌'}")
        model_details_table.add_row("🔍 Loss FID", f"{'✅' if config.get('loss_fid_enabled', True) else '❌'}")
        
        # Configuration avancée
        smooth_alpha = config.get("smooth_alpha", 0.0)
        sigma_l = config.get("sigma_l", 0.0)
        sigma_corr = config.get("sigma_corr", 0.0)
        if smooth_alpha > 0:
            model_details_table.add_row("🌊 Smooth alpha", f"{smooth_alpha}")
        if sigma_l > 0:
            model_details_table.add_row("📏 Sigma L", f"{sigma_l}")
        if sigma_corr > 0:
            model_details_table.add_row("🔗 Sigma corr", f"{sigma_corr}")
        
        # Activity proxies
        include_proxies = config.get("include_activity_proxies", False)
        model_details_table.add_row("⭐ Activity proxies", f"{'✅' if include_proxies else '❌'}")
        if include_proxies:
            proxies_dim = config.get("activity_proxies_dim", 0)
            proj_dim = config.get("proxies_proj_dim", 32)
            conditioning_mode = config.get("conditioning_mode", "concat")
            model_details_table.add_row("  └─ Proxies dim", f"{proxies_dim}")
            model_details_table.add_row("  └─ Proj dim", f"{proj_dim}")
            model_details_table.add_row("  └─ Conditioning", f"{conditioning_mode}")
        
        # Informations techniques
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_details_table.add_row("🔧 Paramètres totaux", f"{total_params:,}")
        model_details_table.add_row("🎯 Paramètres entraînables", f"{trainable_params:,}")
        model_details_table.add_row("💾 Type de données", f"{config.get('model_dtype', 'float32')}")
        model_details_table.add_row("🔄 Interpolation", f"{config.get('interpolate', 'linear')}")
        model_details_table.add_row("🎭 Encode in rest frame", f"{'✅' if config.get('encode_in_rest_frame', True) else '❌'}")
        
        console.print(model_details_table)
        console.print()  # Ligne vide
        
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
    
    # Informations sur le DataLoader et cohérence dataset/modèle
    console.rule("[bold yellow]🚀 PRÉPARATION ENTRAÎNEMENT[/]", style="bold yellow")
    
    dataloader_table = Table(title="Configuration DataLoader & Cohérence", expand=True)
    dataloader_table.add_column("Paramètre", style="cyan", width=25)
    dataloader_table.add_column("Valeur", style="white")
    
    # Configuration DataLoader
    dataloader_table.add_row("📦 Batch size", f"{config['batch_size']}")
    dataloader_table.add_row("👥 Num workers", f"{num_workers}")
    dataloader_table.add_row("📌 Pin memory", f"{'✅' if pin_memory else '❌'}")
    dataloader_table.add_row("🔄 Shuffle", f"{'✅' if config.get('shuffle', True) else '❌'}")
    if prefetch_factor is not None:
        dataloader_table.add_row("⚡ Prefetch factor", f"{prefetch_factor}")
    dataloader_table.add_row("💪 Persistent workers", f"{'✅' if persistent_workers else '❌'}")
    
    # Paramètres d'augmentation
    dataloader_table.add_row("🎯 M_aug", f"{config.get('M_aug', 'N/A')}")
    dataloader_table.add_row("🏃 V_min", f"{config.get('vmin', 'N/A')} km/s")
    dataloader_table.add_row("🏃 V_max", f"{config.get('vmax', 'N/A')} km/s")
    dataloader_table.add_row("🔄 Interpolation", f"{config.get('interpolate', 'linear')}")
    dataloader_table.add_row("📏 Extrapolation", f"{config.get('extrapolate', 'N/A')}")
    
    # Cohérence Activity Proxies
    dataset_has_proxies = dataset.metadata.get("activity_proxies_included", False)
    model_uses_proxies = model.include_activity_proxies
    
    if dataset_has_proxies and model_uses_proxies:
        proxies_status = "✅ Cohérent (dataset + modèle)"
    elif dataset_has_proxies and not model_uses_proxies:
        proxies_status = "⚠️  Dataset a des proxies, modèle non"
    elif not dataset_has_proxies and model_uses_proxies:
        proxies_status = "⚠️  Modèle attend des proxies, dataset non"
    else:
        proxies_status = "✅ Cohérent (aucun proxy)"
    
    dataloader_table.add_row("⭐ Activity Proxies", proxies_status)
    
    # Template cohérence
    dataset_has_template = dataset.template is not None
    model_uses_template = config.get("loss_b_rest", False)
    
    if dataset_has_template and model_uses_template:
        template_status = "✅ Cohérent (template utilisé)"
    elif not dataset_has_template and model_uses_template:
        template_status = "⚠️  Modèle attend template, dataset non"
    else:
        template_status = "✅ Cohérent"
    
    dataloader_table.add_row("� Template", template_status)
    
    console.print(dataloader_table)
    console.print()  # Ligne vide

    # Entraînement par phases
    console.rule("[bold green]🎭 DÉBUT DE L'ENTRAÎNEMENT PAR PHASES[/]", style="bold green")
    
    # Tableau récapitulatif des phases
    phases_overview_table = Table(title="Récapitulatif des Phases d'Entraînement", expand=True)
    phases_overview_table.add_column("Phase", style="cyan")
    phases_overview_table.add_column("Epochs", style="white")
    phases_overview_table.add_column("Optimiseur", style="yellow")
    phases_overview_table.add_column("LR", style="green")
    phases_overview_table.add_column("Paramètres Entraînables", style="magenta")
    
    for phase_config in config["phases"]:
        phase_name = phase_config["name"]
        n_epochs = phase_config["n_epochs"]
        optimizer_name = phase_config.get("optimizer", "N/A").split(".")[-1]  # Juste le nom de la classe
        lr = phase_config.get("optimizer_kwargs", {}).get("lr", "N/A")
        
        # Résumé des paramètres entraînables
        trainable_params = phase_config.get("trainable_params", {})
        trainable_summary = []
        for param, is_trainable in trainable_params.items():
            if is_trainable:
                trainable_summary.append(param)
        trainable_str = ", ".join(trainable_summary) if trainable_summary else "Aucun"
        
        phases_overview_table.add_row(
            phase_name,
            str(n_epochs),
            optimizer_name,
            str(lr),
            trainable_str
        )
    
    console.print(phases_overview_table)
    console.print()  # Ligne vide
    
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

    console.rule("[bold green]🎉 ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS ![/]", style="bold green")
    
    # Tableau de synthèse finale
    final_summary_table = Table(title="Synthèse Finale de l'Expérience", expand=True)
    final_summary_table.add_column("Élément", style="cyan", width=25)
    final_summary_table.add_column("Information", style="white")
    
    final_summary_table.add_row("🏷️ Expérience", f"{exp_name}")
    final_summary_table.add_row("📁 Dossier de sortie", f"{exp_dirs['experiment_dir']}")
    final_summary_table.add_row("💾 Modèle final", f"{final_model_path}")
    final_summary_table.add_row("📊 Checkpoints", f"{exp_dirs['models_dir']}")
    final_summary_table.add_row("📈 Figures", f"{exp_dirs['figures_dir']}")
    final_summary_table.add_row("📋 Logs", f"{exp_dirs['logs_dir']}")
    final_summary_table.add_row("🎭 Phases exécutées", f"{len(config['phases'])}")
    
    # Compter le nombre total d'epochs exécutées
    total_epochs_executed = sum(phase.get("n_epochs", 0) for phase in config["phases"])
    final_summary_table.add_row("⏱️ Total epochs", f"{total_epochs_executed}")
    
    console.print(final_summary_table)
    console.print()
    
    console.print("📋 [bold]Fichiers générés:[/]")
    console.print(f"   • Modèle final: [green]{os.path.basename(final_model_path)}[/]")
    console.print(f"   • Configuration: [green]config.yaml[/]")
    console.print("   • Checkpoints intermédiaires dans [green]models/[/]")
    console.print("   • Figures de suivi dans [green]figures/[/]")
    console.print()
    
    console.print("🚀 [bold]Prochaines étapes suggérées:[/]")
    console.print("   • Utiliser [green]predict.py[/] pour faire des prédictions")
    console.print("   • Analyser les figures dans le dossier [green]figures/[/]")
    console.print("   • Consulter les logs dans le dossier [green]logs/[/]")
    console.print()


if __name__ == "__main__":
    main(
        config_path="src/modeling/configs/aestra_perfect_1000_spectra.yaml",
        dataset_filepath="data/npz_datasets/soapgpu_ns1275_5000-5050_p100_k0p5_phi0.npz",
        output_root_dir="experiments",
        experiment_name="aestra_perfect",
    )
