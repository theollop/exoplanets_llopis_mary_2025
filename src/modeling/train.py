#!/usr/bin/env python3
"""
Script d'entraînement professionnel pour AESTRA avec gestion de config YAML et checkpoints.

Usage:
    python train_pro.py exp0 [--checkpoint path/to/checkpoint.pth]

Exemple:
    python train_pro.py exp0                           # Nouvel entraînement avec config exp0
    python train_pro.py exp0 --checkpoint aestra_joint_100.pth  # Reprendre depuis checkpoint (recherché dans models/)
"""

import argparse
import os
import yaml
import torch
import csv
from datetime import datetime
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
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


def setup_experiment_directories(config, config_name):
    """
    Crée la structure de dossiers pour une expérience d'entraînement.

    Args:
        config: Configuration de l'expérience
        config_name: Nom du fichier de configuration

    Returns:
        dict: Dictionnaire avec les chemins des dossiers créés
    """
    # Nom de l'expérience depuis la config ou défaut
    experiment_name = config.get("experiment_name", f"experiment_{config_name}")
    output_root = config.get("output_root_dir", "experiments")

    # Dossier principal de l'expérience
    exp_dir = os.path.join(output_root, experiment_name)

    # Structure des sous-dossiers
    subdirs = {
        "experiment_dir": exp_dir,
        "models_dir": os.path.join(exp_dir, "models"),
        "figures_dir": os.path.join(exp_dir, "figures"),
        "spectra_dir": os.path.join(exp_dir, "spectra"),
        "logs_dir": os.path.join(exp_dir, "logs"),
    }

    # Créer tous les dossiers
    for dir_path in subdirs.values():
        os.makedirs(dir_path, exist_ok=True)

    # Sauvegarder la configuration utilisée dans le dossier d'expérience
    config_save_path = os.path.join(exp_dir, f"{config_name}_config.yaml")
    with open(config_save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    console.print(f"📁 Structure d'expérience créée dans: {exp_dir}")
    console.print(f"📋 Configuration sauvegardée: {config_save_path}")

    return subdirs


def save_experiment_checkpoint(
    model,
    optimizer,
    scheduler,
    dataset,
    config,
    cfg_name,
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
        cfg_name: Nom de l'expérience (ex: "exp0")
        epoch: Numéro d'epoch actuel
        phase_name: Nom de la phase actuelle
        scaler: Le GradScaler pour mixed precision (peut être None)
        path: Chemin de sauvegarde (optionnel)
        exp_dirs: Dictionnaire des dossiers d'expérience
    """
    if path is None:
        if exp_dirs is not None:
            path = os.path.join(
                exp_dirs["models_dir"],
                f"aestra_{phase_name}_checkpoint.pth",
            )
        else:
            # Fallback vers l'ancien système
            path = f"models/aestra_{phase_name}_checkpoint.pth"

    # Sauvegarde du checkpoint standard
    save_checkpoint(model, optimizer, path, scheduler)

    # Ajout des métadonnées de l'expérience
    ckpt = torch.load(path)
    ckpt.update(
        {
            "cfg_name": cfg_name,
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
        dict: Contient model, optimizer, scheduler, dataset, config, cfg_name, epoch, scaler_state_dict
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
        "cfg_name": ckpt["cfg_name"],
        "epoch": ckpt["epoch"],
        "current_phase": ckpt.get("current_phase", "joint"),
        "checkpoint_data": ckpt,
        "scaler_state_dict": ckpt.get("scaler_state_dict", None),
    }


def load_config(config_name):
    """Charge un fichier de configuration depuis configs/"""
    config_path = f"src/modeling/configs/{config_name}.yaml"

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    console.log(f"📋 Config loaded: {config_path}")
    return config


def save_losses_to_csv(losses_history, cfg_name, phase_name, epoch, csv_dir, config):
    """
    Sauvegarde les losses dans un fichier CSV.

    Args:
        losses_history: Dict avec les listes de losses {'rv': [...], 'fid': [...], etc.}
        cfg_name: Nom de l'expérience
        phase_name: Nom de la phase actuelle
        epoch: Epoch actuelle
        csv_dir: Répertoire de sauvegarde des CSV
        config: Configuration complète (pour les métadonnées)
    """
    if not config.get("save_losses_csv", False):
        return  # CSV désactivé

    os.makedirs(csv_dir, exist_ok=True)

    # Nom du fichier CSV
    csv_filename = f"{cfg_name}_{phase_name}_losses.csv"
    csv_path = os.path.join(csv_dir, csv_filename)

    # Vérifier si le fichier existe déjà pour savoir si on ajoute les headers
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "w" if not file_exists else "a", newline="") as csvfile:
        fieldnames = [
            "timestamp",
            "cfg_name",
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
                    "cfg_name": cfg_name,
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

    scaler = GradScaler(
        init_scale=config.get("grad_scaler_init_scale", 65536.0),
        growth_factor=config.get("grad_scaler_growth_factor", 2.0),
        backoff_factor=config.get("grad_scaler_backoff_factor", 0.5),
        growth_interval=config.get("grad_scaler_growth_interval", 2000),
        enabled=True,
    )

    console.log("🚀 Mixed precision activée avec GradScaler")
    return scaler


def train_phase(
    model,
    dataset,
    dataloader,
    phase_config,
    config,
    cfg_name,
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
                B = batch[0].shape[0]

                # ⚠️ CRITIQUE: Reset gradients à chaque batch
                optimizer.zero_grad()

                # Forward pass avec ou sans autocast selon la configuration
                if autocast_enabled:
                    with autocast():
                        losses = model.get_losses(
                            batch=batch,
                            extrapolate="linear",
                            batch_weights=None,
                            iteration_count=it,
                        )
                        # Calculer la loss totale pour ce batch
                        total_batch_loss = sum(losses.values())
                else:
                    losses = model.get_losses(
                        batch=batch,
                        extrapolate="linear",
                        batch_weights=None,
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
                plot_losses(
                    losses_history, cfg_name, phase_name, epoch + 1, plot_dir, console
                )

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
                        cfg_name,
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
                        cfg_name,
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
                    cfg_name,
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
                    losses_history, cfg_name, phase_name, epoch + 1, csv_dir, config
                )

            # Sauvegarde périodique (tous les 50 epochs)
            if (epoch + 1) % config.get("checkpoint_every", 50) == 0:
                save_experiment_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    dataset,
                    config,
                    cfg_name,
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

    # Plot final de la phase
    plot_every = config.get("plot_every", 0)
    if plot_every > 0:
        plot_dir = (
            exp_dirs["figures_dir"]
            if exp_dirs
            else config.get("plot_dir", "reports/figures")
        )
        plot_losses(losses_history, cfg_name, phase_name, n_epochs, plot_dir, console)

    # Sauvegarde CSV finale de la phase
    if config.get("save_losses_csv", False):
        csv_dir = (
            exp_dirs["logs_dir"] if exp_dirs else config.get("csv_dir", "reports/logs")
        )
        save_losses_to_csv(
            losses_history, cfg_name, phase_name, n_epochs, csv_dir, config
        )


def main(cfg_name=None, checkpoint_path=None, device="cuda"):
    """
    Fonction principale d'entraînement AESTRA.

    Args:
        cfg_name: Nom de la configuration (ex: "base_config", "colab_config")
        checkpoint: Chemin vers un checkpoint pour reprendre l'entraînement
        device: Device à utiliser ("cuda" ou "cpu")
    """
    # Si aucun argument n'est fourni, utiliser argparse pour la ligne de commande
    if cfg_name is None:
        parser = argparse.ArgumentParser(
            description="Entraînement AESTRA avec config YAML"
        )
        parser.add_argument(
            "--cfg_name", help="Nom de l'expérience (ex: exp0)", default="base_config"
        )
        parser.add_argument(
            "--checkpoint",
            help="Chemin vers un checkpoint pour reprendre l'entraînement",
        )
        parser.add_argument(
            "--device", default="cuda", help="Device à utiliser (cuda/cpu)"
        )

        args = parser.parse_args()
        cfg_name = args.cfg_name
        checkpoint_path = args.checkpoint or checkpoint_path
        device = args.device
    else:
        # Création d'un objet args simulé pour compatibilité avec le reste du code
        class Args:
            def __init__(self, cfg_name, checkpoint_path, device):
                self.cfg_name = cfg_name
                self.checkpoint_path = checkpoint_path
                self.device = device

        args = Args(cfg_name, checkpoint_path, device)

    console.rule(f"[bold blue]AESTRA Training - Experiment: {args.cfg_name}[/]")

    # Chargement depuis checkpoint ou nouvelle expérience
    if args.checkpoint_path:
        # Reprendre depuis checkpoint
        exp_data = load_experiment_checkpoint(args.checkpoint_path, args.device)
        model = exp_data["model"]
        dataset = exp_data["dataset"]
        config = exp_data["config"]
        start_epoch = exp_data["epoch"]
        current_phase = exp_data["current_phase"]

        # Récupérer la structure de dossiers depuis la config
        exp_dirs = setup_experiment_directories(config, args.cfg_name)

        console.log(f"🔄 Resuming from epoch {start_epoch}, phase '{current_phase}'")

    else:
        # Nouvelle expérience
        config = load_config(args.cfg_name)
        console.log("✅ Configuration chargée avec succès")

        # Configuration de la structure de dossiers pour l'expérience
        exp_dirs = setup_experiment_directories(config, args.cfg_name)

        current_phase = None
        start_epoch = 0

        console.log("🔧 Début de la création du dataset...")
        console.log(f"📁 dataset_filepath: {config.get('dataset_filepath')}")

        # Création du dataset (NPZ standardisé uniquement)
        try:
            dataset = SpectrumDataset(
                dataset_filepath=config.get(
                    "dataset_filepath",
                    "data/npz_datasets/dataset_1000specs_5000_5050_Kp1e-1_P100.npz",
                ),
                data_dtype=getattr(torch, config.get("data_dtype", "float32")),
                cuda=True,
            )
            console.log("✅ Dataset créé avec succès")
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
                device=args.device,  # ⚠️ NOUVEAU: Passer le device explicitement
                dtype=getattr(
                    torch, config.get("model_dtype", "float32")
                ),  # ⚠️ NOUVEAU: Support dtype modèle
            )
            console.log("✅ Modèle créé avec succès")
        except Exception as e:
            console.log(f"❌ Erreur lors de la création du modèle: {e}")
            raise

        if torch.cuda.is_available():
            console.log("🚀 Déplacement du modèle vers le GPU...")
            model = model.cuda()
            console.log("✅ Modèle sur GPU")
        else:
            console.log("⚠️  CUDA non disponible, utilisation du CPU")

        console.log("🆕 Starting new experiment")

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

    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=config["shuffle"],
        collate_fn=collate_fn,
    )

    console.log(f"📊 Dataset: {len(dataset)} spectres, {dataset.n_pixels} pixels")
    console.log(f"🔧 Modèle: {sum(p.numel() for p in model.parameters())} paramètres")
    console.log(f"Batch size: {config['batch_size']}")

    # Entraînement par phases
    if current_phase is not None:
        # Reprendre depuis une phase spécifique
        phase_found = False
        for i, phase_config in enumerate(config["phases"]):
            if phase_config["name"] == current_phase:
                # Continuer la phase actuelle avec start_epoch
                train_phase(
                    model,
                    dataset,
                    dataloader,
                    phase_config,
                    config,
                    args.cfg_name,
                    start_epoch,
                    exp_dirs,
                )
                phase_found = True

                # Puis continuer avec les phases suivantes (s'il y en a)
                for next_phase_config in config["phases"][i + 1 :]:
                    train_phase(
                        model,
                        dataset,
                        dataloader,
                        next_phase_config,
                        config,
                        args.cfg_name,
                        0,
                        exp_dirs,
                    )
                break

        if not phase_found:
            console.log(
                f"❌ Phase '{current_phase}' introuvable dans la config, démarrage normal"
            )
            for phase_config in config["phases"]:
                train_phase(
                    model,
                    dataset,
                    dataloader,
                    phase_config,
                    config,
                    args.cfg_name,
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
                args.cfg_name,
                0,
                exp_dirs,
            )

    # Sauvegarde finale
    if "exp_dirs" in locals():
        final_path = os.path.join(exp_dirs["models_dir"], "aestra_final.pth")
    else:
        final_path = "models/aestra_final.pth"

    # Pour la sauvegarde finale, on sauve juste le modèle et la config
    ckpt = {
        "model_state_dict": model.state_dict(),
        "model_phase": model.phase,
        "cfg_name": args.cfg_name,
        "epoch": "final",
        "config": config,
        "dataset_metadata": dataset.to_dict(),
    }
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    torch.save(ckpt, final_path)

    console.rule("[bold green]🎉 Entraînement terminé ![/]")
    console.log(f"Modèle final sauvé: {final_path}")


if __name__ == "__main__":
    main()
