#!/usr/bin/env python3
"""
Script d'entraînement professionnel pour AESTRA avec gestion de config YAML et checkpoints.

Usage:
    python train_pro.py exp0 [--checkpoint path/to/checkpoint.pth]

Exemple:
    python train_pro.py exp0                           # Nouvel entraînement avec config exp0
    python train_pro.py exp0 --checkpoint models/aestra_exp0_epoch_100.pth  # Reprendre depuis checkpoint
"""

import argparse
import os
import yaml
import torch
import csv
from datetime import datetime
from torch.utils.data import DataLoader
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
from src.plots_aestra import plot_losses, plot_aestra_analysis

console = Console()


def save_experiment_checkpoint(
    model, optimizer, scheduler, dataset, config, cfg_name, epoch, phase_name, path=None
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
        path: Chemin de sauvegarde (optionnel)
    """
    if path is None:
        path = f"models/aestra_{cfg_name}_phase_{phase_name}_epoch_{epoch}.pth"

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

    torch.save(ckpt, path)
    console.log(f"💾 Experiment checkpoint saved: {path}")


def load_experiment_checkpoint(path, device="cuda"):
    """
    Charge un checkpoint d'expérience complet.

    Returns:
        dict: Contient model, optimizer, scheduler, dataset, config, cfg_name, epoch
    """
    console.log(f"📂 Loading experiment checkpoint: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device)

    # Reconstruction du dataset
    dataset_metadata = ckpt["dataset_metadata"]
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
        device=device,  # ⚠️ NOUVEAU: Passer le device explicitement
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.set_phase(ckpt.get("model_phase", "joint"))

    if torch.cuda.is_available():
        model = model.cuda()

    return {
        "model": model,
        "dataset": dataset,
        "config": config,
        "cfg_name": ckpt["cfg_name"],
        "epoch": ckpt["epoch"],
        "current_phase": ckpt.get("current_phase", "joint"),  # Phase actuelle
        "checkpoint_data": ckpt,
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


def train_phase(
    model, dataset, dataloader, phase_config, config, cfg_name, start_epoch=0
):
    """Entraîne le modèle pour une phase donnée."""
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

                losses = model.get_losses(
                    batch=batch,
                    extrapolate="linear",
                    batch_weights=None,
                    iteration_count=it,
                )

                # Calculer la loss totale pour ce batch
                total_batch_loss = sum(losses.values())

                # ⚠️ CRITIQUE: Backward et step à chaque batch pour libérer la mémoire
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
                plot_dir = config.get("plot_dir", "reports/figures")
                plot_losses(
                    losses_history, cfg_name, phase_name, epoch + 1, plot_dir, console
                )

            # Plots de spectres périodiques
            plot_spectra_every = phase_config.get("plot_spectra_every", 0)
            if plot_spectra_every > 0 and (epoch + 1) % plot_spectra_every == 0:
                spectra_plot_dir = phase_config.get(
                    "spectra_plot_dir", "reports/spectra"
                )
                plot_aestra_analysis(
                    batch,
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
                csv_dir = config.get("csv_dir", "reports/logs")
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
        plot_dir = config.get("plot_dir", "reports/figures")
        plot_losses(losses_history, cfg_name, phase_name, n_epochs, plot_dir, console)

    # Sauvegarde CSV finale de la phase
    if config.get("save_losses_csv", False):
        csv_dir = config.get("csv_dir", "reports/logs")
        save_losses_to_csv(
            losses_history, cfg_name, phase_name, n_epochs, csv_dir, config
        )


def main(cfg_name=None, checkpoint=None, device="cuda"):
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
        checkpoint = args.checkpoint or checkpoint
        device = args.device
    else:
        # Création d'un objet args simulé pour compatibilité avec le reste du code
        class Args:
            def __init__(self, cfg_name, checkpoint, device):
                self.cfg_name = cfg_name
                self.checkpoint = checkpoint
                self.device = device

        args = Args(cfg_name, checkpoint, device)

    console.rule(f"[bold blue]AESTRA Training - Experiment: {args.cfg_name}[/]")

    # Chargement depuis checkpoint ou nouvelle expérience
    if args.checkpoint:
        # Reprendre depuis checkpoint
        exp_data = load_experiment_checkpoint(f"models/{args.checkpoint}", args.device)
        model = exp_data["model"]
        dataset = exp_data["dataset"]
        config = exp_data["config"]
        start_epoch = exp_data["epoch"]
        current_phase = exp_data["current_phase"]

        console.log(f"🔄 Resuming from epoch {start_epoch}, phase '{current_phase}'")

    else:
        # Nouvelle expérience
        config = load_config(args.cfg_name)
        console.log("✅ Configuration chargée avec succès")

        current_phase = None
        start_epoch = 0

        console.log("🔧 Début de la création du dataset...")
        console.log(f"📁 data_root_dir: {config.get('data_root_dir', 'data')}")

        # Création du dataset
        try:
            dataset = SpectrumDataset(
                n_specs=config.get("n_specs", None),
                wavemin=config.get("wavemin", None),
                wavemax=config.get("wavemax", None),
                data_dtype=getattr(torch, config.get("data_dtype", "float32")),
                data_root_dir=config.get("data_root_dir", "data"),
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
                b_obs=dataset.template,
                b_rest=dataset.spectra.mean(dim=0),
                device=args.device,  # ⚠️ NOUVEAU: Passer le device explicitement
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
                    )
                break

        if not phase_found:
            console.log(
                f"❌ Phase '{current_phase}' introuvable dans la config, démarrage normal"
            )
            for phase_config in config["phases"]:
                train_phase(
                    model, dataset, dataloader, phase_config, config, args.cfg_name, 0
                )
    else:
        # Nouvel entraînement - toutes les phases depuis le début
        for phase_config in config["phases"]:
            train_phase(
                model, dataset, dataloader, phase_config, config, args.cfg_name, 0
            )

    # Sauvegarde finale
    final_path = f"models/aestra_{args.cfg_name}_final.pth"
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
