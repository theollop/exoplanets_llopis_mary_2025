from models import AESTRA
from src.dataset import get_dataloader
import torch
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

console = Console()

console.log("🔄 Récupération du dataloader…")
dataloader = get_dataloader(
    n_specs=200,
    wavemin=5000.0,
    wavemax=5010.0,
    batch_size=16,
    shuffle=False,
    vmin=-3,
    vmax=3,
    data_dtype=torch.float32,
    out_dtype=torch.float32,
    interpolate="linear",
)

batch_template = dataloader.dataset.template.unsqueeze(0)
b_wavegrid = dataloader.dataset.b_wavegrid

console.log("🏗️  Création du modèle AESTRA…")
aestra = AESTRA(n_pixels=dataloader.dataset.n_pixels, S=3)

phases = {
    "rv": {
        "n_epochs": 200,
        "l_r": 0.001,
    },
    "joint": {
        "n_epochs": 200,
        "l_r": 0.001,
    },
}
aestra.train()
for phase, config in phases.items():
    console.rule(f"[bold green]Phase: {phase}[/]")
    optimizer = torch.optim.Adam(
        aestra.parameters(),
        lr=config["l_r"],
    )
    aestra.set_phase(phase)

    # Création d’une table pour afficher les losses en direct
    table = Table(expand=True)
    table.add_column("Epoch", justify="right")
    table.add_column("RV", justify="right")
    table.add_column("FID", justify="right")
    table.add_column("C", justify="right")
    table.add_column("Reg", justify="right")
    table.add_column("Total Loss", justify="right")

    # Progress: une barre pour les epochs
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        epoch_task = progress.add_task("Epochs", total=config["n_epochs"])
        for epoch in range(config["n_epochs"]):
            print(f"Epoch {epoch + 1}/{config['n_epochs']}")

            epoch_losses = {
                "rv": 0.0,
                "fid": 0.0,
                "c": 0.0,
                "reg": 0.0,
            }

            for batch in dataloader:
                B = batch[0].shape[0]  # batch size courante
                batch_yobs, batch_yaug, batch_voffset, batch_wavegrid = batch

                # plt.figure(figsize=(10, 5))
                # plt.plot(
                #     batch_wavegrid[0].cpu().numpy(),
                #     batch_yobs[0].cpu().numpy(),
                #     label="Observé",
                # )
                # plt.plot(
                #     batch_wavegrid[0].cpu().numpy(),
                #     batch_yaug[0].cpu().numpy(),
                #     label="Augmenté",
                # )
                # plt.figtext(
                #     0.5,
                #     0.01,
                #     f"v = {batch_voffset[0].cpu().numpy()}",
                #     ha="center",
                #     fontsize=12,
                # )
                # plt.title(f"Batch {B} - Observations")
                # plt.xlabel("Longueur d'onde")
                # plt.ylabel("Flux")
                # plt.legend()
                # plt.show()
                # break

                losses = aestra.get_losses(
                    batch=batch,
                    batch_template=batch_template,
                    b_wavegrid=batch_wavegrid,
                    extrapolate="linear",
                    batch_weights=None,
                )

                epoch_losses["rv"] += losses["rv"] * B
                epoch_losses["fid"] += losses["fid"] * B
                epoch_losses["c"] += losses["c"] * B
                epoch_losses["reg"] += losses["reg"] * B

            epoch_losses["rv"] /= len(dataloader.dataset)
            epoch_losses["fid"] /= len(dataloader.dataset)
            epoch_losses["c"] /= len(dataloader.dataset)
            epoch_losses["reg"] /= len(dataloader.dataset)

            loss = (
                epoch_losses["rv"]
                + epoch_losses["fid"]
                + epoch_losses["c"]
                + epoch_losses["reg"]
            )

            # Ajout d’une ligne dans la table
            table.add_row(
                f"{epoch + 1}/{config['n_epochs']}",
                f"{epoch_losses['rv']:.4f}",
                f"{epoch_losses['fid']:.4f}",
                f"{epoch_losses['c']:.4f}",
                f"{epoch_losses['reg']:.4f}",
                f"{loss:.4f}",
            )

            # Affiche la table mise à jour à chaque epoch
            console.clear()
            console.print(table)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress.advance(epoch_task)

    console.log(f"✅ Phase '{phase}' terminée.\n")
console.log("🎉 Entraînement terminé !")

# Save the model
torch.save(
    aestra.state_dict(),
    "/home/tliopis/Codes/exoplanets_llopis_mary_2025/models/aestra_model_state_dict.pth",
)

# Test

all_rvs = []

with torch.no_grad():
    for batch in dataloader:
        batch_yobs, batch_yaug, batch_voffset, batch_wavegrid = batch

        batch_rvs = aestra.rvestimator(batch_yobs - batch_template)

        all_rvs.append(batch_rvs.cpu().numpy())

all_rvs = np.concatenate(all_rvs, axis=0)

plt.figure(figsize=(10, 5))
plt.title("RV estimés par AESTRA")
plt.plot(
    dataloader.dataset.jdb,
    all_rvs,
    label="RV estimés",
    marker="o",
    linestyle="None",
    markersize=4,
    color="blue",
)
plt.ylabel("RV (m/s)")
plt.xlabel("Temps")
plt.grid()
plt.show()

from astropy.timeseries import LombScargle

# Lomb-Scargle periodogram
t = dataloader.dataset.jdb
frequencies, power = LombScargle(t, all_rvs).autopower(samples_per_peak=10)

# Convert frequencies to periods
period = 1 / frequencies
plt.figure(figsize=(10, 5))
plt.title("Lomb-Scargle Periodogram")
plt.plot(period, power)
plt.xlabel("Période (s)")
plt.ylabel("Puissance")
plt.grid()
plt.show()
