#!/usr/bin/env python3
"""
Test rapide de l'organisation des plots par phase et type.
"""

import os
import tempfile
from src.plots_aestra import create_typed_plot_dir


def test_plot_organization():
    """Test la nouvelle organisation des plots."""

    # Créer un répertoire temporaire pour les tests
    with tempfile.TemporaryDirectory() as temp_dir:
        base_figures_dir = os.path.join(temp_dir, "figures")

        # Test de l'organisation par phase
        print("🧪 Test de l'organisation des plots par phase et type...")

        # Créer des dossiers pour différentes phases
        phases = ["rvonly", "joint"]
        plot_types = [
            "losses",
            "rv_predictions",
            "analysis",
            "activity",
            "ultra_doppler",
        ]

        for phase in phases:
            for plot_type in plot_types:
                typed_dir = create_typed_plot_dir(base_figures_dir, phase, plot_type)
                print(f"📁 Créé: {typed_dir}")

                # Vérifier que le dossier existe
                assert os.path.exists(typed_dir), f"Dossier non créé: {typed_dir}"

        # Afficher la structure créée
        print("\n📂 Structure créée:")
        for root, dirs, files in os.walk(base_figures_dir):
            level = root.replace(base_figures_dir, "").count(os.sep)
            indent = " " * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            sub_indent = " " * 2 * (level + 1)
            for dir_name in dirs:
                print(f"{sub_indent}{dir_name}/")

        print("\n✅ Test réussi ! La nouvelle organisation fonctionne correctement.")
        print("\n📋 Résumé de l'organisation:")
        print("   experiments/")
        print("   └── [experiment_name]/")
        print("       └── figures/")
        print("           ├── rvonly/")
        print("           │   ├── losses/")
        print("           │   ├── rv_predictions/")
        print("           │   ├── analysis/")
        print("           │   ├── activity/")
        print("           │   └── ultra_doppler/")
        print("           └── joint/")
        print("               ├── losses/")
        print("               ├── rv_predictions/")
        print("               ├── analysis/")
        print("               ├── activity/")
        print("               └── ultra_doppler/")
        print("\n🎯 Noms de fichiers simplifiés:")
        print("   - losses_epoch_50.png")
        print("   - rv_predictions_full_epoch_50.png")
        print("   - aestra_analysis_epoch_50.png")
        print("   - activity_comparison_epoch_50.png")
        print("   - ultra_doppler_epoch_50.png")


if __name__ == "__main__":
    test_plot_organization()
