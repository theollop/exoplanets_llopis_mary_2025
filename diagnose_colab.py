#!/usr/bin/env python3
"""
Script de diagnostic pour identifier les problèmes dans Colab.
"""

import yaml
import os
import torch
from pathlib import Path


def diagnose_colab_issue():
    """Diagnostic complet pour identifier le problème sur Colab."""

    print("🔍 DIAGNOSTIC COLAB - Début")
    print("=" * 50)

    # Test 1: Chargement de la configuration
    print("\n1️⃣  Test du chargement de configuration...")
    try:
        config_path = "src/modeling/configs/colab_config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        print(f"✅ Configuration chargée: {config_path}")
        print(f"📁 data_root_dir: {config.get('data_root_dir')}")
    except Exception as e:
        print(f"❌ Erreur chargement config: {e}")
        return

    # Test 2: Vérification des chemins de données
    print("\n2️⃣  Test de l'existence des chemins de données...")
    data_root_dir = config.get("data_root_dir", "data")

    # Chemins qui seront construits par le dataset
    expected_paths = [
        f"{data_root_dir}/rv_datachallenge/Sun_B57001_E61001_planet-FallChallenge1/HARPN/STAR1136_HPN_flux_YVA.npy",
        f"{data_root_dir}/rv_datachallenge/Sun_B57001_E61001_planet-FallChallenge1/HARPN/STAR1136_HPN_Analyse_material.p",
        f"{data_root_dir}/rv_datachallenge/Sun_B57001_E61001_planet-FallChallenge1/HARPN/STAR1136_HPN_Analyse_summary.csv",
        f"{data_root_dir}/rv_datachallenge/masks/G2_mask.txt",
    ]

    for path in expected_paths:
        if os.path.exists(path):
            print(f"✅ Trouvé: {path}")
        else:
            print(f"❌ Manquant: {path}")

    # Test 3: Vérification des imports
    print("\n3️⃣  Test des imports...")
    try:
        from src.dataset import SpectrumDataset

        print("✅ Import SpectrumDataset réussi")
    except Exception as e:
        print(f"❌ Erreur import SpectrumDataset: {e}")
        return

    try:
        from src.modeling.models import AESTRA

        print("✅ Import AESTRA réussi")
    except Exception as e:
        print(f"❌ Erreur import AESTRA: {e}")
        return

    # Test 4: Test de création du dataset (avec gestion d'erreur détaillée)
    print("\n4️⃣  Test de création du dataset...")
    try:
        print("📊 Tentative de création du dataset...")
        dataset = SpectrumDataset(
            n_specs=10,  # Petit test
            wavemin=config.get("wavemin", None),
            wavemax=config.get("wavemax", None),
            data_dtype=getattr(torch, config.get("data_dtype", "float32")),
            data_root_dir=data_root_dir,
        )
        print("✅ Dataset créé avec succès!")
        print(f"📈 Spectres: {dataset.n_specs}, Pixels: {dataset.n_pixels}")
    except FileNotFoundError as e:
        print(f"❌ Fichier non trouvé: {e}")
        print("💡 Suggestion: Vérifiez que les données sont bien dans Google Drive")
    except Exception as e:
        print(f"❌ Erreur dataset: {e}")
        print(f"🔍 Type d'erreur: {type(e).__name__}")
        import traceback

        traceback.print_exc()

    # Test 5: Test de l'environnement CUDA
    print("\n5️⃣  Test de l'environnement CUDA...")
    if torch.cuda.is_available():
        print(f"✅ CUDA disponible: {torch.cuda.get_device_name(0)}")
        print(
            f"🔋 Mémoire GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
    else:
        print("⚠️  CUDA non disponible - utilisation du CPU")

    # Test 6: Test d'espace disque
    print("\n6️⃣  Test de l'espace disque...")
    try:
        import shutil

        total, used, free = shutil.disk_usage("/")
        print(f"💾 Espace disque libre: {free / 1e9:.1f} GB")

        if Path(data_root_dir).exists():
            total_drive, used_drive, free_drive = shutil.disk_usage(data_root_dir)
            print(f"☁️  Espace Google Drive libre: {free_drive / 1e9:.1f} GB")
    except Exception as e:
        print(f"⚠️  Impossible de vérifier l'espace disque: {e}")

    print("\n🎯 DIAGNOSTIC TERMINÉ")
    print("\n💡 CONSEILS DE DÉPANNAGE:")
    print("   • Vérifiez que Google Drive est monté: drive.mount('/content/drive')")
    print("   • Vérifiez que les données sont dans le bon dossier Google Drive")
    print("   • Essayez de redémarrer le runtime Colab si problème de mémoire")
    print("   • Vérifiez les permissions d'accès aux fichiers")


if __name__ == "__main__":
    diagnose_colab_issue()
