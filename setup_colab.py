#!/usr/bin/env python3
"""
Script de setup pour Google Colab.
À exécuter avant l'entraînement dans Colab.
"""

import os
import shutil
from pathlib import Path


def setup_colab_environment():
    """Configure l'environnement Colab pour l'entraînement AESTRA."""

    print("🚀 SETUP COLAB - Configuration de l'environnement")
    print("=" * 60)

    # 1. Vérifier le montage de Google Drive
    print("\n1️⃣  Vérification de Google Drive...")
    if os.path.exists("/content/drive"):
        print("✅ Google Drive détecté")
        if os.path.exists("/content/drive/MyDrive"):
            print("✅ Google Drive monté correctement")
        else:
            print("❌ Google Drive non monté. Exécutez: drive.mount('/content/drive')")
            return False
    else:
        print("❌ Google Drive non détecté. Exécutez: drive.mount('/content/drive')")
        return False

    # 2. Créer les dossiers de résultats
    print("\n2️⃣  Création des dossiers de résultats...")
    result_dirs = [
        "/content/results",
        "/content/results/figures",
        "/content/results/spectra",
        "/content/results/logs",
        "/content/results/models",
    ]

    for dir_path in result_dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"📁 Créé: {dir_path}")

    # 3. Vérifier la présence des données
    print("\n3️⃣  Vérification des données...")

    # Option 1: Données dans le repository cloné
    repo_data_dir = "/content/exoplanets_llopis_mary_2025/data"
    if os.path.exists(repo_data_dir):
        print(f"✅ Données trouvées dans le repository: {repo_data_dir}")
        recommended_config = "colab_test_config"
        data_source = "repository"
    else:
        print(f"❌ Données non trouvées dans le repository: {repo_data_dir}")

        # Option 2: Données sur Google Drive
        drive_data_dir = "/content/drive/MyDrive/PFE/data"
        if os.path.exists(drive_data_dir):
            print(f"✅ Données trouvées sur Google Drive: {drive_data_dir}")
            recommended_config = "colab_config"
            data_source = "drive"
        else:
            print(f"❌ Données non trouvées sur Google Drive: {drive_data_dir}")
            print("\n💡 SOLUTION: Copiez les données dans l'un de ces emplacements")
            return False

    # 4. Vérifier les fichiers critiques
    print(f"\n4️⃣  Vérification des fichiers critiques...")

    if data_source == "repository":
        base_path = repo_data_dir
    else:
        base_path = drive_data_dir

    critical_files = [
        f"{base_path}/rv_datachallenge/Sun_B57001_E61001_planet-FallChallenge1/HARPN/STAR1136_HPN_flux_YVA.npy",
        f"{base_path}/rv_datachallenge/Sun_B57001_E61001_planet-FallChallenge1/HARPN/STAR1136_HPN_Analyse_material.p",
        f"{base_path}/rv_datachallenge/Sun_B57001_E61001_planet-FallChallenge1/HARPN/STAR1136_HPN_Analyse_summary.csv",
        f"{base_path}/rv_datachallenge/masks/G2_mask.txt",
    ]

    all_files_ok = True
    for file_path in critical_files:
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / 1e6
            print(f"✅ {os.path.basename(file_path)} ({size_mb:.1f} MB)")
        else:
            print(f"❌ Manquant: {os.path.basename(file_path)}")
            all_files_ok = False

    if not all_files_ok:
        print("\n❌ Fichiers manquants détectés")
        return False

    # 5. Tester l'import des modules
    print(f"\n5️⃣  Test des imports...")
    try:
        import torch

        print(f"✅ PyTorch {torch.__version__}")

        # Test GPU
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"💾 Mémoire GPU: {memory_gb:.1f} GB")
        else:
            print("⚠️  GPU non disponible")

    except ImportError:
        print("❌ PyTorch non installé")
        return False

    # 6. Recommandations finales
    print(f"\n🎉 SETUP TERMINÉ AVEC SUCCÈS!")
    print(f"\n📋 CONFIGURATION RECOMMANDÉE: {recommended_config}")
    print(f"\n🚀 COMMANDE D'ENTRAÎNEMENT:")
    print(f"   !python src/modeling/train.py --cfg_name {recommended_config}")

    print(f"\n📊 MONITORING EN TEMPS RÉEL:")
    print(f"   - Logs: /content/results/logs/")
    print(f"   - Figures: /content/results/figures/")
    print(f"   - Modèles: /content/results/models/")

    return True


if __name__ == "__main__":
    success = setup_colab_environment()
    if not success:
        print("\n❌ Setup échoué - vérifiez les erreurs ci-dessus")
    else:
        print("\n✅ Prêt pour l'entraînement !")
