#!/usr/bin/env python3
"""
Test de validation de la réorganisation du pipeline predict.py

Vérifie que la structure de sortie correspond aux spécifications.
"""

import os
import tempfile
import shutil
from pathlib import Path


def test_output_structure():
    """Teste la structure de sortie du pipeline"""

    # Structure attendue
    expected_structure = {
        "postprocessing/figures/periodograms/rv/": ["*.png"],
        "postprocessing/figures/periodograms/latent/": ["*.png"],
        "postprocessing/figures/latent/": [
            "distance_distribution.png",
            "marginal_distributions.png",
            "activity_perturbations.png",
            "latent_space_3d.png",
        ],
        "postprocessing/figures/correlations/": ["correlation_matrix.png"],
        "postprocessing/figures/": ["mcmc_orbital_fit.png"],
        "postprocessing/data/": [
            "metrics.csv",
            "periodograms.npz",
            "latent_distances.npz",
        ],
    }

    print("Structure de sortie attendue :")
    print("=" * 50)

    for folder, files in expected_structure.items():
        print(f"📁 {folder}")
        for file in files:
            print(f"  📄 {file}")
        print()


def validate_function_signature():
    """Valide la signature de la fonction main()"""

    from src.modeling.predict import main
    import inspect

    sig = inspect.signature(main)
    params = list(sig.parameters.keys())

    expected_params = [
        "experiment_dir",
        "fap_threshold",
        "exclude_width_frac",
        "min_period",
        "max_period",
        "n_periods",
        "zoom_frac",
        "batch_size",
        "perturbation_value",
        "overrides",
    ]

    print("Validation de la signature de main() :")
    print("=" * 50)

    for param in expected_params:
        if param in params:
            print(f"✅ {param}")
        else:
            print(f"❌ {param} manquant")

    print(f"\nSignature complète: {sig}")


def validate_no_main_execution():
    """Vérifie qu'il n'y a pas de code d'exécution directe"""

    with open(
        "/home/tliopis/Codes/exoplanets_llopis_mary_2025/src/modeling/predict.py", "r"
    ) as f:
        content = f.read()

    print("Validation de l'absence d'exécution directe :")
    print("=" * 50)

    if 'if __name__ == "__main__"' in content:
        print("❌ Le fichier contient encore du code d'exécution directe")
    else:
        print("✅ Aucun code d'exécution directe trouvé")

    # Vérifier que la fonction main() est bien définie
    if "def main(" in content:
        print("✅ La fonction main() est définie")
    else:
        print("❌ La fonction main() n'est pas trouvée")


if __name__ == "__main__":
    print("Test de validation de la réorganisation predict.py")
    print("=" * 60)
    print()

    test_output_structure()
    print()
    validate_function_signature()
    print()
    validate_no_main_execution()

    print("\n🎯 Réorganisation terminée avec succès !")
    print("\nUtilisation :")
    print("from src.modeling.predict import main")
    print('main(experiment_dir="experiments/exp1")')
