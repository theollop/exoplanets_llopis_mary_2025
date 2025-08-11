#!/usr/bin/env python3
"""
Démonstration du pipeline de prédiction AESTRA

Ce script montre comment utiliser la fonction main() du module predict.py
pour lancer l'analyse post-traitement d'une expérience AESTRA.
"""

from src.modeling.predict import main


def demo_simple_prediction():
    """Exemple simple : utilisation avec paramètres par défaut"""
    print("=== Démonstration 1: Prédiction simple ===")
    main(experiment_dir="experiments/exp1")


def demo_custom_parameters():
    """Exemple avec paramètres personnalisés"""
    print("=== Démonstration 2: Prédiction avec paramètres personnalisés ===")
    main(
        experiment_dir="experiments/exp1",
        batch_size=128,
        min_period=5.0,
        max_period=150.0,
        fap_threshold=0.005,
        n_periods=3000,
    )


def demo_multiple_experiments():
    """Exemple pour traiter plusieurs expériences"""
    print("=== Démonstration 3: Traitement de plusieurs expériences ===")

    experiments = ["experiments/exp1", "experiments/exp2", "experiments/exp3"]

    for exp_dir in experiments:
        print(f"\n--- Traitement de {exp_dir} ---")
        try:
            main(experiment_dir=exp_dir, batch_size=64, min_period=2.0)
            print(f"✅ {exp_dir} traité avec succès")
        except Exception as e:
            print(f"❌ Erreur pour {exp_dir}: {e}")


if __name__ == "__main__":
    # Décommentez la fonction que vous voulez tester

    # demo_simple_prediction()
    # demo_custom_parameters()
    # demo_multiple_experiments()

    print("Décommentez une des fonctions demo_* pour tester le pipeline")
