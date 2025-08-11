#!/usr/bin/env python3
"""
Test rapide pour vérifier que les checkpoints se sauvegardent correctement.
"""

import os
import tempfile
import yaml
from src.modeling.train import main


def create_test_config():
    """Crée une config de test avec des epochs courts pour vérifier la sauvegarde."""
    config = {
        "M_aug": 1,
        "autocast_enabled": False,
        "batch_size": 4,  # Très petit batch pour test rapide
        "checkpoint_every": 10,  # Sauvegarde tous les 10 epochs
        "csv_save_every": 0,  # Pas de CSV pour le test
        "cycle_length": 100,
        "data_dtype": "float32",
        "dataset_filepath": "data/npz_datasets/soapgpu_ns120_5000-5010_dx4_sm3_rassine_noise_p53_k0p1_phi0.npz",
        "extrapolate": "linear",
        "grad_scaler_enabled": False,
        "interpolate": "linear",
        "k_reg_init": 0.0,
        "latent_dim": 2,  # Plus petit pour test rapide
        "model_dtype": "float32",
        "out_dtype": "float32",
        "output_root_dir": "test_experiments",
        "experiment_name": "test_checkpoint_fix",
        "plot_every": 0,  # Pas de plots pour le test
        "phases": [
            {
                "name": "test_phase",
                "n_epochs": 5,  # Très court pour test
                "b_obs_trainable": False,
                "b_rest_trainable": False,
                "optimizer": "torch.optim.Adam",
                "optimizer_kwargs": {"lr": 0.001, "weight_decay": 0.0},
                "plot_rv_every": 0,
                "plot_activity_every": 0,
                "plot_spectra_every": 0,
            }
        ],
        "save_losses_csv": False,
        "shuffle": True,
        "sigma_c": 0.01,
        "sigma_v": 10.0,
        "sigma_y": 0.01,
        "use_mixed_precision": False,
        "vmax": 3,
        "vmin": -3,
    }
    return config


def test_checkpoint_saving():
    """Test que les checkpoints se sauvent correctement."""
    print("🧪 Test de sauvegarde des checkpoints...")

    # Créer une config de test
    config = create_test_config()

    # Créer un fichier config temporaire
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
        config_path = f.name

    try:
        # Lancer l'entraînement de test
        print(f"📋 Config temporaire: {config_path}")
        main(config_path=config_path)

        # Vérifier que les modèles ont été créés
        exp_dir = "test_experiments/test_checkpoint_fix"
        models_dir = os.path.join(exp_dir, "models")

        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith(".pth")]
            print(f"✅ Modèles trouvés dans {models_dir}:")
            for model_file in model_files:
                print(f"   - {model_file}")

            if "aestra_final.pth" in model_files:
                print("🎯 aestra_final.pth créé avec succès !")
            else:
                print("❌ aestra_final.pth manquant")
        else:
            print(f"❌ Dossier {models_dir} introuvable")

    except Exception as e:
        print(f"❌ Erreur pendant le test: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Nettoyer le fichier config temporaire
        try:
            os.unlink(config_path)
        except Exception:
            pass


if __name__ == "__main__":
    test_checkpoint_saving()
