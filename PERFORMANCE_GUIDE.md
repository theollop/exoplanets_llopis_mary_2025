# Guide de Performance - create_soap_gpu_paper_dataset

## 🚨 Problèmes identifiés et Solutions

La fonction `create_soap_gpu_paper_dataset` peut causer des crashes système à cause de :

### 1. **Consommation excessive de mémoire**
- **Problème** : Chargement de gros cubes de données sans contrôle de mémoire
- **Solution** : Chargement par chunks et traitement par batches

### 2. **Saturation de la mémoire GPU**
- **Problème** : Transfert de gros tenseurs vers GPU sans vérification
- **Solution** : Gestion automatique de la mémoire GPU avec fallback CPU

### 3. **Boucles intensives sans optimisation**
- **Problème** : Normalisation Rassine sur tous les spectres en séquence
- **Solution** : Traitement par batches avec nettoyage mémoire périodique

## 🛡️ Version Sécurisée Recommandée

### Utilisation automatique (RECOMMANDÉE)
```python
from src.dataset import create_soap_gpu_paper_dataset_safe

# Optimisation automatique selon votre système
create_soap_gpu_paper_dataset_safe(
    cube_filepath="data/soap_gpu_paper/spec_cube_tot.h5",
    spec_filepath="data/soap_gpu_paper/spec_master.npz",
    output_filepath="output.npz",
    n_spectra=1000,
    wavemin=5000,
    wavemax=6000,
    downscaling_factor=2,
    use_rassine=True,
    add_photon_noise=False,
    amplitudes=[10],
    periods=[100],
    auto_optimize=True  # 🔧 Active l'optimisation automatique
)
```

### Configuration manuelle (pour utilisateurs avancés)
```python
from src.dataset import create_soap_gpu_paper_dataset

# Si vous connaissez les limites de votre système
create_soap_gpu_paper_dataset(
    cube_filepath="data/soap_gpu_paper/spec_cube_tot.h5",
    spec_filepath="data/soap_gpu_paper/spec_master.npz",
    output_filepath="output.npz",
    n_spectra=1000,
    wavemin=5000,
    wavemax=6000,
    downscaling_factor=2,
    use_rassine=True,
    add_photon_noise=False,
    amplitudes=[10],
    periods=[100],
    batch_size=50,           # ⚙️ Taille des batches
    max_gpu_memory_gb=4,     # 🎮 Limite GPU en GB
    force_cpu=False          # 💻 Force CPU si True
)
```

## 📊 Paramètres recommandés selon votre système

### Système avec peu de RAM (< 8 GB)
```python
batch_size=25
max_gpu_memory_gb=1
force_cpu=True  # Recommandé
```

### Système standard (8-16 GB RAM)
```python
batch_size=50
max_gpu_memory_gb=4
force_cpu=False
```

### Système haute performance (> 16 GB RAM)
```python
batch_size=100
max_gpu_memory_gb=8
force_cpu=False
```

## ⚠️ Signaux d'alerte

Si vous voyez ces messages, **ARRÊTEZ** et réduisez les paramètres :

- `⚠️ Mémoire système faible`
- `⚠️ Données trop volumineuses`
- `⚠️ Mémoire GPU insuffisante`

## 🔧 Nouvelles fonctionnalités anti-crash

1. **Vérification automatique des ressources système**
2. **Chargement par chunks pour gros datasets**
3. **Traitement par batches avec nettoyage mémoire**
4. **Gestion d'erreurs GPU avec fallback CPU**
5. **Nettoyage automatique de la mémoire**
6. **Optimisations automatiques basées sur le matériel**

## 💡 Conseils d'utilisation

1. **Toujours utiliser `create_soap_gpu_paper_dataset_safe()` en premier**
2. **Commencer avec de petits datasets pour tester**
3. **Monitorer l'utilisation mémoire avec `htop` ou `nvidia-smi`**
4. **Fermer les autres applications avant traitement**
5. **Utiliser un système de swapping suffisant**

## 🐛 En cas de problème

Si le processus crash malgré les optimisations :

1. Réduisez `n_spectra` (ex: 500 au lieu de 1000)
2. Augmentez `downscaling_factor` (ex: 4 au lieu de 2)
3. Désactivez Rassine (`use_rassine=False`)
4. Forcez l'utilisation CPU (`force_cpu=True`)
5. Réduisez la gamme spectrale (`wavemax - wavemin`)

## ✅ Exemple minimal sécurisé

```python
# Dataset minimal pour test
create_soap_gpu_paper_dataset_safe(
    cube_filepath="data/soap_gpu_paper/spec_cube_tot.h5",
    spec_filepath="data/soap_gpu_paper/spec_master.npz",
    output_filepath="test_output.npz",
    n_spectra=100,           # 🔽 Réduit pour test
    wavemin=5000,
    wavemax=5500,            # 🔽 Gamme réduite
    downscaling_factor=4,    # 🔽 Plus de downscaling
    use_rassine=False,       # 🔽 Désactivé pour simplicité
    add_photon_noise=False,
    amplitudes=None,         # 🔽 Pas d'injection
    periods=None,
    auto_optimize=True
)
```
