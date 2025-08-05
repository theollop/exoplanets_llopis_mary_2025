# 🚀 Guide d'utilisation pour Google Colab

## 📋 Instructions rapides

### 1. Setup initial dans Colab

```python
# Étape 1: Monter Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Étape 2: Cloner le repository
!git clone https://github.com/theollop/exoplanets_llopis_mary_2025

# Étape 3: Aller dans le dossier
%cd /content/exoplanets_llopis_mary_2025/

# Étape 4: Installer les dépendances
!pip install -e .
!pip install --no-deps git+https://github.com/patrick-kidger/torchcubicspline.git

# Étape 5: Setup de l'environnement
!python setup_colab.py
```

### 2. Entraînement

```python
# Option A: Configuration de test (rapide, données du repository)
!python src/modeling/train.py --cfg_name colab_test_config

# Option B: Configuration complète (données sur Google Drive)
!python src/modeling/train.py --cfg_name colab_config
```

## 🔧 Résolution des problèmes

### Problème: Script s'arrête après "Config loaded"

**Diagnostic:**
```python
!python setup_colab.py
```

**Solutions courantes:**

1. **Données manquantes:**
   - Vérifiez que les données sont dans `/content/drive/MyDrive/PFE/data/`
   - Ou utilisez `colab_test_config` pour les données du repository

2. **Mémoire insuffisante:**
   ```python
   # Réduire la taille du batch
   !sed -i 's/batch_size: 16/batch_size: 8/' src/modeling/configs/colab_config.yaml
   ```

3. **Timeout Google Drive:**
   ```python
   # Copier les données localement (plus rapide)
   !cp -r /content/drive/MyDrive/PFE/data /content/
   # Puis modifier la config pour utiliser /content/data
   ```

### Problème: CUDA out of memory

```python
# Solution 1: Réduire le batch size
!sed -i 's/batch_size: 16/batch_size: 4/' src/modeling/configs/colab_config.yaml

# Solution 2: Redémarrer le runtime
# Runtime > Restart runtime

# Solution 3: Utiliser CPU seulement
!python src/modeling/train.py --cfg_name colab_config --device cpu
```

### Problème: Fichiers non trouvés

```python
# Vérifier l'arborescence
!find /content -name "*.npy" -type f 2>/dev/null | head -5
!find /content/drive -name "*.npy" -type f 2>/dev/null | head -5

# Lister le contenu des dossiers
!ls -la /content/exoplanets_llopis_mary_2025/data/rv_datachallenge/
```

## 📊 Monitoring de l'entraînement

### Voir les résultats en temps réel

```python
# Logs d'entraînement
!tail -f /content/results/logs/*.csv

# Visualiser les figures
from IPython.display import Image, display
import glob

# Afficher la dernière figure de losses
figures = sorted(glob.glob("/content/results/figures/*losses*.png"))
if figures:
    display(Image(figures[-1]))

# Afficher les spectres
spectra = sorted(glob.glob("/content/results/spectra/*aestra*.png"))
if spectra:
    display(Image(spectra[-1]))
```

### Sauvegarder les résultats sur Google Drive

```python
# Copier tous les résultats vers Google Drive
!mkdir -p "/content/drive/MyDrive/PFE/results"
!cp -r /content/results/* "/content/drive/MyDrive/PFE/results/"

print("✅ Résultats sauvegardés sur Google Drive!")
```

## ⚡ Configurations optimisées

### Configuration rapide (test)
- **Fichier:** `colab_test_config.yaml`
- **Données:** Repository cloné
- **Durée:** ~5-10 minutes
- **Usage:** Tests et debugging

### Configuration complète (production)
- **Fichier:** `colab_config.yaml` 
- **Données:** Google Drive
- **Durée:** Plusieurs heures
- **Usage:** Entraînement final

## 🐛 Debug avancé

### Script de diagnostic complet

```python
# Diagnostic automatique
!python -c "
import os, torch, yaml
print('🔍 DIAGNOSTIC COLAB')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Drive monté: {os.path.exists(\"/content/drive\")}')
print(f'Repo cloné: {os.path.exists(\"/content/exoplanets_llopis_mary_2025\")}')
print(f'Données repo: {os.path.exists(\"/content/exoplanets_llopis_mary_2025/data\")}')
print(f'Données drive: {os.path.exists(\"/content/drive/MyDrive/PFE/data\")}')
"
```

### Logs détaillés

```python
# Activer le mode debug
import logging
logging.basicConfig(level=logging.DEBUG)

# Lancer l'entraînement avec logs détaillés
!python src/modeling/train.py --cfg_name colab_test_config 2>&1 | tee training.log
```

## 📈 Optimisations performances

### Pour GPU Tesla T4 (Colab gratuit)
```yaml
batch_size: 8
n_specs: 500  # Réduire le dataset
```

### Pour GPU Premium (Colab Pro)
```yaml
batch_size: 16
n_specs: null  # Dataset complet
```

### Mode économie de mémoire
```yaml
batch_size: 4
plot_every: 50  # Moins de plots
csv_save_every: 25  # Moins de sauvegardes
```
