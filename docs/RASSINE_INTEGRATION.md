# Amélioration de `create_soap_gpu_paper_dataset` avec Rassine

## 📖 Vue d'ensemble

La fonction `create_soap_gpu_paper_dataset` a été considérablement améliorée pour intégrer la normalisation spectrale avec **Rassine** (Rolling Alpha Shape for a Spectral Improved Normalisation Estimator), un algorithme de normalisation de continuum stellaire de pointe.

## 🌟 Nouvelles fonctionnalités

### 1. Normalisation avec Rassine
- **Algorithme robuste** : Rassine utilise une approche alpha-shape roulante pour estimer le continuum
- **Détection automatique** : Identification intelligente des maxima locaux
- **Sigma clipping** : Suppression automatique des pics cosmiques et aberrants
- **Interpolation adaptative** : Continuum estimé par interpolation cubique/linéaire

### 2. Configuration personnalisable
```python
rassine_config = {
    'smoothing_box': 6,                    # Fenêtre de lissage
    'smoothing_kernel': 'savgol',          # Type de kernel (savgol, gaussian, rectangular)
    'axes_stretching': 'auto_0.5',         # Étirement automatique des axes
    'vicinity_local_max': 7,               # Voisinage pour maxima locaux
    'min_radius': 'auto',                  # Rayon minimum du rolling pin
    'max_radius': 'auto',                  # Rayon maximum
    'speedup': 1.0,                        # Facteur d'accélération
    'mask_telluric': [[6275,6330], ...],   # Masques telluriques
    'feedback': False,                     # Interface graphique désactivée
}
```

### 3. Gestion d'erreurs robuste
- **Fallback automatique** : Si Rassine échoue, utilise une normalisation simple
- **Messages informatifs** : Suivi détaillé du processus de normalisation
- **Import conditionnel** : Gestion gracieuse de l'absence de Rassine

### 4. Métadonnées enrichies
```python
metadata = {
    'n_spectra': n_spectra,
    'wavemin': wavemin, 
    'wavemax': wavemax,
    'downscaling_factor': downscaling_factor,
    'use_rassine': use_rassine,
    'original_pixels': Npix,
    'downscaled_pixels': n_bins
}
```

## 🔧 Interface de la fonction

```python
def create_soap_gpu_paper_dataset(
    cube_filepath,              # Chemin vers le fichier HDF5 du cube
    spec_filepath,              # Chemin vers le fichier .npz du template
    output_filepath,            # Chemin de sortie .npz
    n_spectra,                    # Nombre de spectres à traiter
    wavemin, wavemax,           # Bornes spectrales en Angstrom
    downscaling_factor,         # Facteur de binning
    use_rassine=True,           # Activer Rassine (NOUVEAU)
    rassine_config=None,        # Configuration Rassine (NOUVEAU)
)
```

## 📋 Étapes du processus

### 1. Chargement des données
```python
# Chargement du template et de la grille spectrale
spec_data = np.load(spec_filepath)
template = spec_data["spec"]
wavegrid = spec_data["wavelength"]

# Chargement du cube de spectres
with h5py.File(cube_filepath, "r") as f:
    cube = f["spec_cube"][:n_spectra, mask]
```

### 2. Masquage spectral
```python
# Application du masque de longueur d'onde
mask = (wavegrid >= wavemin) & (wavegrid <= wavemax)
template_masked = template[mask]
wavegrid_masked = wavegrid[mask]
cube_masked = cube[:, mask]
```

### 3. Normalisation Rassine (NOUVEAU)
```python
if use_rassine:
    # Normalisation du template
    template_normalized = _normalize_spectrum_with_rassine(
        wavegrid_masked, template_masked, rassine_config
    )
    
    # Normalisation de chaque spectre
    for i in range(n_spectra):
        cube_normalized[i] = _normalize_spectrum_with_rassine(
            wavegrid_masked, cube_masked[i], rassine_config
        )
```

### 4. Algorithme Rassine simplifié
```python
def _normalize_spectrum_with_rassine(wave, flux, config):
    # 1. Préparation et tri des données
    grid = wave[wave.argsort()]
    spectrei = flux[wave.argsort()]
    
    # 2. Normalisation d'échelle
    len_x = grid.max() - grid.min()
    len_y = spectrei.max() - spectrei.min()
    normalisation = len_y / len_x
    spectre = spectrei / normalisation
    
    # 3. Sigma clipping itératif (suppression des pics)
    for iteration in range(3):
        # Rolling quantiles pour détecter les outliers
        maxi_roll = rolling_quantile(spectre, 0.99, window=100)
        Q3 = rolling_quantile(spectre, 0.75, window=5)
        Q2 = rolling_quantile(spectre, 0.50, window=5)
        
        # Masquage des pics aberrants
        mask = (spectre > Q3 + 1.5*IQ) & (spectre > maxi_roll)
        spectre[mask] = Q2[mask]
    
    # 4. Détection des maxima locaux
    peaks = find_peaks(spectre, height=percentile(spectre, 60))
    
    # 5. Sélection des points d'ancrage
    n_anchors = min(len(peaks), max(20, len(grid) // 50))
    wave_anchors = grid[peaks[:n_anchors]]
    flux_anchors = spectre[peaks[:n_anchors]] * normalisation
    
    # 6. Interpolation du continuum
    interpolator = interp1d(wave_anchors, flux_anchors, 
                           kind='cubic', fill_value='extrapolate')
    continuum = interpolator(grid)
    
    # 7. Normalisation finale
    normalized_spectrum = spectrei / continuum
    return np.clip(normalized_spectrum, 0.1, 2.0)
```

### 5. Downscaling
```python
# Binning par moyenne sur le facteur de downscaling
n_bins = Npix // downscaling_factor
wavegrid_ds = wavegrid_trim.mean(axis=1)
template_ds = template_trim.mean(axis=1)
cube_ds = cube_trim.mean(axis=2)
```

## 🎯 Avantages de Rassine

### Normalisation robuste
- **Estimation intelligente du continuum** : Utilise les maxima locaux plutôt qu'un simple ajustement polynomial
- **Suppression automatique des artefacts** : Sigma clipping itératif pour nettoyer les données
- **Adaptabilité** : S'ajuste automatiquement aux caractéristiques du spectre

### Configuration flexible
- **Paramètres automatiques** : Valeurs par défaut intelligentes avec mode 'auto'
- **Masques configurables** : Exclusion des raies telluriques et raies larges
- **Vitesse ajustable** : Paramètre `speedup` pour équilibrer vitesse/précision

### Robustesse
- **Fallback gracieux** : Continue sans Rassine en cas de problème
- **Gestion d'erreurs** : Messages informatifs et récupération automatique
- **Rétrocompatibilité** : `use_rassine=False` pour retrouver l'ancien comportement

## 📊 Exemples d'utilisation

### Utilisation basique avec Rassine
```python
create_soap_gpu_paper_dataset(
    cube_filepath="data/spec_cube.h5",
    spec_filepath="data/template.npz", 
    output_filepath="data/processed.npz",
    n_spectra=1000,
    wavemin=5000.0,
    wavemax=6500.0,
    downscaling_factor=4,
    use_rassine=True  # Activer Rassine
)
```

### Configuration personnalisée
```python
custom_config = {
    'smoothing_box': 8,           # Plus de lissage
    'speedup': 2.0,               # Plus rapide
    'mask_telluric': [            # Masques personnalisés
        [6275, 6330], [7600, 7700]
    ]
}

create_soap_gpu_paper_dataset(
    # ... paramètres de base ...
    use_rassine=True,
    rassine_config=custom_config
)
```

### Mode legacy (sans Rassine)
```python
create_soap_gpu_paper_dataset(
    # ... paramètres de base ...
    use_rassine=False  # Désactiver Rassine
)
```

## 🔍 Tests et validation

### Script de test
Le fichier `test_rassine_dataset.py` fournit :
- Tests automatisés avec et sans Rassine
- Comparaison des résultats
- Création de dataset de démonstration
- Validation des métadonnées

### Script d'exemple
Le fichier `example_rassine_dataset.py` montre :
- Utilisation basique et avancée
- Différentes configurations Rassine
- Comparaisons de performance
- Bonnes pratiques

## ⚠️ Considérations importantes

### Performance
- **Temps de traitement** : Rassine ajoute du temps de calcul (~20-50% selon la configuration)
- **Mémoire** : Utilisation temporaire de fichiers pour Rassine
- **Paramètre speedup** : Augmenter pour accélérer sur de gros datasets

### Qualité
- **Spectres bruités** : Rassine excelle sur les données réelles bruitées
- **Spectres synthétiques** : Peut être excessif pour des données parfaites
- **Validation** : Toujours vérifier visuellement les premiers résultats

### Dépendances
- **Rassine** : Doit être présent dans `Rassine_public/`
- **Pandas** : Requis pour les rolling quantiles
- **Scipy** : Pour l'interpolation et détection de pics

## 🚀 Prochaines améliorations possibles

1. **Parallélisation** : Traitement multi-thread des spectres
2. **Cache** : Sauvegarde des paramètres Rassine optimaux
3. **Validation automatique** : Métriques de qualité de normalisation
4. **Interface graphique** : Preview des résultats de normalisation
5. **Formats d'entrée** : Support d'autres formats (FITS, ASCII)

## 📚 Références

- **Rassine** : Cretignier et al., 2020, A&A, 640, A42
- **Repository Rassine** : https://github.com/MichaelCretignier/Rassine_public
- **Documentation SOAP** : Pour le contexte du dataset SOAP-GPU
