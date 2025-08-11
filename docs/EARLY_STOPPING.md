# Early Stopping dans AESTRA

## Description

L'Early Stopping est un mécanisme de régularisation qui arrête automatiquement l'entraînement lorsque la métrique surveillée ne s'améliore plus pendant un nombre spécifié d'epochs (patience). Cela permet d'éviter le surapprentissage et de gagner du temps de calcul.

## Configuration

L'Early Stopping se configure dans le fichier YAML de configuration, au niveau de chaque phase d'entraînement :

```yaml
phases:
  - name: rvonly
    n_epochs: 200
    optimizer: "torch.optim.Adam"
    optimizer_kwargs:
      lr: 0.0001
    early_stopping:
      patience: 20           # Nombre d'epochs sans amélioration avant arrêt
      min_delta: 1e-6       # Amélioration minimale considérée comme significative
      metric: "total"       # Métrique à surveiller ("total", "rv", "fid", "c", "reg")
      mode: "min"           # "min" pour minimiser, "max" pour maximiser
      restore_best_weights: true  # Restaurer les meilleurs poids à l'arrêt
```

## Paramètres disponibles

### `patience` (int, défaut: 10)
Nombre d'epochs consécutives sans amélioration de la métrique avant d'arrêter l'entraînement.

### `min_delta` (float, défaut: 0.0)
Amélioration minimale requise pour être considérée comme significative. Par exemple, avec `min_delta: 1e-6`, une amélioration de 5e-7 ne sera pas considérée comme une vraie amélioration.

### `metric` (str, défaut: "total")
Métrique à surveiller pour l'Early Stopping. Options disponibles :
- `"total"` : Loss totale (somme de toutes les losses)
- `"rv"` : Loss des vitesses radiales
- `"fid"` : Loss de fidélité spectrale  
- `"c"` : Loss des coefficients de contamination
- `"reg"` : Loss de régularisation

### `mode` (str, défaut: "min")
Direction d'optimisation :
- `"min"` : Arrêter quand la métrique ne diminue plus (pour les losses)
- `"max"` : Arrêter quand la métrique n'augmente plus (pour les métriques de performance)

### `restore_best_weights` (bool, défaut: true)
Si `true`, les poids du modèle sont restaurés à leur meilleure valeur (epoch avec la meilleure métrique) quand l'Early Stopping se déclenche.

## Exemples de configuration

### Early Stopping sur la loss totale (cas général)
```yaml
early_stopping:
  patience: 15
  min_delta: 1e-6
  metric: "total"
  mode: "min"
  restore_best_weights: true
```

### Early Stopping spécialisé sur les vitesses radiales
```yaml
early_stopping:
  patience: 25
  min_delta: 1e-7
  metric: "rv"
  mode: "min"
  restore_best_weights: true
```

### Early Stopping conservateur (patience élevée)
```yaml
early_stopping:
  patience: 50
  min_delta: 0.0
  metric: "total"
  mode: "min"
  restore_best_weights: false
```

## Désactivation

Pour désactiver l'Early Stopping sur une phase, il suffit de ne pas inclure la section `early_stopping` dans la configuration de la phase.

## Logs et monitoring

Quand l'Early Stopping se déclenche, les informations suivantes sont affichées :
- Epoch d'arrêt
- Nombre d'epochs sans amélioration
- Meilleure valeur de la métrique et son epoch
- Confirmation de la restauration des poids (si activée)

Exemple de sortie :
```
⏹️  Early Stopping déclenché à l'epoch 127
📈 Pas d'amélioration depuis 20 epochs
🏆 Meilleure valeur: 2.451e-04 à l'epoch 107
🔄 Meilleurs poids restaurés (epoch 107)
```

## Bonnes pratiques

1. **Patience** : Commencer avec 15-25 epochs pour les phases courtes, 50+ pour les phases longues
2. **min_delta** : Utiliser des valeurs petites (1e-6 à 1e-8) pour éviter les arrêts prématurés dus au bruit
3. **Métrique** : Surveiller la loss la plus pertinente pour votre objectif (`"rv"` pour l'accuracy RV, `"total"` pour la convergence générale)
4. **restore_best_weights** : Généralement recommandé sur `true` pour éviter de garder des poids dégradés
