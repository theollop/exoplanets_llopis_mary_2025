#!/bin/bash

# Script de configuration de l'environnement virtuel pour le projet exoplanets

echo "🚀 Configuration de l'environnement virtuel pour le projet exoplanets..."

# Création de l'environnement virtuel
echo "📦 Création de l'environnement virtuel..."
python3 -m venv venv

# Activation de l'environnement virtuel
echo "🔧 Activation de l'environnement virtuel..."
source venv/bin/activate

# Mise à jour de pip
echo "⬆️  Mise à jour de pip..."
pip install --upgrade pip

# Installation de PyTorch avec CUDA 12.0
echo "🔥 Installation de PyTorch avec CUDA 12.0..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu120

# Installation des autres dépendances
echo "📚 Installation des autres dépendances depuis requirements.txt..."
pip install -r requirements.txt

# Installation de torchcubicspline depuis GitHub (optionnel)
echo "🌀 Installation de torchcubicspline depuis GitHub..."
echo "   (Optionnel - seulement nécessaire pour l'interpolation par spline cubique)"
read -p "Voulez-vous installer torchcubicspline ? (y/N): " install_cubicspline
if [[ $install_cubicspline =~ ^[Yy]$ ]]; then
    pip install git+https://github.com/patrick-kidger/torchcubicspline.git
    echo "✅ torchcubicspline installé depuis GitHub"
else
    echo "⏭️  torchcubicspline ignoré (vous pouvez l'installer plus tard si nécessaire)"
fi

# Installation du projet en mode développement pour permettre les imports
echo "🔗 Installation du projet en mode développement..."
pip install -e .

echo "✅ Configuration terminée !"
echo ""
echo "Pour activer l'environnement virtuel dans le futur, utilisez:"
echo "source venv/bin/activate"
echo ""
echo "Pour tester que tout fonctionne, vous pouvez essayer:"
echo "cd src/modeling && python -c \"from ..interpolate import shift_spectra_linear; print('Import réussi!')\"" 
