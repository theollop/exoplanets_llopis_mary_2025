"""
Utilitaires pour la gestion des raies spectrales et masques.
"""

import numpy as np
from typing import List, Tuple
import os


def load_g2_mask(
    mask_filepath: str = "data/rv_datachallenge/masks/G2_mask.txt",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Charge le masque G2 contenant les longueurs d'onde et poids des raies spectrales.

    Args:
        mask_filepath: Chemin vers le fichier masque G2

    Returns:
        tuple: (wavelengths, weights) - longueurs d'onde et poids associés
    """
    if not os.path.exists(mask_filepath):
        raise FileNotFoundError(f"Mask file not found: {mask_filepath}")

    data = np.loadtxt(mask_filepath)
    wavelengths = data[:, 0]  # Première colonne: longueurs d'onde
    weights = data[:, 1]  # Deuxième colonne: poids

    return wavelengths, weights


def find_spectral_lines(
    wavelengths: np.ndarray,
    weights: np.ndarray,
    weight_threshold: float = 0.5,
    min_separation: float = 2.0,
) -> List[float]:
    """
    Identifie les raies spectrales significatives à partir du masque.

    Args:
        wavelengths: Longueurs d'onde du masque
        weights: Poids associés
        weight_threshold: Seuil minimum de poids pour considérer une raie
        min_separation: Séparation minimale entre raies (en Å)

    Returns:
        List des longueurs d'onde des raies principales
    """
    # Filtrer les pixels avec un poids élevé
    strong_mask = weights >= weight_threshold
    strong_wavelengths = wavelengths[strong_mask]
    strong_weights = weights[strong_mask]

    if len(strong_wavelengths) == 0:
        return []

    # Trier par poids décroissant
    sorted_indices = np.argsort(strong_weights)[::-1]
    sorted_wavelengths = strong_wavelengths[sorted_indices]

    # Sélectionner les raies en évitant les doublons proches
    selected_lines = []
    for wave in sorted_wavelengths:
        # Vérifier si la raie est suffisamment éloignée des raies déjà sélectionnées
        if not selected_lines or all(
            abs(wave - existing) >= min_separation for existing in selected_lines
        ):
            selected_lines.append(wave)

    return sorted(selected_lines)


def select_lines_in_range(
    spectral_lines: List[float], wave_min: float, wave_max: float, max_lines: int = 6
) -> List[float]:
    """
    Sélectionne les raies spectrales dans une plage de longueurs d'onde donnée.

    Args:
        spectral_lines: Liste des longueurs d'onde des raies
        wave_min: Longueur d'onde minimale
        wave_max: Longueur d'onde maximale
        max_lines: Nombre maximum de raies à retourner

    Returns:
        Liste des longueurs d'onde des raies dans la plage
    """
    lines_in_range = [line for line in spectral_lines if wave_min <= line <= wave_max]
    return lines_in_range[:max_lines]


def get_doppler_shift_range(
    rest_wavelength: float, velocity_range: float = 10.0
) -> Tuple[float, float]:
    """
    Calcule la plage de longueurs d'onde correspondant à un décalage Doppler.

    Args:
        rest_wavelength: Longueur d'onde au repos (Å)
        velocity_range: Plage de vitesse à couvrir (±km/s)

    Returns:
        tuple: (wave_min, wave_max) pour couvrir la plage de vitesses
    """
    c = 299792.458  # Vitesse de la lumière en km/s

    # Décalage Doppler: Δλ/λ = v/c
    relative_shift = velocity_range / c
    delta_lambda = rest_wavelength * relative_shift

    wave_min = rest_wavelength - delta_lambda
    wave_max = rest_wavelength + delta_lambda

    return wave_min, wave_max


def find_best_lines_for_doppler(
    mask_filepath: str, wave_range: Tuple[float, float], max_lines: int = 6
) -> List[float]:
    """
    Trouve les meilleures raies pour détecter les décalages Doppler dans une plage donnée.

    Args:
        mask_filepath: Chemin vers le fichier masque G2
        wave_range: Tuple (wave_min, wave_max) de la plage spectrale
        max_lines: Nombre maximum de raies à retourner

    Returns:
        Liste des longueurs d'onde des meilleures raies
    """
    wavelengths, weights = load_g2_mask(mask_filepath)

    # Trouver toutes les raies significatives
    all_lines = find_spectral_lines(
        wavelengths, weights, weight_threshold=0.7, min_separation=1.5
    )

    # Sélectionner celles dans la plage d'intérêt
    selected_lines = select_lines_in_range(
        all_lines, wave_range[0], wave_range[1], max_lines
    )

    return selected_lines


if __name__ == "__main__":
    # Test du module
    print("🔍 Test du module spectral_lines")

    try:
        # Charger le masque
        wavelengths, weights = load_g2_mask()
        print(f"✅ Masque G2 chargé: {len(wavelengths)} points spectraux")
        print(f"   Plage: {wavelengths.min():.1f} - {wavelengths.max():.1f} Å")

        # Trouver les raies principales
        main_lines = find_spectral_lines(wavelengths, weights, weight_threshold=0.8)
        print(f"✅ {len(main_lines)} raies principales trouvées (seuil=0.8)")

        # Exemple pour la plage 5000-5020 Å
        test_range = (5000.0, 5020.0)
        lines_in_range = select_lines_in_range(main_lines, test_range[0], test_range[1])
        print(f"✅ {len(lines_in_range)} raies dans {test_range[0]}-{test_range[1]} Å:")
        for line in lines_in_range:
            print(f"   {line:.2f} Å")

    except Exception as e:
        print(f"❌ Erreur: {e}")
