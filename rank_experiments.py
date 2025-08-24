#!/usr/bin/env python3
"""
Script pour classer les expériences selon leurs performances de détection de planètes.
Analyse toutes les métriques des périodogrammes dans experiments/*/postprocessing/data/metrics.csv

Métriques principales:
- power_ratio: rapport de puissance à P_inj vs maximum ailleurs (plus grand = meilleur)
- fap_at_Pinj: probabilité de fausse alarme à P_inj (plus petit = meilleur) 
- delta_P: écart absolu entre P_détecté et P_injecté (plus petit = meilleur)
- n_sig_peaks_outside: nombre de pics significatifs hors fenêtre planète (plus petit = meilleur)
"""

import os
import glob
import csv
import numpy as np
import pandas as pd
from collections import defaultdict


def find_all_metrics_files(experiments_root="experiments"):
    """Trouve tous les fichiers metrics.csv dans les expériences."""
    pattern = os.path.join(experiments_root, "*", "postprocessing", "data", "metrics.csv")
    return glob.glob(pattern)


def load_experiment_metrics(csv_file):
    """Charge les métriques d'une expérience depuis son fichier CSV."""
    exp_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(csv_file))))
    
    try:
        df = pd.read_csv(csv_file)
        # Filtrer les lignes de périodogrammes uniquement
        periodo_df = df[df['row_type'] == 'periodogram']

        # Chercher le fichier periodograms.npz dans le même dossier
        periodograms_path = os.path.join(os.path.dirname(csv_file), 'periodograms.npz')
        if not os.path.exists(periodograms_path):
            periodograms_path = None

        return exp_name, periodo_df, periodograms_path
    except Exception as e:
        print(f"Erreur en lisant {csv_file}: {e}")
        return exp_name, pd.DataFrame(), None


def extract_best_metrics_per_series(df, periodograms_path=None):
    """Extrait les meilleures métriques par série RV pour une expérience."""
    results = {}
    # Charger les periodograms si fournis
    periodograms = None
    if periodograms_path is not None:
        try:
            periodograms = np.load(periodograms_path)
        except Exception:
            periodograms = None

    # Grouper par série (v_apparent, v_correct, etc.)
    for series_name in df['series'].unique():
        if pd.isna(series_name):
            continue
        # Garder uniquement les périodogrammes de type v_encode et v_correct
        if series_name not in ('v_encode', 'v_correct'):
            continue

        series_df = df[df['series'] == series_name]
        
        # Métriques de détection pour cette série
        metrics = {}
        
        # Power ratio: plus grand = meilleur
        power_ratio_rows = series_df[series_df['metric'] == 'power_ratio']
        if not power_ratio_rows.empty:
            metrics['power_ratio'] = power_ratio_rows['value'].max()
        
        # FAP at P_inj: plus petit = meilleur
        fap_rows = series_df[series_df['metric'] == 'fap_at_Pinj']
        if not fap_rows.empty:
            valid_faps = fap_rows['value'].dropna()
            if not valid_faps.empty:
                metrics['fap_at_Pinj'] = valid_faps.min()
        
        # Delta P: plus petit = meilleur
        delta_p_rows = series_df[series_df['metric'] == 'delta_P']
        if not delta_p_rows.empty:
            valid_deltas = delta_p_rows['value'].dropna()
            if not valid_deltas.empty:
                metrics['delta_P'] = valid_deltas.min()
        
        # Nombre de pics significatifs hors fenêtre: plus petit = meilleur
        n_peaks_rows = series_df[series_df['metric'] == 'n_sig_peaks_outside']
        if not n_peaks_rows.empty:
            metrics['n_sig_peaks_outside'] = n_peaks_rows['value'].min()
        
        # P_detected: période détectée (si disponible)
        p_det_rows = series_df[series_df['metric'] == 'P_detected']
        P_detected = None
        if not p_det_rows.empty:
            valid_p_det = p_det_rows['value'].dropna()
            if not valid_p_det.empty:
                P_detected = float(valid_p_det.iloc[0])  # Prendre la première valeur valide
                metrics['P_detected'] = P_detected

        # P_inj: période injectée (présente sur chaque ligne periodogram dans la colonne P_inj)
        P_inj = None
        try:
            p_inj_vals = series_df['P_inj'].dropna().unique()
            if len(p_inj_vals) > 0:
                P_inj = float(p_inj_vals[0])
                metrics['P_inj'] = P_inj
        except Exception:
            P_inj = None

        # Si periodograms disponibles, calculer la puissance/amplitude au pic détecté
        if periodograms is not None:
            # Chercher clés robustement (strip pour éviter espaces/retours ligne dans les clés)
            def find_keys_for_series(npz, series):
                pkey = None
                powkey = None
                for k in npz.files:
                    kn = k.strip()
                    if kn == f"{series}_periods" or kn.startswith(f"{series}_periods"):
                        pkey = k
                    if kn == f"{series}_power" or kn.startswith(f"{series}_power"):
                        powkey = k
                return pkey, powkey

            pkey, powkey = find_keys_for_series(periodograms, series_name)
            if pkey is not None and powkey is not None:
                try:
                    periods = periodograms[pkey]
                    power = periodograms[powkey]

                    # puissance à P_inj si dispo
                    if P_inj is not None:
                        idx_inj = int(np.argmin(np.abs(periods - P_inj)))
                        amp_inj = float(power[idx_inj])
                        metrics['peak_power_at_Pinj'] = amp_inj
                        metrics['peak_power_period'] = float(periods[idx_inj])

                    # puissance à P_detected si dispo
                    if P_detected is not None:
                        idx = int(np.argmin(np.abs(periods - P_detected)))
                        amp = float(power[idx])
                        metrics['peak_power_at_Pdet'] = amp
                        metrics['peak_power_period_detected'] = float(periods[idx])

                    # fallback: utiliser le pic global si aucune des deux périodes n'est disponible
                    if ('peak_power_at_Pinj' not in metrics) and ('peak_power_at_Pdet' not in metrics):
                        idx_max = int(np.argmax(power))
                        metrics['peak_power'] = float(power[idx_max])
                        metrics['peak_power_period'] = float(periods[idx_max])
                except Exception:
                    pass

        if metrics:
            results[series_name] = metrics
    
    return results


def rank_experiments_by_metric(all_results, metric_name, higher_is_better=True):
    """Classe toutes les expériences selon une métrique donnée."""
    rankings = []
    
    for exp_name, series_results in all_results.items():
        for series_name, metrics in series_results.items():
            if metric_name in metrics:
                value = metrics[metric_name]
                if not pd.isna(value):
                    rankings.append({
                        'experiment': exp_name,
                        'series': series_name,
                        'metric': metric_name,
                        'value': float(value)
                    })
    
    # Trier selon la direction appropriée
    rankings.sort(key=lambda x: x['value'], reverse=higher_is_better)
    return rankings


def print_ranking(rankings, metric_name, top_k=10):
    """Affiche le classement pour une métrique."""
    print(f"\n{'='*60}")
    print(f"🏆 CLASSEMENT: {metric_name.upper()}")
    print(f"{'='*60}")
    
    if not rankings:
        print("Aucune donnée disponible pour cette métrique.")
        return
    
    print(f"{'Rang':<4} {'Expérience':<30} {'Série':<15} {'Valeur':<12}")
    print("-" * 62)
    
    for i, entry in enumerate(rankings[:top_k], 1):
        exp_name = entry['experiment']
        series_name = entry['series']
        value = entry['value']
        
        # Formatage adapté selon la métrique
        if metric_name in ['fap_at_Pinj']:
            value_str = f"{value:.2e}"
        elif metric_name in ['delta_P']:
            value_str = f"{value:.3f}"
        elif metric_name in ['power_ratio']:
            value_str = f"{value:.2f}"
        else:
            value_str = f"{value:.3g}"
        
        print(f"{i:<4} {exp_name:<30} {series_name:<15} {value_str:<12}")


def create_summary_table(all_results):
    """Crée un tableau de synthèse avec toutes les métriques."""
    summary_data = []
    
    for exp_name, series_results in all_results.items():
        for series_name, metrics in series_results.items():
            row = {
                'experiment': exp_name,
                'series': series_name,
                'power_ratio': metrics.get('power_ratio', np.nan),
                'fap_at_Pinj': metrics.get('fap_at_Pinj', np.nan),
                'delta_P': metrics.get('delta_P', np.nan),
                'n_sig_peaks_outside': metrics.get('n_sig_peaks_outside', np.nan),
                'P_detected': metrics.get('P_detected', np.nan),
                'peak_power_at_Pdet': metrics.get('peak_power_at_Pdet', np.nan),
                'peak_power_at_Pinj': metrics.get('peak_power_at_Pinj', np.nan),
                'peak_power': metrics.get('peak_power', np.nan),
                'peak_power_period': metrics.get('peak_power_period', np.nan)
            }
            summary_data.append(row)
    
    return pd.DataFrame(summary_data)


def main():
    """Fonction principale."""
    print("🔍 Recherche des fichiers de métriques...")
    
    # Trouver tous les fichiers metrics.csv
    metrics_files = find_all_metrics_files()
    
    if not metrics_files:
        print("❌ Aucun fichier metrics.csv trouvé dans experiments/*/postprocessing/data/")
        return
    
    print(f"📊 Trouvé {len(metrics_files)} fichiers de métriques")
    
    # Charger toutes les métriques
    all_results = {}
    
    for csv_file in metrics_files:
        exp_name, df, periodograms_path = load_experiment_metrics(csv_file)
        if not df.empty:
            metrics = extract_best_metrics_per_series(df, periodograms_path)
            if metrics:
                all_results[exp_name] = metrics
                print(f"✅ {exp_name}: {len(metrics)} séries RV analysées")
            else:
                print(f"⚠️  {exp_name}: aucune métrique de périodogramme trouvée (ou seules séries exclues)")
        else:
            print(f"❌ {exp_name}: erreur de lecture")
    
    if not all_results:
        print("❌ Aucune donnée de métrique valide trouvée")
        return
    
    print(f"\n📈 Analyse de {len(all_results)} expériences au total")
    
    # Classements par métrique principale
    metrics_config = {
        'power_ratio': {'higher_is_better': True, 'description': 'Rapport de puissance (plus grand = meilleur)'},
        'fap_at_Pinj': {'higher_is_better': False, 'description': 'Probabilité fausse alarme (plus petit = meilleur)'},
        'delta_P': {'higher_is_better': False, 'description': 'Écart période détectée (plus petit = meilleur)'},
    'n_sig_peaks_outside': {'higher_is_better': False, 'description': 'Pics parasites (plus petit = meilleur)'},
    'peak_power_at_Pdet': {'higher_is_better': True, 'description': 'Puissance au pic détecté (plus grand = meilleur)'},
    'peak_power_at_Pinj': {'higher_is_better': True, 'description': 'Puissance à P_inj (plus grand = meilleur)'}
    }
    
    for metric_name, config in metrics_config.items():
        rankings = rank_experiments_by_metric(
            all_results, 
            metric_name, 
            higher_is_better=config['higher_is_better']
        )
        print_ranking(rankings, metric_name, top_k=10)
    
    # Tableau de synthèse
    print(f"\n{'='*80}")
    print("📋 TABLEAU DE SYNTHÈSE - TOUTES MÉTRIQUES")
    print(f"{'='*80}")
    
    summary_df = create_summary_table(all_results)
    
    # Trier par power_ratio décroissant comme métrique principale
    summary_df_sorted = summary_df.sort_values('power_ratio', ascending=False, na_position='last')
    
    # Afficher le top 15
    print(summary_df_sorted.head(15).to_string(index=False, float_format='%.3g'))
    
    # Sauvegarder le tableau complet
    output_file = "experiments_ranking_summary.csv"
    summary_df_sorted.to_csv(output_file, index=False)
    print(f"\n💾 Tableau complet sauvegardé dans: {output_file}")
    
    # Statistiques finales
    print(f"\n📊 STATISTIQUES:")
    print(f"   • Expériences analysées: {len(all_results)}")
    print(f"   • Total séries RV: {sum(len(series_results) for series_results in all_results.values())}")
    print(f"   • Métriques power_ratio disponibles: {summary_df['power_ratio'].notna().sum()}")
    print(f"   • Métriques FAP disponibles: {summary_df['fap_at_Pinj'].notna().sum()}")
    
    # Meilleure performance globale
    best_power_ratio = summary_df_sorted.iloc[0] if not summary_df_sorted.empty else None
    if best_power_ratio is not None and not pd.isna(best_power_ratio['power_ratio']):
        print(f"\n🥇 MEILLEURE PERFORMANCE:")
        print(f"   • Expérience: {best_power_ratio['experiment']}")
        print(f"   • Série RV: {best_power_ratio['series']}")
        print(f"   • Power ratio: {best_power_ratio['power_ratio']:.2f}")
        if not pd.isna(best_power_ratio['fap_at_Pinj']):
            print(f"   • FAP: {best_power_ratio['fap_at_Pinj']:.2e}")


if __name__ == "__main__":
    main()
