from sklearn.linear_model import LinearRegression
import torch
import numpy as np
from astropy.timeseries import LombScargle

from src.modeling.train import load_experiment_checkpoint
from torch.utils.data import DataLoader
from src.dataset import generate_collate_fn, inject_dataset
from src.utils import clear_gpu_memory
import matplotlib.pyplot as plt
from src.ccf import get_full_ccf_analysis
from scipy.signal import find_peaks


def predict(model, dataset, batch_size=64, perturbation_value=1.0):
    """
    Extracts latent vectors and RV values from the model and dataset.

    Returns a dict containing:
      - all_s: latent vectors for y_obs
      - all_saug: latent vectors for y_aug
      - rv_pred_obs: RV predictions from RVEstimator on r_obs = y_obs - b_obs
      - rv_pred_aug: RV predictions from RVEstimator on r_aug = y_aug - b_obs
      - rv_true_aug: true augmentation offsets used to create y_aug (from collate)
      - all_yact, all_yact_aug, all_yobs_prime
      - all_yact_perturbed: decoder outputs when perturbing each latent dim (+perturbation_value)
    """

    all_s = []
    all_saug = []
    rv_pred_obs_list = []
    rv_pred_aug_list = []
    rv_true_aug_list = []
    all_yact = []
    all_yact_aug = []
    all_yobs_prime = []
    all_yact_perturbed = {}  # Dictionnaire pour organiser par dimension

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=generate_collate_fn(dataset=dataset),
    )

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch_yobs, batch_yaug, batch_voffset_true, batch_wavegrid = batch

            # RV predictions for y_obs and y_aug
            batch_vobs_pred, batch_vaug_pred = model.get_rvestimator_pred(
                batch_yobs=batch_yobs,
                batch_yaug=batch_yaug,
            )

            batch_yobs_prime, batch_yact, batch_yact_aug, batch_s, batch_saug = (
                model.get_spender_pred(
                    batch_yobs=batch_yobs,
                    batch_yaug=batch_yaug,
                    batch_wavegrid=batch_wavegrid,
                    batch_vobs_pred=batch_vobs_pred,
                )
            )

            latent_dim = batch_s.shape[1]

            # Initialiser les listes pour chaque dimension si pas encore fait
            if not all_yact_perturbed:
                for dim in range(latent_dim):
                    all_yact_perturbed[dim] = []

            for dim in range(latent_dim):
                batch_s_dim_perturbed = batch_s.clone()
                batch_s_dim_perturbed[:, dim] += perturbation_value
                batch_yact_perturbed = model.spender.decoder(batch_s_dim_perturbed)
                all_yact_perturbed[dim].append(
                    batch_yact_perturbed.cpu().detach().numpy()
                )

            # Accumuler sorties
            all_s.append(batch_s.cpu().detach().numpy())
            all_saug.append(batch_saug.cpu().detach().numpy())
            rv_true_aug_list.append(batch_voffset_true.cpu().detach().numpy())
            rv_pred_obs_list.append(batch_vobs_pred.cpu().detach().numpy())
            rv_pred_aug_list.append(batch_vaug_pred.cpu().detach().numpy())
            all_yact.append(batch_yact.cpu().detach().numpy())
            all_yact_aug.append(batch_yact_aug.cpu().detach().numpy())
            all_yobs_prime.append(batch_yobs_prime.cpu().detach().numpy())

    # Concatener
    all_s = np.concatenate(all_s, axis=0)
    all_saug = np.concatenate(all_saug, axis=0)
    rv_true_aug = np.concatenate(rv_true_aug_list, axis=0)
    rv_pred_obs = np.concatenate(rv_pred_obs_list, axis=0)
    rv_pred_aug = np.concatenate(rv_pred_aug_list, axis=0)
    all_yact = np.concatenate(all_yact, axis=0)
    all_yact_aug = np.concatenate(all_yact_aug, axis=0)
    all_yobs_prime = np.concatenate(all_yobs_prime, axis=0)

    # Concaténer et réorganiser all_yact_perturbed en (latent_dim, n_spectra, n_pixels)
    latent_dim = len(all_yact_perturbed)
    all_yact_perturbed_array = np.array(
        [np.concatenate(all_yact_perturbed[dim], axis=0) for dim in range(latent_dim)]
    )

    return {
        "all_s": all_s,
        "all_saug": all_saug,
        "rv_pred_obs": rv_pred_obs,
        "rv_pred_aug": rv_pred_aug,
        "rv_true_aug": rv_true_aug,
        # Compatibilité rétro: ancien nom (random offsets d'augmentation)
        "all_rvs": rv_true_aug,
        "all_yact": all_yact,
        "all_yact_aug": all_yact_aug,
        "all_yobs_prime": all_yobs_prime,
        "all_yact_perturbed": all_yact_perturbed_array,
    }


def get_vapparent(
    dataset: dict,
    CCFs_params: dict = None,
):
    """
    Calcule v_apparent qui correspond aux vitesses radiales obtenues par CCFs sur les spectres.
    """

    spectra = dataset["spectra"]
    wavegrid = dataset["wavegrid"]

    res = get_full_ccf_analysis(
        spectra=spectra,
        wavegrid=wavegrid,
        **CCFs_params,
    )

    v_apparent = res["rv"]
    fwhm = res["fwhm"]
    depth = res["depth"]
    bis_span = res["span"]

    return v_apparent, fwhm, depth, bis_span


def get_vref(
    dataset: dict,
    CCFs_params: dict = None,
):
    """
    Calcule v_ref qui correspond aux vitesses radiales obtenues par CCFs sur les spectres sans activités et ne contenant que le signal planétaire.
    """

    spectra = dataset["spectra_no_activity"]
    wavegrid = dataset["wavegrid"]

    res = get_full_ccf_analysis(
        spectra=spectra,
        wavegrid=wavegrid,
        **CCFs_params,
    )

    v_ref = res["rv"]

    return v_ref


def get_vtraditionnal(
    v_apparent: np.ndarray, fwhm: np.ndarray, depth: np.ndarray, bis_span: np.ndarray
):
    """
    Calcule v_traditionnal qui correspond aux vitesses radiales obtenues par la méthode de correction de l'activité traditionnelle.
    """
    # Prepare the feature matrix X with shape (n_samples, n_features)
    X = np.column_stack([fwhm, depth, bis_span])

    # Fit the linear regression model
    model = LinearRegression().fit(X, v_apparent)

    # Predict the activity-related component
    v_pred = model.predict(X)

    # Calculate v_traditionnal by removing the predicted activity component
    v_traditionnal = v_apparent - v_pred

    return v_traditionnal


def compute_latent_distances(all_s, all_saug, seed=None):
    """
    Fig 3.
    - Δs_rand : distances entre paires aléatoires de s_obs (i != j), tirées sans remise au sein de chaque paire.
    - Δs_aug  : distances entre s_obs_i et s_aug_i.
    Retourne (Δs_rand, Δs_aug).
    """
    n = all_s.shape[0]
    if seed is not None:
        np.random.seed(seed)

    # Pour chaque échantillon k, on tire une paire d'indices [i, j] avec i != j
    inds = np.array([np.random.choice(n, size=2, replace=False) for _ in range(n)])

    delta_s_rand = np.linalg.norm(all_s[inds[:, 0]] - all_s[inds[:, 1]], axis=1)

    delta_s_aug = np.linalg.norm(all_s - all_saug, axis=1)
    return delta_s_rand, delta_s_aug


def compute_periodogram_metrics(
    y_values,
    time_values,
    P_inj=None,
    min_period=1.0,
    max_period=None,  # si None → (tmax - tmin)/3
    n_periods=10000,
    fap_threshold=0.01,  # 1% par défaut
    exclude_width_frac=0.05,  # fenêtre ±5% autour de P_inj
    peak_prominence=None,  # None → auto (0.5 * std(power))
    ls_method="baluev",  # méthode FAP: "baluev" (rapide, robuste)
    fit_mean=True,
    center_data=True,
):
    """
    Calcule le périodogramme Lomb–Scargle et un set de métriques de détection.

    Paramètres
    ----------
    y_values : array (N,)
        Série RV (ou résidus) à analyser.
    time_values : array (N,)
        Temps (mêmes unités que celles attendues par LS, typiquement jours).
    P_inj : float | None
        Période injectée (jour). Si None, les métriques liées à P_inj seront None.
    min_period, max_period : float
        Bornes de période (jours). Si max_period=None → (tmax - tmin)/3.
    n_periods : int
        Taille de la grille logarithmique en période.
    fap_threshold : float
        Seuil FAP pour compter les pics “significatifs”.
    exclude_width_frac : float
        Largeur relative de la fenêtre à exclure autour de P_inj pour les stats “hors planète”.
        Exemple: 0.05 → exclut [P_inj*(1-0.05), P_inj*(1+0.05)].
    peak_prominence : float | None
        Prominence pour détecter des pics (scipy.signal.find_peaks).
        None → 0.5 * std(power).
    ls_method : str
        Méthode de FAP (‘baluev’, ‘naive’, ‘bootstrap’). ‘baluev’ recommandé.
    fit_mean, center_data : bool
        Options LombScargle.

    Renvoie
    -------
    periods : (M,)
        Grille de périodes (jours).
    power : (M,)
        Puissance LS correspondante.
    metrics : dict
        {
          "fap_at_Pinj": float|None,
          "power_ratio": float|None,   # P_planète / max(P_hors_planète)
          "n_sig_peaks_outside": int,  # nb de pics hors fenêtre avec FAP < fap_threshold
          "P_detected": float|None,    # meilleur pic dans la fenêtre de P_inj (si P_inj fourni)
          "delta_P": float|None        # |P_detected - P_inj|
        }
    """
    t = np.asarray(time_values, dtype=float)
    y = np.asarray(y_values, dtype=float)

    # borne supérieure par défaut : ~1/3 de la base temporelle
    if max_period is None:
        baseline = t.max() - t.min()
        max_period = max(baseline / 3.0, min_period * 1.1)

    # grille log en période → fréquences
    periods = np.logspace(np.log10(min_period), np.log10(max_period), n_periods)
    frequencies = 1.0 / periods

    # Lomb–Scargle
    ls = LombScargle(t, y, fit_mean=fit_mean, center_data=center_data)
    power = ls.power(frequencies)

    # Détection auto de prominence si non fournie
    if peak_prominence is None:
        peak_prominence = 0.5 * np.std(power)

    # Fenêtre autour de P_inj (pour exclure la planète des stats “hors planète”)
    if P_inj is not None and P_inj > 0:
        mask_planet_window = np.abs(periods - P_inj) <= exclude_width_frac * P_inj
    else:
        mask_planet_window = np.zeros_like(periods, dtype=bool)

    # --- Mesures autour de P_inj ---
    fap_at_Pinj = None
    power_ratio = None
    P_detected = None
    delta_P = None

    if P_inj is not None and P_inj > 0:
        # Pic détecté le plus puissant DANS la fenêtre de la planète
        idx_window = np.where(mask_planet_window)[0]
        if idx_window.size > 0:
            i_best_in_win = idx_window[np.argmax(power[idx_window])]
            P_detected = periods[i_best_in_win]
            P_planet_power = power[i_best_in_win]
            # FAP au niveau de puissance du pic planétaire
            fap_at_Pinj = float(
                ls.false_alarm_probability(P_planet_power, method=ls_method)
            )

            # Rapport de puissance : P_planète / max(P_hors_planète)
            if np.any(~mask_planet_window):
                max_outside = float(np.max(power[~mask_planet_window]))
                # éviter division par ~0
                denom = max_outside if max_outside > 0 else np.nan
                power_ratio = (
                    float(P_planet_power / denom) if denom == denom else None
                )  # nan-check
            else:
                power_ratio = None

            # Erreur de période
            delta_P = float(abs(P_detected - P_inj))
        else:
            # si la fenêtre est trop étroite pour contenir un point de grille
            fap_at_Pinj = None
            power_ratio = None
            P_detected = None
            delta_P = None

    # --- Nombre de pics “significatifs” hors planète (FAP < seuil) ---
    # On détecte d’abord les pics hors fenêtre, puis on évalue leur FAP individuelle
    periods_out = periods[~mask_planet_window]
    power_out = power[~mask_planet_window]

    # Attention: find_peaks travaille sur un signal indexé régulièrement.
    # Ici la grille en période est régulière en log, mais ça reste OK pour détecter des maxima locaux.
    peaks_out, _ = find_peaks(power_out, prominence=peak_prominence)
    if peaks_out.size:
        faps_out = ls.false_alarm_probability(power_out[peaks_out], method=ls_method)
        n_sig_peaks_outside = int(np.sum(faps_out < fap_threshold))
    else:
        n_sig_peaks_outside = 0

    metrics = {
        "fap_at_Pinj": fap_at_Pinj,
        "power_ratio": power_ratio,
        "n_sig_peaks_outside": n_sig_peaks_outside,
        "P_detected": P_detected,
        "delta_P": delta_P,
    }
    return periods, power, metrics


def compute_correlation(latent_s, rv_values):
    """
    Computes the correlation between latent space dimensions and RV values.
    and within latent space dimensions values.
    """
    # Calcul de la corrélation entre les dimensions latentes et les RV
    correlation_matrix = np.corrcoef(latent_s.T, rv_values, rowvar=False)

    # Extraire les corrélations entre les dimensions latentes et les RV
    latent_rv_correlation = correlation_matrix[:-1, -1]

    # Extraire les corrélations entre les dimensions latentes elles-mêmes
    latent_latent_correlation = correlation_matrix[:-1, :-1]

    return latent_rv_correlation, latent_latent_correlation


def main(checkpoint_path: str):
    exp_data = load_experiment_checkpoint(path=checkpoint_path)

    model = exp_data["model"]
    dataset = exp_data["dataset"]

    prediction = predict(model, dataset)

    all_s = prediction["all_s"]
    all_saug = prediction["all_saug"]
    all_rvs = prediction["all_rvs"]
    all_yact = prediction["all_yact"]
    all_yact_aug = prediction["all_yact_aug"]
    all_yobs_prime = prediction["all_yobs_prime"]
    all_yact_perturbed = prediction["all_yact_perturbed"]

    return True


# ==== Fonctions de plots ====
def plot_latent_distance_distribution(
    delta_s_rand, delta_s_aug, save_path=None, show_plot=False
):
    """
    Crée le plot de la Figure 3 avec les distributions de distances latentes.

    Args:
        delta_s_rand: Distances latentes pour les paires aléatoires
        delta_s_aug: Distances latentes pour les paires augmentées
        save_path: Chemin pour sauvegarder la figure (optionnel)
    """
    # Configuration du plot
    plt.figure(figsize=(8, 6))

    # Calcul des statistiques pour les légendes
    mean_rand = np.mean(delta_s_rand)
    mean_aug = np.mean(delta_s_aug)

    # Détermination automatique de la plage des valeurs
    all_values = np.concatenate([delta_s_rand, delta_s_aug])
    min_val = np.min(
        all_values[all_values > 0]
    )  # Éviter les valeurs nulles pour le log
    max_val = np.max(all_values)

    # Extension de la plage pour une meilleure visualisation
    x_min = min_val * 0.5
    x_max = max_val * 2.0

    # Création des histogrammes avec bins adaptés aux données réelles
    bins = np.logspace(np.log10(x_min), np.log10(x_max), 50)

    plt.hist(
        delta_s_rand,
        bins=bins,
        alpha=0.7,
        color="blue",
        label=f"(∆s_rand): {mean_rand:.3e}",
        density=False,
    )
    plt.hist(
        delta_s_aug,
        bins=bins,
        alpha=0.7,
        color="red",
        label=f"(∆s_aug): {mean_aug:.3e}",
        density=False,
    )

    # Configuration des axes et labels
    plt.xlabel("latent distance ∆s", fontsize=12)
    plt.ylabel("N", fontsize=12)
    plt.xscale("log")
    plt.xlim(x_min, x_max)

    # Ajout de la légende
    plt.legend(fontsize=12)

    # Configuration de la grille
    plt.grid(True, alpha=0.3)

    plt.subplots_adjust(bottom=0.15)

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Figure sauvegardée: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_periodogram(
    periods,
    power,
    metrics,
    P_inj=None,
    fap_threshold=0.01,
    exclude_width_frac=0.05,
    peak_prominence=None,
    title="Lomb–Scargle Periodogram",
):
    """
    Affiche le périodogramme avec:
    - ligne de seuil FAP (ex: 1%),
    - bande d'exclusion autour de P_inj,
    - marquage du meilleur pic détecté,
    - annotation des métriques principales.

    Note: Utilise les données déjà calculées par compute_periodogram_metrics
    """
    # Calcul du seuil FAP approximatif basé sur la puissance maximale
    # (approximation raisonnable pour l'affichage)
    max_power = np.max(power)
    fap_level = max_power * fap_threshold  # Approximation simple pour l'affichage

    # Détection de pics hors fenêtre pour marquage visuel
    if P_inj is not None:
        mask_excl = np.abs(periods - P_inj) <= exclude_width_frac * P_inj
    else:
        mask_excl = np.zeros_like(periods, dtype=bool)

    if peak_prominence is None:
        peak_prominence = 0.5 * np.std(power)

    p_out = power[~mask_excl]
    per_out = periods[~mask_excl]
    peaks_out, _ = find_peaks(p_out, prominence=peak_prominence)

    # Prépare la figure
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.semilogx(periods, power, lw=1.6)
    ax.set_xlabel("Période [jours]")
    ax.set_ylabel("Puissance Lomb–Scargle")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)

    # Seuil FAP
    ax.axhline(
        fap_level, ls="--", lw=1.2, label=f"Seuil FAP = {int(fap_threshold * 100)}%"
    )

    # Bande d’exclusion autour de P_inj
    if P_inj is not None and P_inj > 0:
        ax.axvspan(
            P_inj * (1 - exclude_width_frac),
            P_inj * (1 + exclude_width_frac),
            alpha=0.15,
            label=f"Bande autour de $P_{{inj}}$ = {P_inj:.4g} j",
        )

        # Meilleur pic détecté dans la fenêtre (issu des métriques)
        if metrics.get("P_detected") is not None:
            P_detected = metrics["P_detected"]
            # positionne un marqueur sur le pic détecté
            idx = np.argmin(np.abs(periods - P_detected))
            ax.plot(
                periods[idx],
                power[idx],
                marker="o",
                ms=7,
                mec="k",
                mfc="none",
                label=f"Pic détecté @ {P_detected:.4g} j",
            )
            # Ligne verticale P_inj (référence)
            ax.axvline(P_inj, color="k", lw=1.0, alpha=0.5)
    else:
        P_detected = None

    # Marque les pics “hors planète”
    if peaks_out.size:
        ax.plot(
            per_out[peaks_out], p_out[peaks_out], "x", ms=6, label="Pics (hors planète)"
        )

    # Petit encart avec les métriques
    lines = []
    if metrics.get("fap_at_Pinj") is not None:
        lines.append(f"FAP @ P_inj: {metrics['fap_at_Pinj']:.3g}")
    if metrics.get("power_ratio") is not None:
        lines.append(f"Power ratio: {metrics['power_ratio']:.3g}")
    if metrics.get("n_sig_peaks_outside") is not None:
        lines.append(f"# pics FAP<1% (hors): {metrics['n_sig_peaks_outside']}")
    if metrics.get("delta_P") is not None:
        lines.append(f"ΔP: {metrics['delta_P']:.3g} j")
    if lines:
        txt = "\n".join(lines)
        ax.text(
            0.02,
            0.98,
            txt,
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7", alpha=0.9),
            fontsize=10,
        )

    ax.legend(loc="best", frameon=True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    pass

    # Clear GPU memory after running the script
    clear_gpu_memory()

    checkpoint = load_experiment_checkpoint(
        path="experiments/aestra_local_experiment/models/aestra_base_config_final.pth",
    )

    model = checkpoint["model"]
    dataset = checkpoint["dataset"]

    prediction = predict(model, dataset, batch_size=32)

    all_s = prediction["all_s"]
    all_saug = prediction["all_saug"]
    v_correct = prediction["rv_pred_obs"]  # prédictions RVEstimator sur y_obs

    spectra = dataset.spectra.cpu().detach().numpy()
    wavegrid = dataset.wavegrid.cpu().detach().numpy()
    time_values = dataset.time_values.cpu().detach().numpy()
    planets_amplitudes = dataset.planets_amplitudes
    planets_periods = dataset.planets_periods
    planets_phases = dataset.planets_phases

    P_inj = planets_periods[0]  # On prend la première période pour l'exemple
    Kp_inj = planets_periods[0]  # On prend la première période pour l'exemple
    Phi_inj = planets_phases[0]

    CCFs_params = {
        "v_grid": np.arange(-20000, 20000, 100),
        "window_size_velocity": 820,
        "mask_type": "G2",
        "verbose": False,
        "batch_size": 64,
        "normalize": True,
    }

    raw_dataset = np.load(dataset.dataset_filepath, allow_pickle=True)
    print("Raw dataset loaded.")

    v_apparent, fwhm, depth, bis_span = get_vapparent(
        dataset=raw_dataset,
        CCFs_params=CCFs_params,
    )

    v_traditionnal = get_vtraditionnal(
        v_apparent=v_apparent,
        fwhm=fwhm,
        depth=depth,
        bis_span=bis_span,
    )

    v_ref = get_vref(
        dataset=raw_dataset,
        CCFs_params=CCFs_params,
    )

    # plt.figure(figsize=(10, 5))
    # plt.plot(v_correct, label="rv_pred_obs (model)")
    # plt.xlabel("Index")
    # plt.ylabel("Radial Velocity (m/s)")
    # plt.title("Model RV predictions (rv_pred_obs)")
    # plt.legend()
    # plt.show()

    # plt.figure(figsize=(10, 5))
    # plt.plot(v_apparent, label="v_apparent (CCF)")
    # plt.xlabel("Index")
    # plt.ylabel("Radial Velocity (m/s)")
    # plt.title("Radial Velocity (v_apparent)")
    # plt.legend()
    # plt.show()

    periods, power, metrics = compute_periodogram_metrics(
        y_values=v_correct,
        time_values=time_values,
        P_inj=P_inj,
        min_period=10.0,
        max_period=None,  # si None → (tmax - tmin)/3
        n_periods=10000,
        fap_threshold=0.01,  # 1% par défaut
        exclude_width_frac=0.1,  # fenêtre ±5% autour de P_in
        peak_prominence=None,  # None → auto (0.5 * std(power))
        ls_method="baluev",  # méthode FAP: "baluev"
        fit_mean=True,
        center_data=True,
    )
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    # Plot du périodogramme
    plot_periodogram(
        periods=periods,
        power=power,
        metrics=metrics,
        P_inj=P_inj,
        fap_threshold=0.01,
        exclude_width_frac=0.1,
        peak_prominence=None,
        title="Lomb–Scargle Periodogram of v_correct",
    )

    periods, power, metrics = compute_periodogram_metrics(
        y_values=v_apparent,
        time_values=time_values,
        P_inj=P_inj,
        min_period=10.0,
        max_period=None,  # si None → (tmax - tmin)/3
        n_periods=10000,
        fap_threshold=0.01,  # 1% par défaut
        exclude_width_frac=0.1,  # fenêtre ±5% autour de P_inj
        peak_prominence=None,  # None → auto (0.5 * std(power))
        ls_method="baluev",  # méthode FAP: "baluev"
        fit_mean=True,
        center_data=True,
    )

    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    # Plot du périodogramme
    plot_periodogram(
        periods=periods,
        power=power,
        metrics=metrics,
        P_inj=P_inj,
        fap_threshold=0.01,
        exclude_width_frac=0.1,
        peak_prominence=None,
        title="Lomb–Scargle Periodogram of v_apparent",
    )

    periods, power, metrics = compute_periodogram_metrics(
        y_values=v_traditionnal,
        time_values=time_values,
        P_inj=P_inj,
        min_period=10.0,
        max_period=None,  # si None → (tmax - tmin)/3
        n_periods=10000,
        fap_threshold=0.01,  # 1% par défaut
        exclude_width_frac=0.1,  # fenêtre ±5% autour de P_inj
        peak_prominence=None,  # None → auto (0.5 * std(power))
        ls_method="baluev",  # méthode FAP: "baluev"
        fit_mean=True,
        center_data=True,
    )

    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    plot_periodogram(
        periods=periods,
        power=power,
        metrics=metrics,
        P_inj=P_inj,
        fap_threshold=0.01,
        exclude_width_frac=0.1,
        peak_prominence=None,
        title="Lomb–Scargle Periodogram of v_traditionnal",
    )

    v_true = Kp_inj * np.sin(2 * np.pi * time_values / P_inj + Phi_inj)

    periods, power, metrics = compute_periodogram_metrics(
        y_values=v_ref,
        time_values=time_values,
        P_inj=P_inj,
        min_period=10.0,
        max_period=None,  # si None → (tmax - tmin)/3
        n_periods=10000,
        fap_threshold=0.01,  # 1% par défaut
        exclude_width_frac=0.1,  # fenêtre ±5% autour de P_inj
        peak_prominence=None,  # None → auto (0.5 * std(power))
        ls_method="baluev",  # méthode FAP: "baluev"
        fit_mean=True,
        center_data=True,
    )

    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    plot_periodogram(
        periods=periods,
        power=power,
        metrics=metrics,
        P_inj=P_inj,
        fap_threshold=0.01,
        exclude_width_frac=0.1,
        peak_prominence=None,
        title="Lomb–Scargle Periodogram of v_ref",
    )

    # # Calcul des corrélations entre les dimensions latentes et les RV
    # latent_vcorrect_correlation, latent_latent_correlation = compute_correlation(
    #     latent_s=all_s,
    #     rv_values=v_correct,
    # )
    # print("Latent space - RV correlation (v_correct):")
    # for i, corr in enumerate(latent_vcorrect_correlation):
    #     print(f"  Dimension {i}: {corr:.4f}")
    # print("\nLatent space - Latent space correlation (v_correct):")
    # for i in range(latent_latent_correlation.shape[0]):
    #     for j in range(latent_latent_correlation.shape[1]):
    #         print(
    #             f"  Dimension {i} - Dimension {j}: {latent_latent_correlation[i, j]:.4f}"
    #         )
    # print("\n")

    # latent_vapparent_correlation, latent_latent_correlation = compute_correlation(
    #     latent_s=all_s,
    #     rv_values=v_apparent,
    # )
    # print("Latent space - RV correlation (v_apparent):")
    # for i, corr in enumerate(latent_vapparent_correlation):
    #     print(f"  Dimension {i}: {corr:.4f}")
    # print("\nLatent space - Latent space correlation (v_apparent):")
    # for i in range(latent_latent_correlation.shape[0]):
    #     for j in range(latent_latent_correlation.shape[1]):
    #         print(
    #             f"  Dimension {i} - Dimension {j}: {latent_latent_correlation[i, j]:.4f}"
    #         )
    # print("\n")
    # latent_vtraditionnal_correlation, latent_latent_correlation = compute_correlation(
    #     latent_s=all_s,
    #     rv_values=v_traditionnal,
    # )
    # print("Latent space - RV correlation (v_traditionnal):")
    # for i, corr in enumerate(latent_vtraditionnal_correlation):
    #     print(f"  Dimension {i}: {corr:.4f}")
    # print("\nLatent space - Latent space correlation (v_traditionnal):")
    # for i in range(latent_latent_correlation.shape[0]):
    #     for j in range(latent_latent_correlation.shape[1]):
    #         print(
    #             f"  Dimension {i} - Dimension {j}: {latent_latent_correlation[i, j]:.4f}"
    #         )
    # print("\n")
    # latent_vref_correlation, latent_latent_correlation = compute_correlation(
    #     latent_s=all_s,
    #     rv_values=v_ref,
    # )
    # print("Latent space - RV correlation (v_ref):")
    # for i, corr in enumerate(latent_vref_correlation):
    #     print(f"  Dimension {i}: {corr:.4f}")
    # print("\nLatent space - Latent space correlation (v_ref):")
    # for i in range(latent_latent_correlation.shape[0]):
    #     for j in range(latent_latent_correlation.shape[1]):
    #         print(
    #             f"  Dimension {i} - Dimension {j}: {latent_latent_correlation[i, j]:.4f}"
    #         )
    # print("\n")
    # # Calcul des distances latentes
    # delta_s_rand, delta_s_aug = compute_latent_distances(all_s, all_saug, seed=42)
    # print("Latent distances computed.")
