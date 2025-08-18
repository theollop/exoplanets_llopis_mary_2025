# We'll generate a clean PDF from the provided outline using ReportLab.
# We'll also save a matching LaTeX template (.tex) the user can edit/compile.
# Files will be written to /mnt/data so the user can download them.

from datetime import datetime
import os, textwrap, sys

pdf_path = "/mnt/data/rapport_aestra_plan.pdf"
tex_path = "/mnt/data/rapport_aestra_plan.tex"

# ---------- 1) Try to build a nicely formatted PDF with reportlab ----------
pdf_ok = True
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        PageBreak,
        TableOfContents,
        ListFlowable,
        ListItem,
        HRFlowable,
        KeepTogether,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
    from reportlab.lib.units import cm
    from reportlab.lib import colors
except Exception as e:
    pdf_ok = False
    err_import = repr(e)

if pdf_ok:
    # Styles
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitleBig",
            parent=styles["Title"],
            fontSize=24,
            leading=28,
            alignment=TA_CENTER,
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SubTitle",
            parent=styles["Title"],
            fontSize=14,
            leading=18,
            alignment=TA_CENTER,
            textColor=colors.grey,
        )
    )
    styles.add(
        ParagraphStyle(
            name="H1",
            parent=styles["Heading1"],
            fontSize=16,
            leading=20,
            spaceBefore=12,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="H2",
            parent=styles["Heading2"],
            fontSize=13,
            leading=16,
            spaceBefore=10,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="H3",
            parent=styles["Heading3"],
            fontSize=11.5,
            leading=14,
            spaceBefore=8,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Body",
            parent=styles["BodyText"],
            fontSize=10.5,
            leading=14,
            alignment=TA_JUSTIFY,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Mono",
            parent=styles["BodyText"],
            fontName="Courier",
            fontSize=10,
            leading=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TipTitle",
            parent=styles["Heading3"],
            textColor=colors.white,
            backColor=colors.HexColor("#1f2937"),
            fontSize=10.5,
            leading=13,
            spaceBefore=6,
            spaceAfter=2,
            leftIndent=6,
            rightIndent=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TipBody",
            parent=styles["Body"],
            backColor=colors.HexColor("#f3f4f6"),
            borderPadding=(6, 6, 6, 6),
            leftIndent=6,
            rightIndent=6,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Small",
            parent=styles["BodyText"],
            fontSize=9,
            leading=12,
            textColor=colors.grey,
        )
    )

    # Document
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=2 * cm,
    )
    flow = []

    # Cover page
    flow.append(Spacer(1, 4 * cm))
    flow.append(
        Paragraph(
            "Plan de Rapport – AESTRA / RV & Activité stellaire", styles["TitleBig"]
        )
    )
    flow.append(
        Paragraph(
            "Structure éditoriale prête à remplir (sections, équations, figures, bonnes pratiques)",
            styles["SubTitle"],
        )
    )
    flow.append(Spacer(1, 0.8 * cm))
    today = datetime.now().strftime("%d %B %Y")
    flow.append(Paragraph(f"{today}", styles["Small"]))
    flow.append(Spacer(1, 8))
    flow.append(HRFlowable(width="100%", color=colors.HexColor("#9ca3af")))
    flow.append(Spacer(1, 12))
    flow.append(
        Paragraph(
            "Ce document fournit un gabarit clair et cohérent pour rédiger le rapport. "
            "Il inclut des encadrés “Bonnes pratiques”, une table des matières et des sections numérotées.",
            styles["Body"],
        )
    )
    flow.append(PageBreak())

    # Table of contents
    flow.append(Paragraph("Table des matières", styles["H1"]))
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(
            fontSize=11.5,
            name="TOCHeading1",
            leftIndent=10,
            firstLineIndent=-10,
            spaceBefore=4,
            leading=13,
        ),
        ParagraphStyle(
            fontSize=10.5,
            name="TOCHeading2",
            leftIndent=20,
            firstLineIndent=-10,
            spaceBefore=2,
            leading=12,
        ),
        ParagraphStyle(
            fontSize=10,
            name="TOCHeading3",
            leftIndent=30,
            firstLineIndent=-10,
            spaceBefore=1,
            leading=11,
        ),
    ]
    flow.append(toc)
    flow.append(PageBreak())

    # Helper for section headings that register in TOC
    def add_heading(text, level=1):
        style = (
            styles["H1"] if level == 1 else styles["H2"] if level == 2 else styles["H3"]
        )
        para = Paragraph(text, style)
        # Notify the TOC of the new entry
        # ('TOCEntry', (level, text, page number))
        flow.append(para)

        # reportlab auto-adds entries when using afterFlowable; emulate here:
        def add_toc_entry(canvas, doc):
            from reportlab.platypus.tableofcontents import TableOfContents

            canvas.bookmarkPage(text)
            doc.notify("TOCEntry", (level, text, doc.page))

        flow.append(Spacer(1, 0))
        flow[-1].postponed_draw = add_toc_entry  # attach hook
        return para

    def tip_block(title, body_paras):
        elems = []
        elems.append(Paragraph(f"Bonnes pratiques — {title}", styles["TipTitle"]))
        for p in body_paras:
            elems.append(Paragraph(p, styles["TipBody"]))
        return KeepTogether(elems)

    def bullet(items, level=0):
        return ListFlowable(
            [
                ListItem(Paragraph(it, styles["Body"]), leftIndent=level * 12)
                for it in items
            ],
            bulletType="bullet",
            start="•",
            leftIndent=12 + level * 12,
            spaceBefore=2,
            spaceAfter=6,
        )

    # 0. Pages liminaires (non numérotées)
    add_heading("0. Pages liminaires (non numérotées)", level=1)
    flow.append(
        bullet(
            [
                "Page de garde, Remerciements, Résumé/Abstract (10–15 lignes, sans références), Table des matières, Listes (figures/tableaux/abréviations)."
            ]
        )
    )
    flow.append(
        tip_block(
            "Préliminaires",
            [
                "Style impersonnel ; résumer objectifs, méthode, 2–3 résultats chiffrés clés et perspectives en 10–15 lignes max.",
                "Vérifier la cohérence des listes et l’orthographe des titres de figures/tableaux.",
            ],
        )
    )

    # 1. Introduction
    add_heading("1. Introduction (1–2 pages)", level=1)
    flow.append(
        bullet(
            [
                "Contexte (exoplanètes, RV, activité stellaire, effet Doppler, HARPS-N, PLATO, RV Data Challenge).",
                "Problématique scientifique : séparation planète / activité.",
                "Objectifs et contributions : pipeline AESTRA, améliorations envisagées, résultats obtenus.",
                "Positionnement vs littérature : AESTRA 2023, SOAP 2.0 vs SOAP-GPU, RASSINE, transferts de domaine.",
                "Plan du document.",
            ]
        )
    )
    flow.append(
        tip_block(
            "Introduction",
            [
                "Annoncer clairement les questions de recherche ; donner des ordres de grandeur ; définir chaque acronyme à sa première occurrence."
            ],
        )
    )

    # 2. État de l’art
    add_heading("2. État de l’art", level=1)
    flow.append(
        bullet(
            [
                "Comment obtenir les RV & signatures d’activité (FWHM, BIS, profondeur).",
                "Méthodes CCF/masques vs approches ML/DL.",
                "AESTRA : concept, pertes, dé-trending latent.",
                "SOAP 2.0 (CCF) vs SOAP-GPU (spectres) ; normalisation RASSINE.",
            ]
        )
    )
    flow.append(
        Paragraph(
            "Inclure un schéma simple et 1–2 équations clés numérotées ; citations normalisées.",
            styles["Body"],
        )
    )

    # 3. Données et préparation
    add_heading("3. Données et préparation", level=1)
    add_heading("3.0 Matériel sur lequel on travaille", level=2)
    flow.append(
        bullet(
            [
                "Explication d’un spectre : acquisition, allure, normalisation, raies.",
                "Déformations par activité vs exoplanètes et signatures spectrales (comment les différencier).",
                "Équation de décomposition AESTRA ; équation Doppler relativiste + linéarisée.",
                "Illustrations : zoom de raie (activité vs Doppler), GIF SOAP-GPU montrant l’effet sur les RV (CCF).",
                "Interpolation au cœur du réseau (linéaire/cubique) – impact négligeable pour faibles vitesses.",
                "Précision numérique pour faibles vitesses (ex.: float64 pour γ Doppler).",
            ]
        )
    )
    add_heading("3.1 Datasets de spectres", level=2)
    flow.append(
        bullet(
            [
                "SOAP-GPU (~3000 spectres).",
                "RV Data Challenge (HARPS, ~1200 spectres).",
                "Pour chaque dataset : spectres + template + temps d’observation ; différences vs pipeline papier basé CCF ; activité ; production ; outliers et détection.",
            ]
        )
    )
    add_heading("3.2 Normalisation : RASSINE", level=2)
    flow.append(
        Paragraph(
            "Normalisation du template et des spectres ; justification et impacts.",
            styles["Body"],
        )
    )
    add_heading("3.3 Pipeline de génération (`create_soap_gpu_paper_dataset`)", level=2)
    flow.append(
        bullet(
            [
                "Exclusion des outliers, fenêtre en λ, downscaling, lissage, bruit photonique, injections planètes (K/P/φ), proxies CCF, métadonnées…"
            ]
        )
    )
    add_heading("3.4 Splits & validation", level=2)
    flow.append(
        Paragraph(
            "Uniquement train et justification de l’absence de train/val/test classiques.",
            styles["Body"],
        )
    )
    flow.append(
        tip_block(
            "Données",
            [
                "Fournir tableau récap des paramètres ; unités (Å, m/s) ; versions/outils et seeds."
            ],
        )
    )

    # 4. Méthodologie : fonctionnement d’AESTRA
    add_heading("4. Méthodologie : fonctionnement d’AESTRA", level=1)
    add_heading("4.1 Formulation générale", level=2)
    flow.append(
        Paragraph(
            '<font face="Courier">y_obs = D(y_act + b_rest, v_encode)</font> — rôles de y_act, b_rest, v_encode.',
            styles["Mono"],
        )
    )
    add_heading("4.2 Architecture & flux de données", level=2)
    flow.append(
        Paragraph(
            "SPENDER (encodeur/décodeur) ; RV Estimator (CNN → softmax → MLP).",
            styles["Body"],
        )
    )
    add_heading("4.3 Fonctions de perte & régularisation", level=2)
    flow.append(
        Paragraph('<font face="Courier">L_fid, L_RV, L_reg, L_c</font>', styles["Mono"])
    )
    add_heading("4.4 Dé-trending par latents", level=2)
    flow.append(
        Paragraph(
            "Lissage gaussien dans l’espace latent (paramètres : σ_R, k-NN).",
            styles["Body"],
        )
    )
    add_heading("4.5 Protocole d’entraînement", level=2)
    flow.append(
        bullet(
            [
                "Phases RV-only → joint, AMP, early-stopping, augmentation Doppler (linéaire/cubique), checkpoints.",
                "Schéma d’architecture ; équations de pertes numérotées ; choix d’hypers (motivation).",
            ]
        )
    )

    # 5. Expérience baseline
    add_heading("5. Expérience “baseline” (cadre AESTRA, spectres SOAP-GPU)", level=1)
    add_heading("5.1 Protocole expérimental", level=2)
    flow.append(
        bullet(
            [
                "Dataset (ex. 5000–5050 Å, dx2, sm3, SNR, N), splits, seeds ; config AESTRA reproduisant le papier.",
                "Séries de vitesses (v_apparent, v_traditionnal, v_encode, v_correct, v_ref), périodogrammes, FAP, bootstrap.",
                "Traces RV par CCF vs signal injecté : analyse d’allure.",
            ]
        )
    )
    add_heading("5.2 Losses & évolution d’entraînement", level=2)
    flow.append(
        Paragraph("Courbes des pertes, commentaires et diagnostics.", styles["Body"])
    )
    add_heading("5.3 Périodogrammes & FAP", level=2)
    flow.append(
        Paragraph(
            "Lomb-Scargle, significativité des pics, récupération de K si injections (courbes de complétude).",
            styles["Body"],
        )
    )
    add_heading("5.4 Matrice de corrélation & espace latent", level=2)
    flow.append(
        Paragraph(
            "Corrélations entre vitesses et latents ; visualisation 3D ; détection de fuites Doppler (gradients).",
            styles["Body"],
        )
    )

    # 6. Limites & pistes
    add_heading("6. Limites du modèle & pistes d’amélioration", level=1)
    add_heading("6.1 Limites observées (baseline)", level=2)
    flow.append(
        bullet(
            [
                "Amplification d’amplitude (ex.: 0.1 m/s → ~1 m/s), sensibilité au dé-trending, corr(latents, RV)."
            ]
        )
    )
    add_heading("6.2 Expériences d’amélioration", level=2)
    flow.append(
        bullet(
            [
                "(a) b_obs ≠ b_rest (papier) vs b_obs = b_rest (init moyenne, entraînables) — stabilisation & réduction des fuites Doppler.",
                "(b) Encodage en rest-frame (donner y_obs^rest à l’encodeur) ; latents invariants ; meilleure séparation activité/planète.",
                "(c) Conditionnement par indicateurs d’activité CCF (depth, FWHM, bisector span) — concat vs FiLM.",
            ]
        )
    )
    flow.append(
        Paragraph(
            "Pour chaque expérience : protocole, paramètres modifiés, figures clés, critères de succès, analyse des échecs.",
            styles["Body"],
        )
    )

    # 7. Adaptation de domaine
    add_heading(
        "7. Adaptation de domaine : pré-entraînement SOAP-GPU → fine-tuning RV Data Challenge",
        level=1,
    )
    add_heading("7.1 Protocole", level=2)
    flow.append(
        Paragraph(
            "Prétrain : SOAP-GPU rééchantillonné HARPS-N ; Fine-tune : RV Data Challenge ; comparaison from-scratch.",
            styles["Body"],
        )
    )
    add_heading("7.2 Résultats", level=2)
    flow.append(
        Paragraph(
            "Détection (périodogrammes, FAP, AUC), Estimation (biais K, σ_RV, robustesse SNR), figures (pics de période, corner MCMC).",
            styles["Body"],
        )
    )
    add_heading("7.3 Discussion ciblée", level=2)
    flow.append(
        Paragraph(
            "Gains liés à l’alignement de grille + transfert ; limites (mismatch PSF/SNR). Protocole reproductible (YAML, seeds).",
            styles["Body"],
        )
    )

    # 8. Discussion générale
    add_heading("8. Discussion générale", level=1)
    flow.append(
        Paragraph(
            "Mise en perspective vs littérature ; interprétation physique ; menaces sur la validité (mismatch de domaine, taille d’échantillon, choix de σ_R…).",
            styles["Body"],
        )
    )
    flow.append(
        bullet(
            [
                "Difficultés rencontrées : temps de calcul, ressources, cluster fermé, lourdeur des spectres astrophysiques…"
            ]
        )
    )

    # 9. Conclusion & perspectives
    add_heading("9. Conclusion & perspectives", level=1)
    flow.append(
        bullet(
            [
                "Bilan : pipeline, SOAP-GPU + RASSINE, rééchantillonnage, adaptation de domaine.",
                "Perspectives : AESTRAM (par raies), pruning, élargissement spectral (≈4000–6000 Å), autres instruments.",
            ]
        )
    )

    # 10. Références
    add_heading("10. Références", level=1)
    flow.append(
        Paragraph(
            "Bibliographie normalisée ; toutes les références citées. Gestion via BibTeX/Zotero ; cohérence du style.",
            styles["Body"],
        )
    )

    # 11. Annexes
    add_heading("11. Annexes (optionnelles)", level=1)
    flow.append(
        bullet(
            [
                "Configs YAML, scripts run_experiment.sh.",
                "Détails du builder datasets & validations (interpolation, SNR → poids).",
                "Figures supplémentaires (périodogrammes complets, latents, corner MCMC).",
            ]
        )
    )

    # Checklist
    add_heading("Mini-checklist rédaction", level=1)
    flow.append(
        bullet(
            [
                "Pas de 1ʳᵉ personne ; unités SI (m/s, Å) ; variables définies dès la première occurrence.",
                "Équations/figures/tableaux numérotés et appelés ; légendes autonomes.",
                "Chaque résultat avec incertitudes + commentaire interprétatif.",
                "Seeds, splits, versions logicielles, chemins de données documentés (reproductibilité).",
            ]
        )
    )

    # Build the PDF
    doc.build(flow)
else:
    with open(pdf_path.replace(".pdf", "_FAILED.txt"), "w") as f:
        f.write("Impossible de générer le PDF car reportlab n'est pas disponible.\n")
        f.write(f"Erreur d'import: {err_import}\n")

# ---------- 2) Write a LaTeX template matching the outline ----------
latex = r"""
\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.5cm]{geometry}
\usepackage[french]{babel}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{siunitx}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{enumitem}
\usepackage{xcolor}
\hypersetup{colorlinks=true, linkcolor=blue, urlcolor=blue, citecolor=blue}

\setlist[itemize]{noitemsep, topsep=2pt}
\setlist[enumerate]{noitemsep, topsep=2pt}

\newenvironment{goodpractices}[1]{%
  \par\noindent\colorbox{black}{\parbox{\dimexpr\linewidth-2\fboxsep\relax}{\color{white}\textbf{Bonnes pratiques — #1}}}\par
  \vspace{4pt}
  \begin{quote}\small
}{%
  \end{quote}\vspace{6pt}
}

\title{Plan de Rapport --- AESTRA / RV \& Activit\'e stellaire}
\author{}
\date{\today}

\begin{document}
\maketitle
\tableofcontents
\newpage

\section*{0. Pages liminaires (non num\'erot\'ees)}
\begin{itemize}
  \item Page de garde, Remerciements, \textbf{R\'esum\'e/Abstract (10--15 lignes, sans r\'ef\'erences)}, Table des mati\`eres, Listes (figures/tableaux/abr\'eviations).
\end{itemize}
\begin{goodpractices}{Pr\'eliminaire}
Style impersonnel; r\'esumer objectifs, m\'ethode, \emph{r\'esultats chiffr\'es cl\'es}, perspectives en 10--15 lignes max.
\end{goodpractices}

\section{Introduction (1--2 pages)}
\begin{enumerate}[label=\arabic*.]
  \item Contexte (exoplan\`etes, RV, activit\'e stellaire, effet Doppler, HARPS-N, PLATO, RV Data Challenge).
  \item Probl\'ematique scientifique (s\'eparation plan\`ete/activit\'e)
  \item Objectifs et contributions (pipeline AESTRA, am\'eliorations envisag\'ees, r\'esultats)
  \item Positionnement vs litt\'erature (AESTRA 2023, SOAP 2.0 vs SOAP-GPU, RASSINE, transferts de domaine)
  \item Plan du document
\end{enumerate}
\begin{goodpractices}{Introduction}
Annoncer clairement les questions de recherche; chiffres d'ordre de grandeur; acronymes d\'efinis \`a la premi\`ere occurrence.
\end{goodpractices}

\section{\'Etat de l'art}
\begin{itemize}
  \item Obtenir RV \& signatures d’activit\'e (FWHM, BIS, profondeur)
  \item M\'ethodes CCF/masques vs approches ML/DL
  \item AESTRA (concept, pertes, d\'e-trending latent)
  \item SOAP 2.0 (CCF) vs \textbf{SOAP-GPU (spectres)}; normalisation \textbf{RASSINE}
\end{itemize}

\section{Donn\'ees et pr\'eparation}
\subsection*{3.0 Mat\'eriel sur lequel on travaille}
\begin{itemize}
  \item Explication spectre, acquisition, allure, normalisation, raies, etc.
  \item D\'eformations par activit\'e ou exoplan\`etes; signatures spectrales (comment diff\'erencier).
  \item \'Equation de d\'ecomposition AESTRA; \'equation Doppler relativiste + lin\'earis\'ee.
  \item Illustrations: zoom de raie (activit\'e vs Doppler); GIF SOAP-GPU montrant l'effet sur les RV (CCF).
  \item Interpolation (lin\'eaire/cubique) --- impact n\'egligeable pour faibles vitesses.
  \item Pr\'ecision num\'erique pour faibles vitesses (float64 pour $\gamma$ Doppler).
\end{itemize}

\subsection{Datasets de spectres}
\begin{itemize}
  \item \textbf{SOAP-GPU} ($\sim$ 3000 spectres)
  \item \textbf{RV Data Challenge (HARPS)} ($\sim$ 1200 spectres)
  \item Pour chaque dataset: spectres, template, temps d'observation, diff\'erences vs pipeline papier (CCF), activit\'e, production, outliers (\& d\'etection).
\end{itemize}

\subsection{Normalisation: RASSINE}
Normalisation du template et des spectres; justification et impacts.

\subsection{Pipeline de g\'en\'eration (\texttt{create\_soap\_gpu\_paper\_dataset})}
Exclusion des outliers; fen\^etre en $\lambda$; downscaling; lissage; bruit photonique; injections plan\`etes ($K/P/\phi$); proxies CCF; m\'etadonn\'ees.

\subsection{Splits \& validation}
Uniquement train; justification (pourquoi pas train/val/test).

\begin{goodpractices}{Donn\'ees}
Tableau r\'ecapitulatif des param\`etres; unit\'es (\AA, m/s); versions/outils.
\end{goodpractices}

\section{M\'ethodologie: fonctionnement d'AESTRA}
\subsection{Formulation g\'en\'erale}
\'Equation centrale:
\begin{equation}
y_{\mathrm{obs}} = D\!\big(y_{\mathrm{act}} + b_{\mathrm{rest}}, v_{\mathrm{encode}}\big).
\end{equation}
R\^oles de $y_{\mathrm{act}}$, $b_{\mathrm{rest}}$, $v_{\mathrm{encode}}$.

\subsection{Architecture \& flux de donn\'ees}
SPENDER (encodeur/d\'ecodeur); RV Estimator (CNN $\to$ softmax $\to$ MLP).

\subsection{Pertes \& r\'egularisation}
$\mathcal{L}_{\text{fid}}$, $\mathcal{L}_{\text{RV}}$, $\mathcal{L}_{\text{reg}}$, $\mathcal{L}_{\text{c}}$.

\subsection{D\'e-trending par latents}
Lissage gaussien dans l’espace latent (param\`etres: $\sigma_R$, k-NN).

\subsection{Protocole d’entra\^inement}
Phases \textbf{RV-only} $\to$ \textbf{joint}, AMP, early-stopping, augmentation Doppler (lin\'eaire/cubique), checkpoints.

\section{Exp\'erience ``baseline'' (cadre AESTRA, spectres SOAP-GPU)}
\subsection{Protocole exp\'erimental}
Dataset (5000--5050~\AA, dx2, sm3, SNR, N), splits, seeds; configuration AESTRA (papier). S\'eries de vitesses ($v_{\mathrm{apparent}}, v_{\mathrm{traditionnal}}, v_{\mathrm{encode}}, v_{\mathrm{correct}}, v_{\mathrm{ref}}$), p\'eriodogrammes, FAP, bootstrap. Traces RV par CCF vs signal inject\'e.

\subsection{Losses \& \'evolution}
Courbes des pertes et analyse.

\subsection{P\'eriodogrammes \& FAP}
Lomb--Scargle, significativit\'e des pics; r\'ecup\'eration de $K$ si injections (compl\'etude).

\subsection{Matrice de corr\'elation \& espace latent}
Corr\'elations entre vitesses et latents; 3D; fuites Doppler (gradients).

\section{Limites du mod\`ele \& pistes d’am\'elioration}
\subsection{Limites observ\'ees (baseline)}
Amplification d’amplitude (0.1~m/s $\to$ $\sim$1~m/s); sensibilit\'e au d\'e-trending; corr(latents,RV).

\subsection{Exp\'eriences d’am\'elioration}
\begin{enumerate}[label=(\alph*)]
  \item $b_{\mathrm{obs}} \neq b_{\mathrm{rest}}$ (papier) vs $b_{\mathrm{obs}} = b_{\mathrm{rest}}$ (init moyenne, entra\^inables) --- stabilisation \& r\'eduction fuite Doppler.
  \item Encodage en rest-frame: $y_{\mathrm{obs}}^{\mathrm{rest}} = D^{-1}(y_{\mathrm{obs}}, v_{\mathrm{encode}})$; latents plus invariants; meilleure s\'eparation activit\'e/plan\`ete.
  \item Conditionnement par indicateurs d'activit\'e CCF (depth, FWHM, bisector span) --- concat vs FiLM.
\end{enumerate}
Pour chaque: protocole, param\`etres modifi\'es, figures cl\'es, crit\`eres de succ\`es, analyse des \'ech\`ecs.

\section{Adaptation de domaine: pr\'e-entra\^inement SOAP-GPU $\to$ fine-tuning RV Data Challenge}
\subsection{Protocole}
Pr\'etrain: SOAP-GPU r\'eechantillonn\'e HARPS-N; Fine-tune: RV Data Challenge; comparaison from-scratch HARPS.

\subsection{R\'esultats}
D\'etection: p\'eriodogrammes, FAP, AUC. Estimation: biais $K$, $\sigma_{\mathrm{RV}}$, robustesse vs SNR. Figures: pics de p\'eriode, corner plots MCMC (P, K, $\phi$, $e$).

\subsection{Discussion cibl\'ee}
Gains li\'es \`a l’alignement de grille + transfert; limites (mismatch PSF/SNR). Protocole reproductible (YAML, seeds), comparaison statistique claire.

\section{Discussion g\'en\'erale}
Mise en perspective des r\'esultats vs litt\'erature; interpr\'etation physique; menaces sur la validit\'e (mismatch de domaine, taille d’\'echantillon, choix de $\sigma_R$, etc.).\\
\textbf{Difficult\'es rencontr\'ees}: temps de calcul, ressources, cluster ferm\'e, lourdeur des spectres.

\section{Conclusion \& perspectives}
Bilan des contributions (pipeline, SOAP-GPU+RASSINE, r\'eechantillonnage, adaptation de domaine).\\
Perspectives: \textbf{AESTRAM} (par raies), pruning, \'elargissement spectral ($\approx$4000--6000~\AA), autres instruments.

\section{R\'ef\'erences}
Bibliographie normalis\'ee; toutes les r\'ef\'erences \emph{cit\'ees} dans le texte (Bib\TeX/Zotero).

\appendix
\section{Annexes (optionnelles)}
A. Configs YAML, scripts \texttt{run\_experiment.sh}.\\
B. D\'etails du builder datasets \& validations (interpolations, SNR$\to$poids).\\
C. Figures suppl\'ementaires (p\'eriodogrammes complets, latents, corner MCMC).

\section*{Mini-checklist r\'edaction}
\begin{itemize}
  \item Pas de 1\`ere personne; unit\'es SI (m/s, \AA); variables d\'efinies \`a la premi\`ere occurrence.
  \item \'Equations/figures/tableaux num\'erot\'es \& appel\'es; l\'egendes autonomes.
  \item Chaque r\'esultat avec incertitudes \& commentaire interpr\'etatif.
  \item Seeds, splits, versions logicielles, chemins de donn\'ees document\'es (reproductibilit\'e).
\end{itemize}

\end{document}
"""
with open(tex_path, "w", encoding="utf-8") as f:
    f.write(latex)

print(
    "PDF created:" if pdf_ok else "PDF generation failed; LaTeX template saved.",
    pdf_path,
)
print("LaTeX template:", tex_path)
