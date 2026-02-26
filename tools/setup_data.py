"""Prépare les données DPE pour la compétition Codabench.

Source     : data/dpe_2025.csv  (généré par tools/fetch_dpe_data.py)
Target     : etiquette_dpe  (classification A → G)

Structure de sortie :
    dev_phase/input_data/train/train_features.csv
    dev_phase/input_data/train/train_labels.csv
    dev_phase/input_data/test/test_features.csv
    dev_phase/input_data/private_test/private_test_features.csv
    dev_phase/reference_data/test_labels.csv
    dev_phase/reference_data/private_test_labels.csv

Usage :
    python tools/setup_data.py
    python tools/setup_data.py --input data/dpe_2025.csv --seed 123
"""

from pathlib import Path
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split


# ── Chemins de sortie ─────────────────────────────────────────────────────────
DATA_DIR = Path("dev_phase") / "input_data"
REF_DIR  = Path("dev_phase") / "reference_data"

# ── Variable cible ────────────────────────────────────────────────────────────
TARGET = "etiquette_dpe"

# ── Métadonnées / identifiants (non prédictifs) ───────────────────────────────
META_COLS = [
    "numero_dpe",
    "date_derniere_modification_dpe",
    "date_etablissement_dpe",
    "modele_dpe",
    "version_dpe",
    "methode_application_dpe",
    "identifiant_ban",
    "score_ban",
    "statut_geocodage",
]

# ── Colonnes leaky : directement dérivées de la target ───────────────────────
# L'étiquette DPE (A-G) est déterminée par conso_5_usages_par_m2_ep
# et emission_ges_5_usages_par_m2. Toutes les colonnes ci-dessous sont
# calculées dans le même bilan énergétique que la target.
LEAKY_COLS = [
    # Consommations énergie primaire (critère étiquette)
    "conso_5_usages_ep",
    "conso_5_usages_par_m2_ep",
    "conso_chauffage_ep",
    "conso_ecs_ep",
    "conso_refroidissement_ep",
    "conso_eclairage_ep",
    "conso_auxiliaires_ep",
    # Consommations énergie finale
    "conso_5 usages_ef",
    "conso_5 usages_par_m2_ef",
    "conso_chauffage_ef",
    "conso_ecs_ef",
    "conso_refroidissement_ef",
    "conso_eclairage_ef",
    "conso_auxiliaires_ef",
    # Émissions GES (second critère de l'étiquette)
    "emission_ges_5_usages",
    "emission_ges_5_usages par_m2",
    "emission_ges_chauffage",
    "emission_ges_ecs",
    "emission_ges_refroidissement",
    "emission_ges_eclairage",
    "emission_ges_auxiliaires",
    # Étiquette GES (calculée simultanément)
    "etiquette_ges",
    # Consommations / émissions par énergie principale (n1)
    "conso_5 usages_ef_energie_n1",
    "conso_chauffage_ef_energie_n1",
    "conso_ecs_ef_energie_n1",
    "emission_ges_5_usages_energie_n1",
    "emission_ges_chauffage_energie_n1",
    "emission_ges_ecs_energie_n1",
    # Coûts énergétiques (déduits de la consommation)
    "cout_total_5_usages_energie_n1",
    "cout_chauffage_energie_n1",
    "cout_ecs_energie_n1",
    "cout_total_5_usages",
    "cout_chauffage",
    "cout_ecs",
    "cout_refroidissement",
    "cout_eclairage",
    "cout_auxiliaires",
]

# Départements DOM-TOM exclus (comme dans l'EDA)
DOM_TOM = ["971", "972", "973", "974", "988"]


def make_csv(data: pd.DataFrame, filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(filepath, index=False)
    print(f"  → {filepath}  ({len(data)} lignes × {data.shape[1]} colonnes)")


def main(input_path: Path, seed: int) -> None:

    # ── 1. Chargement ─────────────────────────────────────────────────────────
    print(f"Chargement de {input_path} ...")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"  {len(df)} lignes × {len(df.columns)} colonnes")

    # ── 2. Filtre France métropolitaine ───────────────────────────────────────
    before = len(df)
    df["code_departement_ban"] = df["code_departement_ban"].astype(str).str.strip()
    df = df[~df["code_departement_ban"].isin(DOM_TOM)]
    print(f"  Filtre DOM-TOM : {before - len(df)} lignes supprimées → {len(df)} restantes")

    # ── 3. Supprimer les lignes sans target ───────────────────────────────────
    before = len(df)
    df = df.dropna(subset=[TARGET])
    if before - len(df) > 0:
        print(f"  Suppression NaN target : {before - len(df)} lignes → {len(df)} restantes")

    # ── 4. Séparation features / target ───────────────────────────────────────
    cols_to_drop = set(META_COLS + LEAKY_COLS + [TARGET])
    feature_cols = [c for c in df.columns if c not in cols_to_drop]

    X = df[feature_cols].copy()
    y = df[[TARGET]].copy()

    print(f"\n  Features retenues : {len(feature_cols)} colonnes")
    print(f"  {feature_cols}")
    print(f"\n  Distribution de la target ({TARGET}) :")
    dist = y[TARGET].value_counts().sort_index()
    for label, count in dist.items():
        print(f"    {label} : {count:>6}  ({100*count/len(y):.1f}%)")

    # ── 5. Split stratifié ────────────────────────────────────────────────────
    # Stratification idéale : étiquette × région (comme l'EDA).
    # Chaque groupe doit avoir ≥ ceil(1 / test_size_1er_split) = 4 membres.
    # Si le dataset est trop petit, repli automatique sur étiquette seule.
    region = df["code_region_ban"].fillna(-1).astype(float).astype(int).astype(str)

    p_label  = y[TARGET].value_counts(normalize=True).min()
    p_region = df["code_region_ban"].value_counts(normalize=True).min()
    min_n_required = round(4 / (p_label * p_region))
    print(f"\n  N minimum pour stratification étiquette × région : ~{min_n_required:,} lignes")

    if len(df) >= min_n_required:
        print("  → Stratification : étiquette × région")
        strat = y[TARGET].astype(str) + "_" + region
    else:
        print(f"  → Dataset trop petit ({len(df):,} < {min_n_required:,})")
        print("     Repli sur : étiquette seule")
        print("     Pour la stratification complète, relancez fetch_dpe_data.py")
        print("     avec DATE_FILTER = Option B (date_derniere_modification en 2025, ~674k lignes)")
        strat = y[TARGET]

    # 70 % train | 15 % test | 15 % private_test
    X_train, X_rem, y_train, y_rem = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=strat
    )
    strat_rem = strat.loc[X_rem.index]
    X_test, X_priv, y_test, y_priv = train_test_split(
        X_rem, y_rem, test_size=0.50, random_state=seed, stratify=strat_rem
    )

    n_total = len(df)
    print(f"\n  Split (seed={seed}) :")
    print(f"    train        : {len(X_train):>6}  ({100*len(X_train)/n_total:.1f}%)")
    print(f"    test         : {len(X_test):>6}  ({100*len(X_test)/n_total:.1f}%)")
    print(f"    private_test : {len(X_priv):>6}  ({100*len(X_priv)/n_total:.1f}%)")

    # ── 6. Export ─────────────────────────────────────────────────────────────
    print("\nExport des fichiers ...")

    # Train — features ET labels (visible par les participants)
    make_csv(X_train, DATA_DIR / "train" / "train_features.csv")
    make_csv(y_train, DATA_DIR / "train" / "train_labels.csv")

    # Test — features visibles, labels en référence (leaderboard public)
    make_csv(X_test,  DATA_DIR  / "test"         / "test_features.csv")
    make_csv(y_test,  REF_DIR                    / "test_labels.csv")

    # Private test — features visibles, labels cachés (leaderboard privé)
    make_csv(X_priv,  DATA_DIR  / "private_test" / "private_test_features.csv")
    make_csv(y_priv,  REF_DIR                    / "private_test_labels.csv")

    print("\nDone. Pour tester le pipeline complet :")
    print("  python ingestion_program/ingestion.py")
    print("  python scoring_program/scoring.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prépare les données DPE pour Codabench")
    parser.add_argument(
        "--input", type=Path, default=Path("data/dpe_2025.csv"),
        help="Chemin du CSV source (défaut : data/dpe_2025.csv)",
    )
    parser.add_argument(
        "--seed", type=int, default=123,
        help="Graine aléatoire (défaut : 123, aligné avec l'EDA)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Fichier source introuvable : {args.input}\n"
            "Lancez d'abord : python tools/fetch_dpe_data.py"
        )

    main(args.input, args.seed)
