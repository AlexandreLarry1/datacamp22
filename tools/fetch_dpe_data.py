"""
Récupère un extrait du dataset DPE Logements existants via l'API ADEME.
Source : https://data.ademe.fr/datasets/dpe03existant

Filtres appliqués :
  - date_derniere_modification_dpe <= 2025-12-31
  - date_etablissement_dpe entre 2025-01-01 et 2025-12-31

Usage :
    python tools/fetch_dpe_data.py              # 100 000 lignes par défaut
    python tools/fetch_dpe_data.py --n 20000    # nombre personnalisé
"""

import argparse
import os
import time
import urllib.parse

import pandas as pd
import requests

API_URL = "https://data.ademe.fr/data-fair/api/v1/datasets/dpe03existant/lines"

# Colonnes utilisées dans l'EDA (0% de valeurs manquantes sur le dataset 2025)
# Note : certains noms contiennent des espaces (ex. "conso_5 usages_ef") — ils sont
# inclus dans `select` et filtrés côté client si l'API ne les renvoie pas.
SELECTED_COLUMNS = [
    "numero_dpe",
    "date_derniere_modification_dpe",
    "date_etablissement_dpe",
    "modele_dpe",
    "version_dpe",
    "methode_application_dpe",
    "etiquette_dpe",
    "etiquette_ges",
    "type_batiment",
    "annee_construction",
    "periode_construction",
    "type_installation_chauffage",
    "type_installation_ecs",
    "hauteur_sous_plafond",
    "nombre_niveau_logement",
    "surface_habitable_logement",
    "classe_inertie_batiment",
    "classe_altitude",
    "zone_climatique",
    "code_departement_ban",
    "code_region_ban",
    "identifiant_ban",
    "coordonnee_cartographique_x_ban",
    "coordonnee_cartographique_y_ban",
    "score_ban",
    "statut_geocodage",
    "code_postal_brut",
    "deperditions_enveloppe",
    "deperditions_ponts_thermiques",
    "deperditions_murs",
    "deperditions_planchers_hauts",
    "deperditions_planchers_bas",
    "deperditions_portes",
    "deperditions_baies_vitrees",
    "deperditions_renouvellement_air",
    "qualite_isolation_enveloppe",
    "qualite_isolation_menuiseries",
    "besoin_chauffage",
    "besoin_ecs",
    "besoin_refroidissement",
    "apport_interne_saison_chauffe",
    "apport_interne_saison_froide",
    "apport_solaire_saison_chauffe",
    "apport_solaire_saison_froide",
    "conso_5_usages_ep",
    "conso_5_usages_par_m2_ep",
    "conso_chauffage_ep",
    "conso_ecs_ep",
    "conso_refroidissement_ep",
    "conso_eclairage_ep",
    "conso_auxiliaires_ep",
    "conso_5 usages_ef",
    "conso_5 usages_par_m2_ef",
    "conso_chauffage_ef",
    "conso_ecs_ef",
    "conso_refroidissement_ef",
    "conso_eclairage_ef",
    "conso_auxiliaires_ef",
    "emission_ges_5_usages",
    "emission_ges_5_usages par_m2",
    "emission_ges_chauffage",
    "emission_ges_ecs",
    "emission_ges_refroidissement",
    "emission_ges_eclairage",
    "emission_ges_auxiliaires",
    "type_energie_n1",
    "conso_5 usages_ef_energie_n1",
    "conso_chauffage_ef_energie_n1",
    "conso_ecs_ef_energie_n1",
    "cout_total_5_usages_energie_n1",
    "cout_chauffage_energie_n1",
    "cout_ecs_energie_n1",
    "emission_ges_5_usages_energie_n1",
    "emission_ges_chauffage_energie_n1",
    "emission_ges_ecs_energie_n1",
    "cout_total_5_usages",
    "cout_chauffage",
    "cout_ecs",
    "cout_refroidissement",
    "cout_eclairage",
    "cout_auxiliaires",
    "type_energie_principale_chauffage",
    "type_energie_principale_ecs",
]

# Filtre de dates via syntaxe Lucene (paramètre qs de l'API Data Fair)
#
# Deux options selon la taille de dataset souhaitée :
#
#   Option A — ~19k lignes  : DPEs établis ET modifiés en 2025
#     "date_derniere_modification_dpe:[2000-01-01 TO 2025-12-31]"
#     " AND date_etablissement_dpe:[2025-01-01 TO 2025-12-31]"
#
#   Option B — ~674k lignes : tous les DPEs mis à jour en 2025 (= filtre EDA)
#     "date_derniere_modification_dpe:[2025-01-01 TO 2025-12-31]"
#
# L'Option B est nécessaire pour que la stratification étiquette × région
# fonctionne (chaque groupe doit avoir ≥ 2 membres).
DATE_FILTER = "date_derniere_modification_dpe:[2025-01-01 TO 2025-12-31]"

PAGE_SIZE = 10_000  # max autorisé par l'API

# L'API rejette les noms de colonnes avec des espaces dans le paramètre select.
# On envoie uniquement les colonnes sans espaces ; les autres sont filtrées côté client.
_COLUMNS_FOR_SELECT = [c for c in SELECTED_COLUMNS if " " not in c]


def fetch_page(after: str | None, size: int) -> tuple[list[dict], str | None]:
    """Récupère une page via pagination par curseur (after + sort).

    L'API Data Fair limite size+skip <= 10 000 ; le paramètre `after` permet
    de paginer sans skip en reprenant après la dernière valeur du champ de tri.
    """
    params = {
        "select": ",".join(_COLUMNS_FOR_SELECT),
        "size": min(size, PAGE_SIZE),
        "sort": "numero_dpe",
        "qs": DATE_FILTER,
    }
    if after:
        params["after"] = after
    resp = requests.get(API_URL, params=params, timeout=60)
    if not resp.ok:
        print(f"  Erreur API ({resp.status_code}) : {resp.text[:500]}")
    resp.raise_for_status()
    data = resp.json()
    rows = data["results"]
    total = data.get("total")
    if total is not None and after is None:   # afficher uniquement sur la 1re page
        print(f"  Total disponible avec ce filtre : {total:,} lignes")
    # L'API renvoie une URL "next" contenant le bon paramètre `after` encodé.
    next_url = data.get("next")
    if next_url:
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(next_url).query)
        next_after = qs.get("after", [None])[0]
    else:
        next_after = None
    return rows, next_after


def fetch_dpe_data(n_rows: int) -> pd.DataFrame:
    """Récupère n_rows lignes en paginant par curseur."""
    all_rows: list[dict] = []
    after: str | None = None

    while len(all_rows) < n_rows:
        remaining = n_rows - len(all_rows)
        size = min(remaining, PAGE_SIZE)

        print(f"  Requête API : after={after!r}, size={size} ...")
        rows, after = fetch_page(after, size)

        if not rows:
            print("  Plus de données disponibles.")
            break

        all_rows.extend(rows)

        if not after:
            break

        if len(all_rows) < n_rows:
            time.sleep(1)

    df = pd.DataFrame(all_rows)

    # Garder uniquement les colonnes attendues (les colonnes avec espaces
    # ne sont pas demandées via select mais peuvent être présentes)
    cols_present = [c for c in SELECTED_COLUMNS if c in df.columns]
    df = df[cols_present]

    print(f"\n  Total récupéré : {len(df)} lignes, {len(df.columns)} colonnes")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage basique : suppression des lignes sans target ou surface."""
    initial = len(df)

    df = df.dropna(subset=["conso_5_usages_ep", "surface_habitable_logement"])
    df = df[df["conso_5_usages_ep"] > 0]
    df = df[df["surface_habitable_logement"] > 0]
    # Filtrer les valeurs aberrantes (le DPE va de ~0 à ~800 kWh EP/m²/an)
    df = df[df["conso_5_usages_par_m2_ep"] <= 1000]

    dropped = initial - len(df)
    if dropped > 0:
        print(f"  Nettoyage : {dropped} lignes supprimées ({initial} → {len(df)})")

    return df.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Fetch DPE data from ADEME API")
    parser.add_argument(
        "--n", type=int, default=100_000,
        help="Nombre de lignes à récupérer (défaut : 100 000)",
    )
    parser.add_argument(
        "--output", type=str, default="data/dpe_2025.csv",
        help="Chemin du fichier de sortie",
    )
    parser.add_argument("--no-clean", action="store_true", help="Ne pas nettoyer les données")
    args = parser.parse_args()

    print(f"Récupération de {args.n} lignes depuis l'API ADEME DPE...")
    print(f"Filtre : {DATE_FILTER}\n")
    df = fetch_dpe_data(args.n)

    if df.empty:
        print("Aucune donnée récupérée. Vérifiez votre connexion.")
        return

    if not args.no_clean:
        print("\nNettoyage des données...")
        df = clean_data(df)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nFichier sauvegardé : {args.output}")
    print(f"  {len(df)} lignes × {len(df.columns)} colonnes")
    print("\nAperçu des colonnes :")
    print(df.dtypes.to_string())
    print("\nTarget — conso_5_usages_par_m2_ep (kWh EP/m²/an) :")
    print(df["conso_5_usages_par_m2_ep"].describe().to_string())


if __name__ == "__main__":
    main()
