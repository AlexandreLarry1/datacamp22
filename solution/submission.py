from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer, OrdinalEncoder, StandardScaler,
)


# ── Feature definitions ──────────────────────────────────────────────────────

# Numerical features: physical characteristics, energy metrics, coordinates…
NUMERICAL_FEATURES = [
    "annee_construction",
    "hauteur_sous_plafond",
    "nombre_niveau_logement",
    "surface_habitable_logement",
    "coordonnee_cartographique_x_ban",
    "coordonnee_cartographique_y_ban",
    "code_postal_brut",
    "deperditions_enveloppe",
    "deperditions_ponts_thermiques",
    "deperditions_murs",
    "deperditions_planchers_hauts",
    "deperditions_planchers_bas",
    "deperditions_portes",
    "deperditions_baies_vitrees",
    "deperditions_renouvellement_air",
    "besoin_chauffage",
    "besoin_ecs",
    "besoin_refroidissement",
    "apport_interne_saison_chauffe",
    "apport_interne_saison_froide",
    "apport_solaire_saison_chauffe",
    "apport_solaire_saison_froide",
]

# Categorical features: building type, heating system, location…
CATEGORICAL_FEATURES = [
    "type_batiment",
    "periode_construction",
    "type_installation_chauffage",
    "type_installation_ecs",
    "classe_inertie_batiment",
    "classe_altitude",
    "zone_climatique",
    "code_departement_ban",
    "code_region_ban",
    "qualite_isolation_enveloppe",
    "qualite_isolation_menuiseries",
    "type_energie_n1",
    "type_energie_principale_chauffage",
    "type_energie_principale_ecs",
]


def get_model():
    """Return a scikit-learn compatible pipeline for DPE label prediction.

    The pipeline follows the following pattern :
      - A ColumnTransformer dispatches columns to specialised sub-pipelines.
      - Each sub-pipeline handles imputation then encoding/scaling.
      - A final classifier operates on the concatenated output.
    """

    # Sub-pipeline for numerical columns: impute missing → scale
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Sub-pipeline for categorical columns:
    #   cast to str (handles NaN-induced float columns like code_region_ban)
    #   → impute missing → ordinal encode
    categorical_pipeline = Pipeline([
        ("to_str", FunctionTransformer(
            lambda X: X.astype(str), feature_names_out="one-to-one",
        )),
        ("imputer", SimpleImputer(strategy="constant",
                                  fill_value="missing")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value",
                                   unknown_value=-1)),
    ])

    # Combine both sub-pipelines in a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    # Full pipeline: preprocessing → classifier
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1,
        )),
    ])
