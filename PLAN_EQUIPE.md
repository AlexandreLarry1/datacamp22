# Plan de travail — Compétition Codabench

## Résumé du projet

Ce projet est un template pour héberger une compétition de Machine Learning sur Codabench. Les participants soumettent un modèle via une fonction `get_model()`, et le système l'entraîne, génère des prédictions, puis calcule un score automatiquement.

---

## Répartition de l'équipe

| Membre | Rôle | Fichiers principaux |
|--------|------|---------------------|
| **Personne A** | Ingestion — Données | `tools/setup_data.py`, `dev_phase/`, `ingestion_program/bench_utils/` |
| **Personne B** | Ingestion — Pipeline | `ingestion_program/ingestion.py`, `solution/submission.py` |
| **Personne C** | Scoring — Évaluation | `scoring_program/scoring.py` |
| **Personne D** | Scoring — Déploiement & Pages | `competition.yaml`, `pages/`, `tools/create_bundle.py`, `template_starting_kit.ipynb` |

---

## Étapes à suivre

### Phase 1 — Cadrage (ensemble)

- [ ] Choisir le problème ML (classification, régression, etc.)
- [ ] Choisir le jeu de données réel
- [ ] Définir la/les métrique(s) d'évaluation (accuracy, F1, RMSE, etc.)
- [ ] Se mettre d'accord sur le format des données (colonnes, types, noms de fichiers)

---

### Phase 2 — Ingestion (Personnes A & B)

#### Personne A — Données

- [ ] Remplacer le dataset dummy dans `tools/setup_data.py` par le vrai jeu de données
- [ ] Définir les splits : train / test / private_test
- [ ] Générer les CSV dans `dev_phase/input_data/` et `dev_phase/reference_data/`
- [ ] Mettre à jour la constante `N_SAMPLES` dans `ingestion_program/bench_utils/__init__.py`
- [ ] Vérifier que les fichiers générés sont cohérents (pas de fuite de labels dans les données de test)

#### Personne B — Pipeline d'entraînement

- [ ] Adapter `ingestion_program/ingestion.py` au nouveau format de données si nécessaire
- [ ] Vérifier que `get_train_data()` et `evaluate_model()` fonctionnent avec le vrai dataset
- [ ] Créer une solution de référence (baseline) dans `solution/submission.py`
- [ ] Tester le pipeline d'ingestion localement : `python ingestion_program/ingestion.py`
- [ ] S'assurer que les fichiers de prédictions sont correctement générés

---

### Phase 3 — Scoring (Personnes C & D)

#### Personne C — Évaluation

- [ ] Adapter `scoring_program/scoring.py` à la métrique choisie
- [ ] Ajouter des métriques supplémentaires si nécessaire (précision, rappel, etc.)
- [ ] Vérifier que `scores.json` contient toutes les colonnes attendues par `competition.yaml`
- [ ] Tester le scoring localement avec les prédictions générées par l'ingestion
- [ ] Gérer les cas d'erreur (prédictions manquantes, format incorrect)

#### Personne D — Déploiement & Documentation

- [ ] Mettre à jour `competition.yaml` (titre, description, dates, colonnes du leaderboard)
- [ ] Rédiger les pages de la compétition :
  - `pages/participate.md` — Instructions de soumission
  - `pages/seed.md` — Template du modèle
  - `pages/timeline.md` — Calendrier des phases
  - `pages/terms.md` — Conditions d'utilisation
- [ ] Compléter le notebook `template_starting_kit.ipynb` (EDA, baseline, guide)
- [ ] Mettre à jour `requirements.txt` avec les dépendances nécessaires
- [ ] Vérifier le `Dockerfile` dans `tools/`

---

### Phase 4 — Intégration & Tests (ensemble)

- [ ] Lancer le pipeline complet localement :
  ```bash
  python tools/setup_data.py
  python ingestion_program/ingestion.py
  python scoring_program/scoring.py
  ```
- [ ] Tester via Docker avec `python tools/run_docker.py`
- [ ] Vérifier que le CI passe sur GitHub Actions
- [ ] Générer le bundle : `python tools/create_bundle.py`
- [ ] Uploader le bundle sur Codabench
- [ ] Faire une soumission test sur Codabench et vérifier le score

---

### Phase 5 — Finalisation (ensemble)

- [ ] Relire toutes les pages et le notebook
- [ ] Valider que le leaderboard affiche les bonnes colonnes
- [ ] Tester avec 2-3 modèles différents pour vérifier la robustesse
- [ ] Ouvrir la compétition aux participants

---

## Dépendances entre les tâches

```
Phase 1 (Cadrage)
    │
    ├── Phase 2 : Ingestion (A & B travaillent en parallèle)
    │       │
    │       ▼
    ├── Phase 3 : Scoring (C & D travaillent en parallèle)
    │       │     C a besoin des prédictions de B pour tester
    │       │     D peut commencer les pages dès la Phase 1
    │       ▼
    Phase 4 : Intégration (tout le monde)
        │
        ▼
    Phase 5 : Finalisation (tout le monde)
```

> **Note** : La Personne D peut commencer la rédaction des pages et la config dès la Phase 1, sans attendre la fin de l'ingestion.
