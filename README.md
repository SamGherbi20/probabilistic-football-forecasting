# project_eleven — Phase 2 : prédiction de résultats LaLiga

Prédiction de l'issue des matchs de première division espagnole (**victoire à
domicile / nul / défaite**) par apprentissage automatique, à partir de données
réelles de matchs. Ce dépôt constitue la deuxième phase de `project_eleven`,
qui appliquait la même démarche à des données issues du jeu *Football Manager
2024* ; on passe ici aux données réelles.

## Objectif

À partir de l'historique des matchs de LaLiga (saisons 2020‑2021 à 2025‑2026),
construire des **features disponibles avant le coup d'envoi** (niveau Elo, forme
récente, xG, repos), puis entraîner et évaluer un modèle de classification à
trois classes. L'accent est mis sur une évaluation **honnête et reproductible** :
split temporel, comparaison systématique à des baselines, et absence de fuite de
données du futur vers le passé.

## Données

Les matchs proviennent de [fbref.com](https://fbref.com) (données Opta). Le
dépôt versionne, dans `data/`, un CSV de calendrier par saison — une ligne par
match joué, avec les colonnes d'origine (date, équipes, xG domicile/extérieur,
score, etc.).

> **Note.** Ces fichiers sont réutilisés à des fins strictement pédagogiques.
> Les pages HTML brutes ne sont pas redistribuées ; se référer aux conditions
> d'utilisation de fbref pour tout autre usage. La saison 2025‑2026 est partielle
> et ne comporte pas de colonne xG sur fbref.

## Structure du projet

```
project_eleven/
├── data/                                  # Données brutes VERSIONNÉES (entrée)
│   ├── laliga_2020_2021_fixtures.csv      #   un CSV de calendrier par saison
│   ├── laliga_2021_2022_fixtures.csv      #   (colonnes fbref d'origine)
│   ├── laliga_2022_2023_fixtures.csv
│   ├── laliga_2023_2024_fixtures.csv
│   ├── laliga_2024_2025_fixtures.csv
│   └── laliga_2025_2026_fixtures.csv
├── src/                                    # Code réutilisable, importable
│   ├── __init__.py
│   └── features.py                        #   build_pre_match_features()
├── notebook_project_eleven_phase_two.ipynb # Analyse : pipeline complet + résultats
├── requirements.txt                        # Dépendances épinglées
├── .gitignore
└── README.md
```

Ce qui va dans chaque dossier :

- **`data/`** — uniquement les 6 CSV bruts (entrées du pipeline). Rien d'autre :
  les fichiers intermédiaires produits par le notebook (`laliga_matchs_20-26.csv`,
  `liga_pre_match_features.csv`) sont **régénérés** à chaque exécution et
  volontairement ignorés par Git (voir `.gitignore`).
- **`src/`** — le code qu'on veut tester, réutiliser et importer sans le
  dupliquer. Aujourd'hui : le feature engineering (`features.py`). Le notebook
  importe cette fonction plutôt que d'en recopier la logique.
- **racine** — le notebook d'analyse, la configuration du projet
  (`requirements.txt`, `.gitignore`) et ce README.

> Pour un projet qui grossirait, on scinderait `data/` en `data/raw/` (versionné)
> et `data/processed/` (généré, ignoré), et on déplacerait le notebook dans un
> dossier `notebooks/`. Avec un seul notebook, le garder à la racine évite les
> acrobaties d'import et reste parfaitement lisible.

## Installation & exécution

```bash
# 1. Cloner puis se placer dans le dépôt
git clone <url-du-dépôt> && cd project_eleven

# 2. Environnement virtuel + dépendances
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt

# 3. Lancer le notebook
jupyter lab notebook_project_eleven_phase_two.ipynb
```

Le notebook est conçu pour tourner d'un bloc : **Kernel → Restart & Run All**
régénère l'intégralité des fichiers intermédiaires et des résultats à partir des
seuls CSV de `data/`.

## Méthodologie

### Features construites (avant match)

| Feature | Description |
|---|---|
| `elo_home`, `elo_away` | Notation Elo de chaque équipe, mise à jour match après match |
| `elo_diff` | Différence d'Elo domicile − extérieur |
| `form_points_home_5`, `form_points_away_5` | Points moyens sur les 5 derniers matchs (victoire 3, nul 1.5, défaite 0) |
| `form_goal_diff_home_5`, `form_goal_diff_away_5` | Goal difference moyen sur les 5 derniers matchs |
| `form_xG_home_5`, `form_xG_away_5` | xG moyen sur les 5 derniers matchs |
| `xG_diff` | Différence des xG moyens domicile − extérieur |
| `rest_diff` | Différence de jours de repos entre les deux équipes |
| `day` | Jour de la semaine du match (encodé en entier) |

Toutes ces features sont calculées **avant** la mise à jour de l'Elo et de
l'historique du match courant : un match n'utilise jamais d'information issue de
lui-même ou du futur (pas de fuite temporelle).

### Évaluation

- **Split temporel** : entraînement sur les 3 premières saisons (2020‑2021 à
  2022‑2023, 1140 matchs), test sur les 2 suivantes (2023‑2024 et 2024‑2025,
  760 matchs). La saison 2025‑2026 (partielle, sans xG) est mise de côté.
- **Modèle** : régression logistique multinomiale, dans un `Pipeline`
  (imputation médiane → normalisation → classifieur) pour que le prétraitement
  soit appris sur le seul jeu d'entraînement.
- **Baselines** : (1) *prior de classe* — toujours prédire l'issue la plus
  fréquente (« domicile ») ; (2) *Elo seul* — même modèle sur la seule feature
  `elo_diff`.
- **Métriques** : accuracy, log‑loss, matrice de confusion.

Le facteur `K` de l'Elo est réglé par balayage (courbes accuracy / log‑loss sur
le même jeu de test), qui confirme `K = 20`.

## Résultats

Sur les 2 saisons de test (760 matchs) :

| Modèle | Accuracy | Log‑loss |
|---|---:|---:|
| Régression logistique (toutes features) | 0.536 | 0.974 |
| Baseline — prior (toujours domicile) | 0.442 | 1.073 |
| Baseline — Elo seul (`elo_diff`) | 0.536 | 0.967 |

Deux enseignements, assumés :

1. **Le modèle bat nettement le prior** (53.6 % vs 44.2 %) : il apprend donc un
   signal réel. **Mais l'Elo seul fait aussi bien**, et même légèrement mieux en
   log‑loss. Les features de forme, xG et repos n'apportent quasiment rien
   au‑delà de la différence de niveau (léger sur‑apprentissage du modèle
   complet). L'analyse des coefficients le confirme : `elo_diff`, `elo_away` et
   `elo_home` dominent.
2. **Le modèle ne prédit presque jamais le nul** (rappel ≈ 0 sur cette classe).
   C'est le problème classique des matchs nuls en prédiction football, et le
   principal axe d'amélioration.

## Limites connues

- **Démarrage à froid (cold start).** Pour les 5 premiers matchs d'une équipe,
  les features de forme sont indisponibles et renvoient 0 (moyenne sur un
  historique vide). Ces lignes de début d'historique sont donc dégradées. Une
  amélioration consisterait à les exclure ou à initialiser la forme par un prior.
- **Elo sans remise à zéro inter‑saison.** L'Elo court en continu sur toutes les
  saisons ; les équipes promues héritent de l'Elo de base (1500) sans régression
  vers la moyenne entre saisons.
- **Saison 2025‑2026 écartée** de l'évaluation : partielle et sans colonne xG sur
  fbref (ses xG seraient imputés, ce qui dégraderait le signal).

## Feuille de route (phase 3)

- **Backtest walk‑forward** : réentraîner saison par saison plutôt qu'un unique
  split, pour une estimation plus robuste et réaliste.
- **Traiter les nuls** : modèle ordinal, pondération des classes, ou modèle de
  nul dédié (type Davidson sur l'Elo).
- **Calibration des probabilités** : score de Brier, courbe de fiabilité.
- **Comparaison aux cotes des bookmakers** : le vrai test de valeur d'un modèle
  de prédiction sportive (edge vs marché).
