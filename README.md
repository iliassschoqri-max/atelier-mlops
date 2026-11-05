# Atelier MLOps - Git + DVC + MLflow (Dataset Iris)

Projet complet et reproductible de classification du dataset Iris avec
Scikit-learn, versionne avec **Git** et **DVC**, et suivi avec **MLflow**.

## Objectifs

- Versionner le code avec Git.
- Versionner les donnees avec DVC.
- Construire un pipeline ML reproductible (`dvc repro`).
- Suivre les experimentations avec MLflow (parametres, metriques, artefacts).
- Comparer et restaurer des versions du projet.

## Prerequis

- Python 3.10+
- Git
- Un compte GitHub (pour la publication)

```bash
python --version
git --version
dvc --version
mlflow --version
```

## Installation

```bash
git clone <URL_DU_DEPOT>
cd atelier-mlops
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Structure du projet

```
atelier-mlops/
├── .github/
│   └── workflows/
│       └── mlops.yml
├── .dvc/
│   └── config
├── data/
│   └── raw/
│       ├── iris.csv          # suivi par DVC (non versionne par Git)
│       ├── iris.csv.dvc      # reference DVC versionnee par Git
│       └── .gitignore
├── src/
│   ├── prepare_data.py
│   ├── train.py              # Random Forest + MLflow
│   └── train_knn.py          # KNN + MLflow (comparaison)
├── models/
│   ├── model.pkl             # sortie DVC (ignoree par Git)
│   └── model_knn.pkl
├── reports/
│   ├── metrics.json          # metriques DVC (cache: false)
│   └── metrics_knn.json
├── params.yaml
├── dvc.yaml
├── dvc.lock
├── requirements.txt
└── .gitignore
```

## Demarrage rapide

```bash
# 1. Generer le dataset brut (si absent)
python src/prepare_data.py

# 2. Recuperer les donnees versionnees par DVC
dvc pull

# 3. Executer le pipeline complet
dvc repro

# 4. Afficher les metriques
dvc metrics show

# 5. Ouvrir l'interface MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db
# puis http://localhost:5000
```

## Etape 1 - Git

```bash
git init -b main
git config user.name "Etudiant MLOps"
git config user.email "etudiant@example.com"
git add .
git commit -m "Initialisation du projet Iris"

git switch -c develop
# ... modifications ...
git add .
git commit -m "Ajout de la preparation du dataset"
```

Le fichier `.gitignore` exclut les environnements virtuels, le cache Python,
la base MLflow (`mlflow.db`, `mlruns/`) et les modeles (`models/`).

## Etape 2 - DVC : versioning des donnees

```bash
dvc init
git add .dvc .dvcignore
git commit -m "Initialisation de DVC"

# Si le fichier a deja ete commite par erreur dans Git :
git rm --cached data/raw/iris.csv
dvc add data/raw/iris.csv
git add data/raw/iris.csv.dvc data/raw/.gitignore
git commit -m "Versioning du dataset Iris avec DVC"
```

### Remote DVC local (demonstration)

```bash
mkdir ../dvc-storage
dvc remote add -d storage ../dvc-storage
dvc remote list
dvc push
```

En production, utiliser un stockage partage (S3, Azure Blob, GCS, etc.).

## Etape 3 - MLflow : tracking

Les scripts `src/train.py` et `src/train_knn.py` :

- lisent les hyperparametres depuis `params.yaml` ;
- entrainent le modele et calculent Accuracy, Precision, Recall, F1-score ;
- enregistrent parametres, metriques et modele dans MLflow ;
- ecrivent `reports/metrics.json` (et `metrics_knn.json`).

```bash
python src/train.py
python src/train_knn.py
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Etape 4 - Pipeline DVC

`dvc.yaml` declare l'etape `train` : commande, dependances, parametres,
sorties et metriques. `dvc.lock` est genere par DVC et versionne par Git.

```bash
dvc repro
dvc dag
dvc metrics show
```

## Resultats compares

Experiences enregistrees dans MLflow (experiment `TP_Iris_Classification`) :

| Modele | Parametres | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|---|
| RandomForest | `n_estimators=100, max_depth=5` | 0.9333 | 0.9333 | 0.9333 | 0.9333 |
| RandomForest | `n_estimators=200, max_depth=8` | 0.9000 | 0.9024 | 0.9000 | 0.8997 |
| KNN | `n_neighbors=5` | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Conclusion : augmenter le nombre d'arbres n'ameliore pas forcement le modele.
KNN obtient ici de meilleurs resultats sur ce decoupage particulier.

## Comparer et restaurer une version

```bash
git tag -a v1.0 -m "Premiere version du modele Iris"
git tag -a v2.0 -m "Deuxieme version du modele Iris"
git tag

# Restauration de v1.0
git checkout v1.0
dvc checkout
dvc pull        # si le cache local ne contient pas les donnees
dvc repro       # regenere les sorties non cachees (ex: metriques)

# Retour sur la branche principale
git switch main
dvc checkout
```

## Reproduire sur une autre machine

```bash
git clone <URL_DU_DEPOT>
cd atelier-mlops
pip install -r requirements.txt

# Configurer le remote DVC si necessaire, puis :
dvc pull
dvc repro

mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Integration continue (GitHub Actions)

Le workflow `.github/workflows/mlops.yml` installe les dependances, recupere
les donnees (`dvc pull`), execute le pipeline (`dvc repro`) et affiche les
metriques. En CI reelle, configurer les identifiants du remote DVC via les
secrets GitHub et installer le backend de stockage.

## Reponses aux questions de reflexion

1. **Git vs DVC** : Git versionne le code et les fichiers texte ; DVC versionne
   les donnees volumineuses et les artefacts via des fichiers `.dvc` et un cache.
2. **Dataset volumineux dans Git** : Git conserve tout l'historique binaire,
   ce qui alourdit le depot et degrade les performances.
3. **`dvc add` vs `dvc repro`** : `dvc add` met un fichier sous suivi DVC ;
   `dvc repro` execute les etapes du pipeline lorsque dependances/parametres
   changent.
4. **`dvc.lock`** : enregistre les empreintes resolvees des dependances,
   parametres et sorties pour garantir la reproductibilite.
5. **Comparaison MLflow** : l'interface MLflow compare les runs (parametres,
   metriques, artefacts) cote a cote.
6. **Retrouver donnees et modele d'un commit** : `git checkout <commit>` puis
   `dvc checkout` (et `dvc pull` si necessaire).
7. **Reproductibilite** : elle garantit des resultats verifiables, auditables et
   comparables, indispensables en production.
8. **DVC vs MLflow pour les modeles** : DVC versionne les fichiers de modele ;
   MLflow suit les runs, les metriques et gere le cycle de vie des modeles.

## Livrables

- Depot GitHub du projet.
- `.dvc/`, `dvc.yaml`, `dvc.lock`, `params.yaml`.
- Scripts Python (`src/prepare_data.py`, `src/train.py`, `src/train_knn.py`).
- Captures d'ecran MLflow.
- Tableau comparatif de deux experiences ou plus.
- Demonstration de restauration (`git checkout` + `dvc checkout`).
