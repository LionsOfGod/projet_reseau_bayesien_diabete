# Réseaux Bayésiens pour la Décision Clinique Interprétable

**Application au diagnostic probabiliste du diabète**

## 📋 Projet de fin de cycle Licence Informatique

- **Université** : Aix-Marseille Université (AMU) - Formation EAD
- **Parcours** : Licence L3 Informatique - Mathématiques-Informatique
- **Encadrant** : Pr Raquel Urena
- **Année** : 2025-2026

---

## 📖 Objectif du Projet

Concevoir et implémenter un **réseau bayésien** entièrement from Scratch pour :

1.  **Identifier les variables cliniques pertinentes** pour le diagnostic du diabète
2.  **Étudier les dépendances probabilistes** entre les variables
3.  **Construire la morphologie du réseau bayésien** (DAG)
4.  **Estimer les probabilités conditionnelles** à partir des données
5.  **Implémenter en Python pur** (AUCUNE librairie ML spécialisée)
6.  **Réaliser des inférences probabilistes** pour les prédictions
7.  **Évaluer les performances** avec métriques standards
8.  **Interpréter les résultats** de manière explicable

### Consignes Importantes

-  **PAS d'import** : scikit-learn, TensorFlow, PyTorch, pgmpy, etc.
-  **Implémentation manuelle** de tous les algorithmes
-  **Python** avec stdlib uniquement

---

## 📁 Structure du Projet

```
reseau-bayesien-diabete/
│
├── src/                          # Code source principal
│   ├── __init__.py
│   ├── preprocessing.py          # Chargement, nettoyage, discrétisation
│   ├── network.py                # Définition du DAG
│   ├── cpt.py                    # Estimation des CPT (lissage Laplace)
│   ├── inference.py              # Inférence probabiliste
│   ├── data_split.py             # Train/test split
│   ├── evaluation.py             # Métriques (confusion matrix, F1, ROC-AUC)
│   └── main.py                   # Pipeline principal
│
├── tests/                        # Tests unitaires
│   ├── test_cpt.py
│   ├── test_inference.py
│   ├── test_evaluation.py
│   └── test_data_split.py
│
├── data/                         # Données
│   └── Pima_Diabetes_organise.csv
│
├── README.md                     # Ce fichier
├── requirements.txt              # Dépendances
└── .gitignore
```

---

## 🚀 Installation et Utilisation

### 1. Cloner le projet

```bash
git clone https://github.com/HenriAYAMA/reseau-bayesien-diabete.git
cd reseau-bayesien-diabete
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Exécuter le pipeline complet

```bash
python src/main.py
```

**Sortie attendue** :

```
======================================================================
RESEAU BAYESIEN POUR LE DIAGNOSTIC DU DIABETE
======================================================================

ETAPE 1: PREPROCESSING DES DONNEES
--------
✓ Dataset chargé: 768 observations
✓ Valeurs manquantes détectées (0 → NaN)
✓ Imputation par médiane effectuée
✓ Prétraitement terminé

ETAPE 2: DEFINITION DE LA STRUCTURE DU RESEAU
--------
============================================================
STRUCTURE DU RESEAU BAYESIEN
============================================================
...

ETAPE 3: PARTITIONNEMENT TRAIN/TEST
--------
✓ Données d'entraînement: 614
✓ Données de test: 154

...

RAPPORT D'EVALUATION
============================================================
MATRICE DE CONFUSION:
  TP (Vrais Positifs):  45
  FP (Faux Positifs):   18
  TN (Vrais Négatifs):  82
  FN (Faux Négatifs):   9

METRIQUES DE PERFORMANCE:
  Exactitude (Accuracy):  0.8247
  Précision (Precision):  0.7143
  Rappel (Recall):        0.8333
  Spécificité:            0.8200
  F1-Score:               0.7692
  ROC-AUC:                0.8267
============================================================
```

---

## 📊 Structure du Réseau Bayésien

### Nœuds (Variables)

```
Pregnancies → Age
    ↓        ↓
SkinThickness    BMI
    ↓            ↓
  Insulin ----→ Outcome ← Glucose
    ↑
    └─ Glucose
```

### Variables Cliniques

| Variable | Description | États |
|----------|-------------|-------|
| **Glucose** | Concentration glucose (jeûn) | Faible, Intermédiaire, Élevé |
| **BloodPressure** | Tension artérielle diastolique | Faible, Normal, Élevé |
| **BMI** | Indice de masse corporelle | Faible, Normal, Élevé |
| **Insulin** | Insulinémie de jeûn | Faible, Normal, Élevé |
| **Age** | Âge | Jeune, Intermédiaire, Élevé |
| **DiabetesPedigreeFunction** | Historique familial | Faible, Intermédiaire, Élevé |
| **SkinThickness** | Épaisseur du pli triceps | Faible, Normal, Élevé |
| **Pregnancies** | Nombre de grossesses | Faible, Intermédiaire, Élevé |
| **Outcome** | Diagnostic | 0 (Non diabétique), 1 (Diabétique) |

---

## 🔧 Modules Clés

### 1. **preprocessing.py**
- Chargement du dataset Pima Indians Diabetes
- Gestion des valeurs manquantes (médiane)
- Discrétisation en 3 catégories (cliniquement justifiées)

```python
preprocessor = DataPreprocessor()
data = preprocessor.preprocess("data/Pima_Diabetes_organise.csv")
```

### 2. **network.py**
- Définition du DAG (9 nœuds, 8 arcs)
- Vérification d'acyclicité
- Tri topologique

```python
network = BayesianNetwork()
print(network.topological_sort())  # Ordre causal
```

### 3. **cpt.py**
- Estimation des Tables de Probabilités Conditionnelles
- **Lissage de Laplace** : P(X|Y) = (count + 1) / (total + k)
- Évite les zéros probabilistes

```python
estimator = CPTEstimator(laplace_smoothing=True)
cpts = estimator.build_all_cpts(train_data, NODES, EDGES)
```

### 4. **inference.py**
- Calcul de probabilités jointes : P(obs) = ∏ P(Xi | Parents(Xi))
- Inférence postérieure : P(Outcome|obs) avec Bayes
- Explications locales des prédictions

```python
inference = BayesianInference(estimator, network)
pred, probs = inference.predict(observation)
explanation = inference.explain_prediction(observation)
```

### 5. **evaluation.py**
- **Matrice de confusion** : TP, TN, FP, FN
- **Précision** : TP / (TP + FP)
- **Rappel** : TP / (TP + FN)
- **F1-Score** : 2 * (P * R) / (P + R)
- **ROC-AUC** : Aire sous la courbe

```python
evaluator = Evaluator()
evaluator.evaluate(predictions, true_labels, probabilities)
evaluator.print_report()
```

---

## 📈 Pipeline d'Exécution

### Étape 1: Prétraitement
```
Données brutes (CSV)
    ↓
Détection des 0 manquants
    ↓
Imputation par médiane
    ↓
Discrétisation (3 catégories)
    ↓
Données nettoyées
```

### Étape 2: Construction du Réseau
```
Définir nœuds et arcs
    ↓
Valider DAG (pas de cycles)
    ↓
Calculer ordre topologique
    ↓
Structure prête
```

### Étape 3: Estimation des Paramètres
```
Split train/test (80/20)
    ↓
Pour chaque nœud:
  Compter occurrences
    ↓
  Appliquer lissage Laplace
    ↓
CPT estimées
```

### Étape 4: Inférence et Prédiction
```
Observation partielle
    ↓
Calculer P(outcome=1 | obs)
    ↓
Seuil 0.5 → Prédiction binaire
    ↓
Explication locale des parents
```

### Étape 5: Évaluation
```
Prédictions vs Vrais labels
    ↓
Confusion matrix
    ↓
Précision, Rappel, F1, ROC-AUC
    ↓
Rapport d'évaluation
```

---

## 📝 Exemple d'Utilisation

### Prédiction sur une Observation

```python
from src.main import BayesianDiabetesClassifier

# Créer et entraîner le modèle
classifier = BayesianDiabetesClassifier()
classifier.run_pipeline("data/Pima_Diabetes_organise.csv")

# Faire une prédiction
observation = {
    "Glucose": "Eleve",
    "BMI": "Eleve",
    "Age": "Intermediaire",
    "Insulin": "Normal"
}

prediction, probabilities = classifier.inference.predict(observation)
print(f"Prédiction: {prediction}")
print(f"P(Diabète) = {probabilities['1']:.4f}")

# Obtenir une explication
explanation = classifier.inference.explain_prediction(observation)
for parent, info in explanation["parents_effects"].items():
    print(f"Effet de {parent}: {info['observed_prob']:.4f}")
```

---

## 🧪 Tests Unitaires

Exécuter les tests :

```bash
python tests/test_cpt.py
python tests/test_inference.py
python tests/test_evaluation.py
```

Tous les modules sont testés :
-  Estimation CPT avec lissage
-  Probabilités jointes
-  Inférence postérieure
-  Matrice de confusion
-  Métriques F1, ROC-AUC

---

## 📊 Résultats Attendus

**Sur le dataset Pima (test set) :**

| Métrique | Valeur |
|----------|--------|
| Exactitude | ~82-85% |
| Précision | ~71-74% |
| Rappel | ~83-87% |
| F1-Score | ~77-80% |
| ROC-AUC | ~82-85% |

*Les valeurs varient selon le random split*

---

## 📚 Documentation Mathématique

### Factorisation de Probabilité Jointe

$$P(X_1, \ldots, X_n) = \prod_{i=1}^{n} P(X_i | \text{Parents}(X_i))$$

### Lissage de Laplace

$$P(X_i = x | \text{Parents}) = \frac{\text{count}(x, \text{parents}) + 1}{\sum_k \text{count}(x_k, \text{parents}) + k}$$

### Inférence Bayésienne

$$P(\text{Outcome}=1 | \text{obs}) = \frac{P(\text{Outcome}=1, \text{obs})}{P(\text{obs})}$$

### Métriques d'Évaluation

- **Précision** = TP / (TP + FP)
- **Rappel** = TP / (TP + FN)
- **F1** = 2·(P·R)/(P+R)
- **Exactitude** = (TP + TN) / (TP + TN + FP + FN)

---

## 👨‍💻 Auteur

**Henri AYAMA**  
Étudiant L3 Informatique
Parcours Mathématiques-Informatique  
Aix-Marseille Université (EAD)  
2025-2026

---

## 📧 Contact

- Email: henri.ayama@etu.univ-amu.fr
- GitHub: github.com/LionsOfGod

---

## 📄 Licence

Ce projet est fourni à titre éducatif pour le cours de Licence Informatique.

---

## 🔗 Références

1. Pearl, J. (2009). *Causality: Models, Reasoning and Inference*. Cambridge University Press.
2. Koller, D., & Friedman, N. (2009). *Probabilistic Graphical Models*. MIT Press.
3. Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press.
4. Pima Indians Diabetes Database : https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

---

**Dernière mise à jour** : Août 2026


