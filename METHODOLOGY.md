# Méthodologie Détaillée - Réseaux Bayésiens pour le Diabète

## Table des matières

1. [Prétraitement des Données](#prétraitement)
2. [Structure du Réseau](#structure)
3. [Estimation des CPT](#cpt)
4. [Inférence Probabiliste](#inférence)
5. [Évaluation](#évaluation)

---

## Prétraitement des Données {#prétraitement}

### 1.1 Chargement et Inspection

Le dataset Pima Indians Diabetes contient 768 observations et 9 variables.

```
Pregnancies: Nombre de grossesses (entier)
Glucose: mg/dL (continu)
BloodPressure: mmHg (continu)
SkinThickness: mm (continu)
Insulin: µU/mL (continu)
BMI: kg/m² (continu)
DiabetesPedigreeFunction: Score 0-1 (continu)
Age: années (entier)
Outcome: 0 ou 1 (binaire)
```

### 1.2 Gestion des Valeurs Manquantes

**Problème** : Les valeurs manquantes sont codées en 0 (biologiquement impossible pour les mesures cliniques).

**Solution** :
1. Identifier les 0 invalides dans les colonnes cliniques
2. Remplacer par `NaN`
3. Imputer avec la **médiane** de chaque colonne

```python
# Exemple d'imputation
median_glucose = sorted(valid_glucose_values)[len(valid_glucose_values) // 2]
missing_glucose = [g if g != 0 else median_glucose for g in glucose_column]
```

### 1.3 Discrétisation des Variables

**Pourquoi discrétiser** ?
- Réduire la dimensionalité (continues → 3 catégories)
- Faciliter l'inférence
- Justification clinique des seuils

**Seuils utilisés** (cliniquement établis) :

| Variable | Faible | Intermédiaire | Élevé |
|----------|--------|---------------|-------|
| **Glucose** | < 100 | 100-125 | ≥ 126 |
| **BMI** | < 25 | 25-29 | ≥ 30 |
| **Age** | < 30 | 30-39 | ≥ 40 |
| **Insulin** | < 50 | 50-129 | ≥ 130 |
| **BloodPressure** | < 60 | 60-79 | ≥ 80 |

---

## Structure du Réseau Bayésien {#structure}

### 2.1 Définition du DAG

**Nœuds** : 9 variables (X₁, ..., X₉)

**Arcs** : 8 relations causales présumées

```
Pregnancies ──→ Age ──────┐
                           │
SkinThickness ─┬───────→ Insulin ──────┐
               │                       │
               └───→ BMI ──────────────┤
                                       ├──→ Outcome
Glucose ───┬──────────────────────────┤
           │                          │
           └─────→ Insulin ──────────┘
```

### 2.2 Justification des Arcs

| Arc | Justification Clinique |
|-----|------------------------|
| Pregnancies → Age | L'âge augmente avec le nombre de grossesses |
| SkinThickness → Insulin | Composition corporelle affecte sensibilité insuline |
| SkinThickness → BMI | L'épaisseur cutanée reflète la masse grasse |
| Glucose → Insulin | Homéostasie glucose-insuline |
| **Glucose → Outcome** | Hyperglycémie est un facteur majeur du diabète |
| **BMI → Outcome** | Obésité augmente le risque de diabète |
| **Age → Outcome** | Le risque augmente avec l'âge |
| **Insulin → Outcome** | Insulinémie anormale indique une dysfonction |

### 2.3 Propriétés du DAG

- **Acyclique** : Pas de boucles causales
- **Ordre topologique** : Pregnancies, SkinThickness, Glucose → Age, BMI, Insulin → Outcome
- **Factorisation** :

$$P(X_1, ..., X_9) = P(Pregnancies) \cdot P(SkinThickness) \cdot P(Glucose) \cdot P(Age|Pregnancies) \cdot P(BMI|SkinThickness) \cdot P(Insulin|SkinThickness, Glucose) \cdot P(BloodPressure) \cdot P(DiabetesPedigreeFunction) \cdot P(Outcome|Age, BMI, Insulin, Glucose)$$

---

## Estimation des CPT {#cpt}

### 3.1 Probabilités Inconditionnelles

Pour un nœud sans parents (racine du DAG) :

$$P(X = x) = \frac{\text{count}(X = x) + 1}{N + k}$$

où :
- `count(X = x)` = nombre d'observations avec X = x
- N = nombre total d'observations
- k = nombre d'états de X (3 dans notre cas)
- **+1** et **+k** = **lissage de Laplace**

### 3.2 Probabilités Conditionnelles

Pour un nœud avec parents :

$$P(X_i = x | \text{Parents} = p) = \frac{\text{count}(X_i = x, \text{Parents} = p) + 1}{\text{count}(\text{Parents} = p) + k}$$

### 3.3 Lissage de Laplace

**Problème sans lissage** :
- Une configuration absente des données → Probabilité = 0
- Invalidait toute l'inférence pour cette configuration

**Solution (Laplace)** :
- Ajouter +1 à chaque comte
- Normaliser par (total + k) au lieu de total
- Garantit P(x) > 0 partout
- Interprétation : "pseudo-observation" fictive de chaque état

**Exemple** :
```
Données: 100 obs, Outcome=1 vu 30 fois
Sans lissage: P(Outcome=1) = 30/100 = 0.30
Avec lissage: P(Outcome=1) = (30+1)/(100+2) = 31/102 ≈ 0.304
```

### 3.4 Construction des CPT

Algorithme :

```
Pour chaque nœud X:
  Si X n'a pas de parents:
    CPT(X) = {état: P(état)} inconditionnel
  Sinon:
    Pour chaque combinaison de valeurs parentales:
      CPT(X | parents=val) = {état: P(état | parents=val)}
    Fin pour
  Fin si
Fin pour
```

---

## Inférence Probabiliste {#inférence}

### 4.1 Probabilité Jointe

Pour une observation complète (toutes les variables observées) :

$$P(\text{obs}) = \prod_{i=1}^{9} P(X_i = \text{obs}_i | \text{Parents}(X_i) = \text{obs}_{\text{parents}})$$

**Algorithme** :
1. Parcourir les nœuds en ordre topologique
2. Pour chaque nœud, récupérer sa probabilité conditionnelle dans la CPT
3. Multiplier toutes les probabilités

**Exemple** :
```
P(Glucose=Eleve, BMI=Eleve, Age=Intermediaire, Outcome=1)
= P(Glucose=Eleve)
  × P(Age=Intermediaire | Pregnancies=Faible)
  × P(BMI=Eleve | SkinThickness=Eleve)
  × P(Outcome=1 | Glucose=Eleve, BMI=Eleve, Age=Intermediaire, Insulin=Normal)
```

### 4.2 Probabilité Postérieure (Diagnostic)

**Objectif** : Calculer P(Outcome | observations)

**Utilise la loi de Bayes** :

$$P(\text{Outcome}=1 | \text{obs}_{\text{autres}}) = \frac{P(\text{Outcome}=1, \text{obs}_{\text{autres}})}{P(\text{obs}_{\text{autres}})}$$

**Calcul** :
1. P(Outcome=1, obs) = ∏ P(Xi | Parents) avec Outcome=1
2. P(Outcome=0, obs) = ∏ P(Xi | Parents) avec Outcome=0
3. P(Outcome=1 | obs) = P(Outcome=1, obs) / (P(Outcome=1, obs) + P(Outcome=0, obs))

**Décision binaire** :
```
Si P(Outcome=1 | obs) ≥ 0.5:
  Prédiction = "Diabétique"
Sinon:
  Prédiction = "Non diabétique"
```

### 4.3 Explications Locales

**Idée** : Pour chaque parent direct de Outcome, analyser son influence

Pour chaque parent P ∈ {Glucose, BMI, Age, Insulin} :

$$\text{Effet}(P = \text{alt}) = P(\text{Outcome}=1 | P = \text{alt}, \text{autres}) - P(\text{Outcome}=1 | P = \text{obs}, \text{autres})$$

Les effets positifs augmentent le risque, les négatifs le diminuent.

---

## Évaluation {#évaluation}

### 5.1 Matrice de Confusion

|  | Prédiction=1 | Prédiction=0 |
|---|---|---|
| **Réalité=1** | TP | FN |
| **Réalité=0** | FP | TN |

### 5.2 Métriques Principales

**Précision** (exactitude des positifs prédits) :
$$\text{Precision} = \frac{TP}{TP + FP}$$

**Rappel** (capacité à identifier les diabétiques) :
$$\text{Recall} = \frac{TP}{TP + FN}$$

**F1-Score** (moyenne harmonique) :
$$F1 = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

**Exactitude globale** :
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

### 5.3 Courbe ROC et AUC

La courbe ROC affiche :
- **Axe Y** : TPR (Recall) = TP/(TP+FN)
- **Axe X** : FPR = FP/(FP+TN)

**AUC** = Aire sous la courbe
- AUC = 1.0 : Discriminateur parfait
- AUC = 0.5 : Pas mieux que l'aléatoire
- AUC = 0.0 : Complètement inversé

Approximation simple :
$$\text{AUC} \approx \frac{\text{TPR} + \text{Spécificité}}{2} = \frac{\text{Recall} + (TN/(TN+FP))}{2}$$

---

## Considérations Pratiques

### Validation Croisée

Pour une estimation plus robuste des performances :

```
Pour k = 1 à 5 folds:
  1. Diviser les données en k parts
  2. Utiliser 4 parts pour l'entraînement
  3. Tester sur la 1ère part
  4. Enregistrer les métriques
Fin pour

Moyenne des k résultats = Performance estimée
```

### Calibration des Probabilités

Les probabilités du modèle peuvent ne pas être bien calibrées :

**Correction de Platt** :
1. Entraîner un modèle logistique sur les probabilités du modèle
2. Utiliser ce modèle pour recalibrer les probabilités

### Équilibre des Classes

Si le dataset est déséquilibré (plus de non-diabétiques) :
- Utiliser **Stratified Split** (maintient les proportions)
- Considérer **weighted metrics** ou **F1-score** plutôt qu'accuracy

---

## Références

1. Pearl, J. (2009). *Causality: Models, Reasoning and Inference*.
2. Koller & Friedman (2009). *Probabilistic Graphical Models*.
3. Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*.
4. Darwiche, A. (2009). *Modeling and Reasoning with Bayesian Networks*.
