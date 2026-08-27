"""
Configuration du projet Réseaux Bayésiens pour le Diagnostic du Diabète
"""

import os

# ==================== CHEMINS ====================

# Répertoire racine du projet
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Répertoires
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
TESTS_DIR = os.path.join(PROJECT_ROOT, "tests")
NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, "notebooks")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Fichiers de données
DATA_PATH = os.path.join(DATA_DIR, "Pima_Diabetes_organise.csv")

# ==================== PARAMETRES DU MODELE ====================

# Hyperparamètres
LAPLACE_SMOOTHING = True  # Toujours utiliser le lissage de Laplace
RANDOM_STATE = 42  # Pour la reproductibilité

# Partitionnement train/test
TEST_SIZE = 0.2  # 80% train, 20% test
VALIDATION_SIZE = 0.1  # 10% pour validation croisée

# Seuils de décision
PREDICTION_THRESHOLD = 0.5  # Seuil pour classification binaire

# ==================== VARIABLES DU RESEAU ====================

# Nœuds du réseau
NETWORK_NODES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome"
]

# Arcs du réseau (causalité présumée)
NETWORK_EDGES = [
    ("Pregnancies", "Age"),
    ("SkinThickness", "Insulin"),
    ("SkinThickness", "BMI"),
    ("Glucose", "Insulin"),
    ("Glucose", "Outcome"),
    ("BMI", "Outcome"),
    ("Age", "Outcome"),
    ("Insulin", "Outcome")
]

# ==================== SEUILS DE DISCRETISATION ====================

DISCRETIZATION_THRESHOLDS = {
    "Glucose": {
        "Faible": (0, 100),
        "Intermediaire": (100, 126),
        "Eleve": (126, float('inf'))
    },
    "BloodPressure": {
        "Faible": (0, 60),
        "Normal": (60, 80),
        "Eleve": (80, float('inf'))
    },
    "BMI": {
        "Faible": (0, 25),
        "Normal": (25, 30),
        "Eleve": (30, float('inf'))
    },
    "Insulin": {
        "Faible": (0, 50),
        "Normal": (50, 130),
        "Eleve": (130, float('inf'))
    },
    "SkinThickness": {
        "Faible": (0, 20),
        "Normal": (20, 35),
        "Eleve": (35, float('inf'))
    },
    "Age": {
        "Jeune": (0, 30),
        "Intermediaire": (30, 40),
        "Eleve": (40, float('inf'))
    },
    "DiabetesPedigreeFunction": {
        "Faible": (0, 0.30),
        "Intermediaire": (0.30, 0.60),
        "Eleve": (0.60, float('inf'))
    },
    "Pregnancies": {
        "Faible": (0, 2.99),
        "Intermediaire": (3, 6.99),
        "Eleve": (7, float('inf'))
    }
}

# Variables avec valeurs manquantes codées en 0
VARIABLES_WITH_MISSING = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

# ==================== LOGGING ====================

VERBOSE = True  # Afficher les détails d'exécution
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR

# ==================== AFFICHAGE ====================

# Largeur du texte pour les affichages
DISPLAY_WIDTH = 70

# Nombre de décimales pour les probabilités
PRECISION_DECIMALS = 4
