"""
Module de prétraitement des données
Chargement, nettoyage, discrétisation des données cliniques
"""

import csv
from collections import defaultdict


class DataPreprocessor:
    """Classe pour prétraiter les données du dataset Pima Indians Diabetes"""
    
    COLUMNS = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
        "Outcome",
    ]
    
    # Variables cliniques avec valeurs manquantes codées en 0
    VARIABLES_WITH_MISSING = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
    ]
    
    def __init__(self):
        """Initialise le préprocesseur"""
        self.data = []
        self.statistics = {}
    
    def load_data(self, filepath):
        """
        Charge le dataset CSV
        
        Args:
            filepath (str): Chemin vers le fichier CSV
            
        Returns:
            list: Liste de dictionnaires {colonne: valeur}
        """
        data = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Convertir les valeurs en float
                    converted_row = {}
                    for col in self.COLUMNS:
                        converted_row[col] = float(row[col])
                    data.append(converted_row)
            
            print(f"✓ Dataset chargé: {len(data)} observations")
            self.data = data
            return data
        
        except FileNotFoundError:
            print(f"✗ Erreur: Fichier {filepath} introuvable")
            return []
    
    def handle_missing_values(self):
        """
        Remplace les valeurs manquantes (codées en 0) par NaN
        """
        for row in self.data:
            for col in self.VARIABLES_WITH_MISSING:
                if row[col] == 0:
                    row[col] = None
        
        print("✓ Valeurs manquantes détectées (0 → NaN)")
    
    def impute_missing_values(self):
        """
        Impute les valeurs manquantes par la médiane
        """
        # Calculer la médiane pour chaque colonne
        for col in self.VARIABLES_WITH_MISSING:
            values = [row[col] for row in self.data if row[col] is not None]
            
            if len(values) == 0:
                continue
            
            # Calculer la médiane
            values.sort()
            n = len(values)
            if n % 2 == 0:
                median = (values[n//2 - 1] + values[n//2]) / 2
            else:
                median = values[n//2]
            
            # Imputer les valeurs manquantes
            for row in self.data:
                if row[col] is None:
                    row[col] = median
            
            self.statistics[col] = {"median": median, "count": n}
        
        print("✓ Imputation par médiane effectuée")
    
    def discretize_glucose(self):
        """
        Discrétise Glucose en 3 catégories:
        - Faible: < 100
        - Intermediaire: 100-125
        - Eleve: >= 126
        """
        for row in self.data:
            value = row["Glucose"]
            if value < 100:
                row["Glucose"] = "Faible"
            elif value < 126:
                row["Glucose"] = "Intermediaire"
            else:
                row["Glucose"] = "Eleve"
    
    def discretize_blood_pressure(self):
        """
        Discrétise BloodPressure en 3 catégories:
        - Faible: < 60
        - Normal: 60-79
        - Eleve: >= 80
        """
        for row in self.data:
            value = row["BloodPressure"]
            if value < 60:
                row["BloodPressure"] = "Faible"
            elif value < 80:
                row["BloodPressure"] = "Normal"
            else:
                row["BloodPressure"] = "Eleve"
    
    def discretize_skin_thickness(self):
        """
        Discrétise SkinThickness en 3 catégories:
        - Faible: < 20
        - Normal: 20-34
        - Eleve: >= 35
        """
        for row in self.data:
            value = row["SkinThickness"]
            if value < 20:
                row["SkinThickness"] = "Faible"
            elif value < 35:
                row["SkinThickness"] = "Normal"
            else:
                row["SkinThickness"] = "Eleve"
    
    def discretize_insulin(self):
        """
        Discrétise Insulin en 3 catégories:
        - Faible: < 50
        - Normal: 50-129
        - Eleve: >= 130
        """
        for row in self.data:
            value = row["Insulin"]
            if value < 50:
                row["Insulin"] = "Faible"
            elif value < 130:
                row["Insulin"] = "Normal"
            else:
                row["Insulin"] = "Eleve"
    
    def discretize_bmi(self):
        """
        Discrétise BMI en 3 catégories:
        - Faible: < 25
        - Normal: 25-29
        - Eleve: >= 30
        """
        for row in self.data:
            value = row["BMI"]
            if value < 25:
                row["BMI"] = "Faible"
            elif value < 30:
                row["BMI"] = "Normal"
            else:
                row["BMI"] = "Eleve"
    
    def discretize_pregnancies(self):
        """
        Discrétise Pregnancies en 3 catégories:
        - Faible: 0-2
        - Intermediaire: 3-6
        - Eleve: >= 7
        """
        for row in self.data:
            value = int(row["Pregnancies"])
            if value <= 2:
                row["Pregnancies"] = "Faible"
            elif value <= 6:
                row["Pregnancies"] = "Intermediaire"
            else:
                row["Pregnancies"] = "Eleve"
    
    def discretize_age(self):
        """
        Discrétise Age en 3 catégories:
        - Jeune: < 30
        - Intermediaire: 30-39
        - Eleve: >= 40
        """
        for row in self.data:
            value = int(row["Age"])
            if value < 30:
                row["Age"] = "Jeune"
            elif value < 40:
                row["Age"] = "Intermediaire"
            else:
                row["Age"] = "Eleve"
    
    def discretize_diabetes_pedigree(self):
        """
        Discrétise DiabetesPedigreeFunction en 3 catégories:
        - Faible: < 0.30
        - Intermediaire: 0.30-0.59
        - Eleve: >= 0.60
        """
        for row in self.data:
            value = row["DiabetesPedigreeFunction"]
            if value < 0.30:
                row["DiabetesPedigreeFunction"] = "Faible"
            elif value < 0.60:
                row["DiabetesPedigreeFunction"] = "Intermediaire"
            else:
                row["DiabetesPedigreeFunction"] = "Eleve"
    
    def discretize_outcome(self):
        """
        Convertit Outcome en chaîne (pour cohérence)
        """
        for row in self.data:
            row["Outcome"] = str(int(row["Outcome"]))
    
    def preprocess(self, filepath):
        """
        Pipeline complet de prétraitement
        
        Args:
            filepath (str): Chemin vers le CSV
            
        Returns:
            list: Données prétraitées
        """
        self.load_data(filepath)
        self.handle_missing_values()
        self.impute_missing_values()
        
        # Discrétiser toutes les variables
        self.discretize_glucose()
        self.discretize_blood_pressure()
        self.discretize_skin_thickness()
        self.discretize_insulin()
        self.discretize_bmi()
        self.discretize_pregnancies()
        self.discretize_age()
        self.discretize_diabetes_pedigree()
        self.discretize_outcome()
        
        print("✓ Prétraitement terminé")
        return self.data
