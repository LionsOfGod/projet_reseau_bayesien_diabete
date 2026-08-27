"""
Pipeline principal du réseau bayésien pour le diagnostic du diabète
Exécute la chaîne complète: données → entraînement → évaluation
"""

from preprocessing import DataPreprocessor
from network import BayesianNetwork
from cpt import CPTEstimator
from inference import BayesianInference
from data_split import DataSplitter
from evaluation import Evaluator


class BayesianDiabetesClassifier:
    """Pipeline complet du classificateur bayésien"""
    
    def __init__(self):
        """Initialise le pipeline"""
        self.data = None
        self.train_data = None
        self.test_data = None
        self.network = None
        self.cpt_estimator = None
        self.inference = None
        self.evaluator = None
    
    def run_pipeline(self, data_path):
        """
        Exécute le pipeline complet
        
        Args:
            data_path (str): Chemin vers le CSV
        """
        
        print("\n" + "="*70)
        print("RESEAU BAYESIEN POUR LE DIAGNOSTIC DU DIABETE")
        print("="*70 + "\n")
        
        # Étape 1: Prétraitement
        print("ETAPE 1: PREPROCESSING DES DONNEES")
        print("-"*70)
        self.preprocess_data(data_path)
        print()
        
        # Étape 2: Structure du réseau
        print("ETAPE 2: DEFINITION DE LA STRUCTURE DU RESEAU")
        print("-"*70)
        self.build_network()
        print()
        
        # Étape 3: Split train/test
        print("ETAPE 3: PARTITIONNEMENT TRAIN/TEST")
        print("-"*70)
        self.split_data()
        print()
        
        # Étape 4: Estimation des CPT
        print("ETAPE 4: ESTIMATION DES CPT")
        print("-"*70)
        self.estimate_cpts()
        print()
        
        # Étape 5: Inférence et prédictions
        print("ETAPE 5: INFERENCE ET PREDICTIONS")
        print("-"*70)
        predictions, probabilities = self.make_predictions()
        print()
        
        # Étape 6: Évaluation
        print("ETAPE 6: EVALUATION DES PERFORMANCES")
        print("-"*70)
        self.evaluate(predictions, probabilities)
        print()
        
        # Étape 7: Exempldes d'explications
        print("ETAPE 7: EXEMPLES D'EXPLICATIONS LOCALES")
        print("-"*70)
        self.explain_predictions()
        print()
        
        print("="*70)
        print("PIPELINE TERMINE AVEC SUCCES")
        print("="*70 + "\n")
    
    def preprocess_data(self, data_path):
        """Prétraite les données"""
        preprocessor = DataPreprocessor()
        self.data = preprocessor.preprocess(data_path)
        print(f"✓ Total d'observations: {len(self.data)}")
    
    def build_network(self):
        """Construit la structure du réseau"""
        self.network = BayesianNetwork()
        self.network.print_structure()
    
    def split_data(self):
        """Divise les données en train/test"""
        splitter = DataSplitter()
        self.train_data, self.test_data = splitter.train_test_split(
            self.data,
            test_size=0.2,
            random_state=42
        )
        print(f"✓ Données d'entraînement: {len(self.train_data)}")
        print(f"✓ Données de test: {len(self.test_data)}")
    
    def estimate_cpts(self):
        """Estime les CPT à partir des données d'entraînement"""
        self.cpt_estimator = CPTEstimator(laplace_smoothing=True)
        self.cpt_estimator.build_all_cpts(
            self.train_data,
            self.network.NODES,
            self.network.EDGES
        )
        
        # Afficher un exemple de CPT
        self.cpt_estimator.print_cpt("Outcome")
    
    def make_predictions(self):
        """Effectue les prédictions sur l'ensemble de test"""
        self.inference = BayesianInference(self.cpt_estimator, self.network)
        
        predictions = []
        probabilities = []
        
        for i, observation in enumerate(self.test_data):
            if i == 0:
                print(f"Traitement de {len(self.test_data)} observations...")
            
            if (i + 1) % max(1, len(self.test_data) // 10) == 0:
                print(f"  → {i+1}/{len(self.test_data)} complétées")
            
            pred, probs = self.inference.predict(observation)
            predictions.append(pred)
            probabilities.append(probs.get("1", 0.5))
        
        print(f"✓ Prédictions effectuées")
        return predictions, probabilities
    
    def evaluate(self, predictions, probabilities):
        """Évalue les performances du modèle"""
        self.evaluator = Evaluator()
        
        true_labels = [str(int(obs["Outcome"])) for obs in self.test_data]
        
        self.evaluator.evaluate(predictions, true_labels, probabilities)
        self.evaluator.print_confusion_table()
        self.evaluator.print_report()
    
    def explain_predictions(self):
        """Affiche des exemples d'explications locales"""
        
        print("\nEXEMPLES D'EXPLICATIONS LOCALES:")
        print("="*70)
        
        # Afficher 3 exemples
        n_examples = min(3, len(self.test_data))
        
        for idx in range(n_examples):
            observation = self.test_data[idx]
            true_label = observation["Outcome"]
            
            prediction, probabilities = self.inference.predict(observation)
            prob_diabetic = probabilities.get("1", 0.5)
            
            print(f"\nExample {idx + 1}:")
            print(f"  Vrai label: {true_label}")
            print(f"  Prédiction: {prediction}")
            print(f"  P(Diabète) = {prob_diabetic:.4f}")
            
            print(f"  Profil du patient:")
            for node in ["Glucose", "BMI", "Age", "Insulin"]:
                if node in observation:
                    print(f"    {node}: {observation[node]}")
            
            # Explication
            explanation = self.inference.explain_prediction(observation)
            
            print(f"  Facteurs influents:")
            for parent, info in explanation["parents_effects"].items():
                if "alternatives" in info:
                    for alt_value, alt_info in info["alternatives"].items():
                        effect = alt_info.get("effect", 0)
                        print(f"    Si {parent}={alt_value}: effet={effect:+.4f}")


def main():
    """Fonction principale"""
    
    # Chemin vers les données
    data_path = "data/Pima_Diabetes_organise.csv"
    
    # Créer et exécuter le pipeline
    classifier = BayesianDiabetesClassifier()
    classifier.run_pipeline(data_path)


if __name__ == "__main__":
    main()
