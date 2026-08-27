"""
Module d'évaluation des performances
Métriques: Matrice de confusion, Précision, Rappel, F1, ROC-AUC
"""

import math


class Evaluator:
    """Classe pour évaluer les performances du modèle"""
    
    def __init__(self):
        """Initialise l'évaluateur"""
        self.tp = 0  # True Positives
        self.fp = 0  # False Positives
        self.tn = 0  # True Negatives
        self.fn = 0  # False Negatives
        self.predictions = []
        self.true_labels = []
        self.probabilities = []
    
    def evaluate(self, predictions, true_labels, probabilities=None):
        """
        Évalue les prédictions contre les vrais labels
        
        Args:
            predictions (list): Valeurs prédites [0, 1, 1, ...]
            true_labels (list): Vrais labels [0, 1, 1, ...]
            probabilities (list): Probabilités associées (optionnel)
        """
        
        if len(predictions) != len(true_labels):
            raise ValueError("Longueurs différentes")
        
        self.predictions = predictions
        self.true_labels = true_labels
        self.probabilities = probabilities if probabilities else []
        
        self._compute_confusion_matrix()
    
    def _compute_confusion_matrix(self):
        """Calcule la matrice de confusion"""
        
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        
        for pred, true in zip(self.predictions, self.true_labels):
            pred_val = int(pred)
            true_val = int(true)
            
            if pred_val == 1 and true_val == 1:
                self.tp += 1
            elif pred_val == 1 and true_val == 0:
                self.fp += 1
            elif pred_val == 0 and true_val == 0:
                self.tn += 1
            elif pred_val == 0 and true_val == 1:
                self.fn += 1
    
    def accuracy(self):
        """
        Calcule l'exactitude (Accuracy)
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
        
        Returns:
            float: Exactitude
        """
        total = self.tp + self.tn + self.fp + self.fn
        if total == 0:
            return 0
        return (self.tp + self.tn) / total
    
    def precision(self):
        """
        Calcule la précision (Precision)
        Precision = TP / (TP + FP)
        
        Returns:
            float: Précision
        """
        if self.tp + self.fp == 0:
            return 0
        return self.tp / (self.tp + self.fp)
    
    def recall(self):
        """
        Calcule le rappel (Recall/Sensitivity)
        Recall = TP / (TP + FN)
        
        Returns:
            float: Rappel
        """
        if self.tp + self.fn == 0:
            return 0
        return self.tp / (self.tp + self.fn)
    
    def f1_score(self):
        """
        Calcule le F1-score
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        
        Returns:
            float: F1-score
        """
        prec = self.precision()
        rec = self.recall()
        
        if prec + rec == 0:
            return 0
        
        return 2 * (prec * rec) / (prec + rec)
    
    def specificity(self):
        """
        Calcule la spécificité (Specificity)
        Specificity = TN / (TN + FP)
        
        Returns:
            float: Spécificité
        """
        if self.tn + self.fp == 0:
            return 0
        return self.tn / (self.tn + self.fp)
    
    def confusion_matrix(self):
        """
        Retourne la matrice de confusion
        
        Returns:
            dict: Matrice de confusion {TP, TN, FP, FN}
        """
        return {
            "TP": self.tp,
            "TN": self.tn,
            "FP": self.fp,
            "FN": self.fn
        }
    
    def roc_auc(self):
        """
        Calcule l'aire sous la courbe ROC (ROC-AUC)
        
        Utilise la formule: AUC = (TP/(TP+FN) + TN/(TN+FP)) / 2
        ou mieux: compter les paires concordantes
        
        Returns:
            float: ROC-AUC score
        """
        
        if not self.probabilities:
            # Approximation simple si pas de probabilités
            return self._approximate_roc_auc()
        
        return self._compute_roc_auc_from_probs()
    
    def _approximate_roc_auc(self):
        """Approximation simple de ROC-AUC"""
        sensitivity = self.recall()  # TPR
        specificity = self.specificity()  # TNR
        
        # Approximation: moyenne des deux taux
        return (sensitivity + specificity) / 2
    
    def _compute_roc_auc_from_probs(self):
        """Calcule ROC-AUC à partir des probabilités"""
        
        # Créer des paires (probabilité, label)
        pairs = list(zip(self.probabilities, self.true_labels))
        
        # Trier par probabilité décroissante
        pairs.sort(key=lambda x: x[0], reverse=True)
        
        # Compter les paires concordantes
        n_positives = sum(1 for label in self.true_labels if int(label) == 1)
        n_negatives = sum(1 for label in self.true_labels if int(label) == 0)
        
        if n_positives == 0 or n_negatives == 0:
            return 0.5
        
        concordant = 0
        discordant = 0
        
        for i, (prob_i, label_i) in enumerate(pairs):
            for j in range(i+1, len(pairs)):
                prob_j, label_j = pairs[j]
                
                label_i_val = int(label_i)
                label_j_val = int(label_j)
                
                # Si i est positif et j est négatif
                if label_i_val == 1 and label_j_val == 0:
                    if prob_i > prob_j:
                        concordant += 1
                    elif prob_i < prob_j:
                        discordant += 1
                
                # Si i est négatif et j est positif
                elif label_i_val == 0 and label_j_val == 1:
                    if prob_i < prob_j:
                        concordant += 1
                    elif prob_i > prob_j:
                        discordant += 1
        
        total = concordant + discordant
        if total == 0:
            return 0.5
        
        return concordant / total
    
    def print_report(self):
        """Affiche un rapport complet d'évaluation"""
        
        print("\n" + "="*60)
        print("RAPPORT D'EVALUATION")
        print("="*60)
        
        # Matrice de confusion
        print("\nMATRICE DE CONFUSION:")
        print(f"  TP (Vrais Positifs):  {self.tp}")
        print(f"  FP (Faux Positifs):   {self.fp}")
        print(f"  TN (Vrais Négatifs):  {self.tn}")
        print(f"  FN (Faux Négatifs):   {self.fn}")
        
        # Métriques
        print("\nMETRIQUES DE PERFORMANCE:")
        print(f"  Exactitude (Accuracy):  {self.accuracy():.4f}")
        print(f"  Précision (Precision):  {self.precision():.4f}")
        print(f"  Rappel (Recall):        {self.recall():.4f}")
        print(f"  Spécificité:            {self.specificity():.4f}")
        print(f"  F1-Score:               {self.f1_score():.4f}")
        print(f"  ROC-AUC:                {self.roc_auc():.4f}")
        
        print("="*60 + "\n")
    
    def print_confusion_table(self):
        """Affiche la matrice de confusion sous forme de tableau"""
        
        print("\n" + "="*50)
        print("MATRICE DE CONFUSION")
        print("="*50)
        print(f"{'':20}{'Prédiction':^20}")
        print(f"{'Réalité':20}{'Positif':^10}{'Négatif':^10}")
        print("-"*50)
        print(f"{'Positif':20}{str(self.tp):^10}{str(self.fn):^10}")
        print(f"{'Négatif':20}{str(self.fp):^10}{str(self.tn):^10}")
        print("="*50 + "\n")
