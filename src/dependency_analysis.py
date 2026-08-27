"""
Module d'analyse des dépendances statistiques
Test du chi-deux pour identifier les dépendances entre variables
"""

import math


class DependencyAnalyzer:
    """Classe pour analyser les dépendances entre variables"""
    
    def __init__(self, significance_level=0.05):
        """
        Initialise l'analyseur
        
        Args:
            significance_level (float): Seuil de p-value pour l'indépendance
        """
        self.significance_level = significance_level
        self.dependencies = []
    
    def chi_square_test(self, data, var_x, var_y):
        """
        Effectue le test du chi-deux pour l'indépendance
        entre deux variables
        
        H0: X et Y sont indépendantes
        H1: X et Y sont dépendantes
        
        Args:
            data (list): Observations
            var_x (str): Première variable
            var_y (str): Deuxième variable
            
        Returns:
            dict: {chi2_stat, p_value, is_dependent}
        """
        
        # Créer le tableau de contingence
        contingency_table = self._build_contingency_table(data, var_x, var_y)
        
        # Calculer la statistique du chi-deux
        chi2_stat = self._compute_chi2(contingency_table)
        
        # Calculer les degrés de liberté
        n_rows = len(contingency_table)
        n_cols = len(contingency_table[list(contingency_table.keys())[0]])
        df = (n_rows - 1) * (n_cols - 1)
        
        # Approximer la p-value (utiliser une approximation simple)
        # Pour une implémentation précise, il faudrait une table de chi-deux
        p_value = self._approximate_p_value(chi2_stat, df)
        
        # Déterminer l'indépendance
        is_dependent = p_value < self.significance_level
        
        return {
            "variables": (var_x, var_y),
            "chi2_statistic": chi2_stat,
            "degrees_of_freedom": df,
            "p_value": p_value,
            "is_dependent": is_dependent,
            "contingency_table": contingency_table
        }
    
    def _build_contingency_table(self, data, var_x, var_y):
        """
        Construit le tableau de contingence
        
        Returns:
            dict: {x_value: {y_value: count}}
        """
        
        table = {}
        
        for observation in data:
            x_val = observation.get(var_x)
            y_val = observation.get(var_y)
            
            if x_val is None or y_val is None:
                continue
            
            if x_val not in table:
                table[x_val] = {}
            
            if y_val not in table[x_val]:
                table[x_val][y_val] = 0
            
            table[x_val][y_val] += 1
        
        return table
    
    def _compute_chi2(self, contingency_table):
        """
        Calcule la statistique du chi-deux
        
        χ² = Σ (O - E)² / E
        
        où O = observé, E = attendu sous indépendance
        """
        
        # Calculer les totaux
        row_totals = {}
        col_totals = {}
        n_total = 0
        
        for x_val, y_dict in contingency_table.items():
            row_total = sum(y_dict.values())
            row_totals[x_val] = row_total
            n_total += row_total
            
            for y_val, count in y_dict.items():
                if y_val not in col_totals:
                    col_totals[y_val] = 0
                col_totals[y_val] += count
        
        # Calculer chi-deux
        chi2 = 0
        
        for x_val, y_dict in contingency_table.items():
            for y_val, observed in y_dict.items():
                # Fréquence attendue sous indépendance
                expected = (row_totals[x_val] * col_totals[y_val]) / n_total
                
                if expected > 0:
                    chi2 += (observed - expected) ** 2 / expected
        
        return chi2
    
    def _approximate_p_value(self, chi2_stat, df):
        """
        Approxime la p-value du chi-deux
        
        Utilise une approximation simple basée sur la distribution
        """
        
        # Approximation très simple
        # Pour une vraie distribution, utiliser scipy.stats.chi2.sf
        
        if df <= 0:
            return 1.0
        
        # Approximation rude : pour df=1, chi2=3.84 → p≈0.05
        if chi2_stat < 0:
            return 1.0
        
        # Utiliser une formule approximative
        # Ceci est une approximation grossière
        if chi2_stat < 2.706:  # df=1
            return 0.1
        elif chi2_stat < 3.841:
            return 0.05
        elif chi2_stat < 5.024:
            return 0.025
        else:
            return 0.001
    
    def analyze_all_pairs(self, data, variables):
        """
        Analyse toutes les paires de variables
        
        Args:
            data (list): Observations
            variables (list): Liste des variables
            
        Returns:
            list: Résultats de chi-deux pour chaque paire
        """
        
        results = []
        
        for i in range(len(variables)):
            for j in range(i+1, len(variables)):
                var_x = variables[i]
                var_y = variables[j]
                
                result = self.chi_square_test(data, var_x, var_y)
                results.append(result)
        
        return results
    
    def print_dependency_report(self, results):
        """Affiche un rapport des dépendances"""
        
        print("\n" + "="*70)
        print("RAPPORT D'ANALYSE DES DEPENDANCES")
        print("="*70)
        
        print(f"\nSeuil de signification: p < {self.significance_level}")
        print()
        
        # Dépendances trouvées
        dependent_pairs = [r for r in results if r["is_dependent"]]
        independent_pairs = [r for r in results if not r["is_dependent"]]
        
        print(f"Paires dépendantes: {len(dependent_pairs)}")
        print(f"Paires indépendantes: {len(independent_pairs)}")
        
        print("\nPAIRES DEPENDANTES (p < 0.05):")
        print("-"*70)
        
        for result in sorted(dependent_pairs, key=lambda r: r["p_value"]):
            var_x, var_y = result["variables"]
            chi2 = result["chi2_statistic"]
            p_val = result["p_value"]
            print(f"  {var_x:25} ↔ {var_y:25} | χ²={chi2:7.2f} | p={p_val:.4f}")
        
        if not dependent_pairs:
            print("  Aucune paire dépendante trouvée")
        
        print("\nPAIRES INDEPENDANTES (p ≥ 0.05):")
        print("-"*70)
        
        for result in sorted(independent_pairs, key=lambda r: r["p_value"], reverse=True)[:10]:
            var_x, var_y = result["variables"]
            chi2 = result["chi2_statistic"]
            p_val = result["p_value"]
            print(f"  {var_x:25} ↔ {var_y:25} | χ²={chi2:7.2f} | p={p_val:.4f}")
        
        if len(independent_pairs) > 10:
            print(f"  ... et {len(independent_pairs) - 10} autres paires indépendantes")
        
        print("="*70 + "\n")
    
    def get_dependent_variables(self, results, target_variable=None):
        """
        Récupère les variables dépendantes
        
        Args:
            results (list): Résultats du chi-deux
            target_variable (str): Variable cible (optionnel)
            
        Returns:
            list: Variables dépendantes
        """
        
        dependencies = {}
        
        for result in results:
            if result["is_dependent"]:
                var_x, var_y = result["variables"]
                
                if target_variable:
                    if var_x == target_variable:
                        if var_x not in dependencies:
                            dependencies[var_x] = []
                        dependencies[var_x].append(var_y)
                    elif var_y == target_variable:
                        if var_y not in dependencies:
                            dependencies[var_y] = []
                        dependencies[var_y].append(var_x)
                else:
                    if var_x not in dependencies:
                        dependencies[var_x] = []
                    if var_y not in dependencies:
                        dependencies[var_y] = []
                    
                    dependencies[var_x].append(var_y)
                    dependencies[var_y].append(var_x)
        
        return dependencies
