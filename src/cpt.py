"""
Module d'estimation des Tables de Probabilités Conditionnelles (CPT)
Utilise le lissage de Laplace pour éviter les zéros
"""

from collections import defaultdict


class CPTEstimator:
    """Classe pour estimer les CPT à partir des données"""
    
    def __init__(self, laplace_smoothing=True):
        """
        Initialise l'estimateur de CPT
        
        Args:
            laplace_smoothing (bool): Appliquer le lissage de Laplace
        """
        self.laplace_smoothing = laplace_smoothing
        self.cpts = {}
    
    def estimate_cpt(self, data, node, parents):
        """
        Estime P(node | parents) à partir des données
        
        Args:
            data (list): Liste d'observations (dicts)
            node (str): Nœud cible
            parents (list): Parents du nœud
            
        Returns:
            dict: CPT structurée comme {(parent_values...): {state: prob}}
        """
        
        # Récupérer tous les états possibles du nœud
        node_states = set()
        for observation in data:
            node_states.add(observation[node])
        node_states = sorted(list(node_states))
        
        # Cas 1: Le nœud n'a pas de parents (probabilités inconditionnelles)
        if not parents:
            return self._estimate_unconditional_cpt(data, node, node_states)
        
        # Cas 2: Le nœud a des parents
        return self._estimate_conditional_cpt(data, node, parents, node_states)
    
    def _estimate_unconditional_cpt(self, data, node, node_states):
        """Estime P(node) pour un nœud sans parents"""
        
        # Compter les occurrences de chaque état
        counts = defaultdict(int)
        for observation in data:
            counts[observation[node]] += 1
        
        # Calculer les probabilités avec lissage de Laplace
        n = len(data)
        k = len(node_states)
        
        cpt = {}
        for state in node_states:
            count = counts[state]
            
            if self.laplace_smoothing:
                # Lissage: (count + 1) / (n + k)
                prob = (count + 1) / (n + k)
            else:
                prob = count / n if n > 0 else 0
            
            cpt[state] = prob
        
        return {(): cpt}  # Clé vide pour "pas de parents"
    
    def _estimate_conditional_cpt(self, data, node, parents, node_states):
        """Estime P(node | parents) pour un nœud avec parents"""
        
        # Compter les occurrences conjointes
        joint_counts = defaultdict(lambda: defaultdict(int))
        parent_counts = defaultdict(int)
        
        for observation in data:
            # Extraire les valeurs des parents
            parent_values = tuple(observation[p] for p in parents)
            node_value = observation[node]
            
            joint_counts[parent_values][node_value] += 1
            parent_counts[parent_values] += 1
        
        # Construire la CPT
        cpt = {}
        
        # Récupérer tous les états possibles des parents
        parent_value_sets = set()
        for parent_values in parent_counts.keys():
            parent_value_sets.add(parent_values)
        
        # Calculer P(node | parent_values)
        n_states = len(node_states)
        
        for parent_values in parent_value_sets:
            total = parent_counts[parent_values]
            
            node_cpt = {}
            for state in node_states:
                count = joint_counts[parent_values].get(state, 0)
                
                if self.laplace_smoothing:
                    # Lissage: (count + 1) / (total + k)
                    prob = (count + 1) / (total + n_states)
                else:
                    prob = count / total if total > 0 else 0
                
                node_cpt[state] = prob
            
            cpt[parent_values] = node_cpt
        
        return cpt
    
    def build_all_cpts(self, data, nodes, edges):
        """
        Construit les CPT pour tous les nœuds
        
        Args:
            data (list): Données d'entraînement
            nodes (list): Liste des nœuds
            edges (list): Liste des arcs (parent, child)
            
        Returns:
            dict: {node: CPT}
        """
        
        # Calculer les parents de chaque nœud
        parents_map = {node: [] for node in nodes}
        for parent, child in edges:
            parents_map[child].append(parent)
        
        # Estimer la CPT pour chaque nœud
        self.cpts = {}
        for node in nodes:
            cpt = self.estimate_cpt(data, node, parents_map[node])
            self.cpts[node] = {
                "parents": parents_map[node],
                "cpt": cpt
            }
        
        print(f"✓ {len(self.cpts)} CPT construites")
        return self.cpts
    
    def get_probability(self, node, node_value, parent_values=None):
        """
        Récupère la probabilité P(node=node_value | parent_values)
        
        Args:
            node (str): Nœud
            node_value (str): Valeur du nœud
            parent_values (dict ou tuple): Valeurs des parents
            
        Returns:
            float: Probabilité
        """
        
        if node not in self.cpts:
            return 0.5  # Par défaut
        
        cpt_data = self.cpts[node]
        cpt = cpt_data["cpt"]
        parents = cpt_data["parents"]
        
        # Construire la clé
        if parent_values is None:
            key = ()
        elif isinstance(parent_values, dict):
            key = tuple(parent_values.get(p) for p in parents)
        else:
            key = parent_values
        
        # Récupérer la probabilité
        if key in cpt:
            return cpt[key].get(node_value, 0.5)
        
        # Si la configuration n'existe pas dans les données d'entraînement
        # Retourner une probabilité uniforme
        if key in cpt:
            n_states = len(cpt[key])
            return 1.0 / n_states
        
        return 0.5
    
    def print_cpt(self, node):
        """Affiche la CPT d'un nœud"""
        if node not in self.cpts:
            print(f"Nœud {node} non trouvé")
            return
        
        cpt_data = self.cpts[node]
        parents = cpt_data["parents"]
        cpt = cpt_data["cpt"]
        
        print(f"\nCPT pour {node}:")
        print(f"Parents: {parents}")
        
        for parent_values, node_cpt in cpt.items():
            if parent_values:
                print(f"  Sachant {dict(zip(parents, parent_values))}:")
            else:
                print(f"  Inconditionnel:")
            
            for node_value, prob in sorted(node_cpt.items()):
                print(f"    P({node}={node_value}) = {prob:.4f}")
