"""
Module de définition du réseau bayésien
Structure du DAG (Directed Acyclic Graph)
"""


class BayesianNetwork:
    """Classe représentant la structure du réseau bayésien"""
    
    # Liste des variables (nœuds)
    NODES = [
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
    
    # Relations causales (parent → enfant)
    EDGES = [
        ("Pregnancies", "Age"),
        ("SkinThickness", "Insulin"),
        ("SkinThickness", "BMI"),
        ("Glucose", "Insulin"),
        ("Glucose", "Outcome"),
        ("BMI", "Outcome"),
        ("Age", "Outcome"),
        ("Insulin", "Outcome")
    ]
    
    def __init__(self):
        """Initialise la structure du réseau"""
        self.parents = self._compute_parents()
        self.children = self._compute_children()
    
    def _compute_parents(self):
        """
        Calcule la liste des parents pour chaque nœud
        
        Returns:
            dict: {nœud: [parents]}
        """
        parents = {node: [] for node in self.NODES}
        
        for parent, child in self.EDGES:
            parents[child].append(parent)
        
        return parents
    
    def _compute_children(self):
        """
        Calcule la liste des enfants pour chaque nœud
        
        Returns:
            dict: {nœud: [enfants]}
        """
        children = {node: [] for node in self.NODES}
        
        for parent, child in self.EDGES:
            children[parent].append(child)
        
        return children
    
    def get_parents(self, node):
        """
        Retourne les parents d'un nœud
        
        Args:
            node (str): Nom du nœud
            
        Returns:
            list: Liste des parents
        """
        return self.parents.get(node, [])
    
    def get_children(self, node):
        """
        Retourne les enfants d'un nœud
        
        Args:
            node (str): Nom du nœud
            
        Returns:
            list: Liste des enfants
        """
        return self.children.get(node, [])
    
    def is_acyclic(self):
        """
        Vérifie que le DAG est acyclique (pas de boucles)
        
        Returns:
            bool: True si acyclique
        """
        # Utilise un parcours DFS pour détecter les cycles
        
        def has_cycle(node, visited, rec_stack):
            visited.add(node)
            rec_stack.add(node)
            
            for child in self.children.get(node, []):
                if child not in visited:
                    if has_cycle(child, visited, rec_stack):
                        return True
                elif child in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        visited = set()
        for node in self.NODES:
            if node not in visited:
                if has_cycle(node, visited, set()):
                    return False
        
        return True
    
    def topological_sort(self):
        """
        Retourne l'ordre topologique des nœuds
        
        Returns:
            list: Nœuds triés topologiquement
        """
        in_degree = {node: 0 for node in self.NODES}
        
        for parent, child in self.EDGES:
            in_degree[child] += 1
        
        queue = [node for node in self.NODES if in_degree[node] == 0]
        result = []
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            for child in self.children[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        
        return result
    
    def print_structure(self):
        """Affiche la structure du réseau"""
        print("\n" + "="*60)
        print("STRUCTURE DU RESEAU BAYESIEN")
        print("="*60)
        print(f"\nNœuds ({len(self.NODES)}): {', '.join(self.NODES)}")
        print(f"\nArcs ({len(self.EDGES)}):")
        
        for parent, child in self.EDGES:
            print(f"  {parent} → {child}")
        
        print(f"\nOrdre topologique: {' → '.join(self.topological_sort())}")
        print(f"Acyclique: {self.is_acyclic()}")
        print("="*60 + "\n")
