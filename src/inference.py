"""
Module d'inférence probabiliste
Calcul des probabilités jointes et postérieures
"""


class BayesianInference:
    """Classe pour effectuer l'inférence dans le réseau bayésien"""
    
    def __init__(self, cpt_estimator, network):
        """
        Initialise l'inférence
        
        Args:
            cpt_estimator: Objet CPTEstimator avec les CPT construites
            network: Objet BayesianNetwork avec la structure
        """
        self.cpt_estimator = cpt_estimator
        self.network = network
    
    def joint_probability(self, observation):
        """
        Calcule la probabilité jointe P(observation)
        
        Utilise la factorisation: P(X1, X2, ..., Xn) = ∏ P(Xi | Parents(Xi))
        
        Args:
            observation (dict): {variable: value}
            
        Returns:
            float: Probabilité jointe
        """
        
        probability = 1.0
        
        # Utiliser l'ordre topologique pour assurer la causalité
        for node in self.network.topological_sort():
            if node not in observation:
                continue
            
            node_value = observation[node]
            parents = self.network.get_parents(node)
            
            # Extraire les valeurs des parents
            parent_values = {p: observation.get(p) for p in parents}
            
            # Récupérer P(node | parents)
            cond_prob = self.cpt_estimator.get_probability(
                node, node_value, parent_values
            )
            
            probability *= cond_prob
        
        return probability
    
    def posterior_probability(self, observation, target_node, target_value):
        """
        Calcule P(target_node=target_value | observation)
        
        Utilise Bayes: P(A|B) = P(A,B) / P(B)
        
        Args:
            observation (dict): Observations partielles
            target_node (str): Nœud cible
            target_value: Valeur du nœud cible
            
        Returns:
            float: Probabilité postérieure
        """
        
        # Créer deux observations complètes
        obs_with_target_1 = observation.copy()
        obs_with_target_1[target_node] = target_value
        
        # Remplir les variables manquantes avec une valeur par défaut
        for node in self.network.NODES:
            if node not in obs_with_target_1:
                obs_with_target_1[node] = self._get_default_value(node)
        
        # Pour la probabilité sans le nœud cible
        obs_without_target = observation.copy()
        for node in self.network.NODES:
            if node not in obs_without_target:
                obs_without_target[node] = self._get_default_value(node)
        
        # Calculer P(observation, target_value)
        p_joint = self.joint_probability(obs_with_target_1)
        
        # Calculer P(observation) en marginalisant sur tous les états de target_node
        p_obs = 0.0
        for value in self._get_node_states(target_node):
            obs_with_value = obs_without_target.copy()
            obs_with_value[target_node] = value
            p_obs += self.joint_probability(obs_with_value)
        
        # P(target_value | observation) = P(target_value, observation) / P(observation)
        if p_obs == 0:
            return 0.5  # Probabilité par défaut
        
        return p_joint / p_obs
    
    def predict(self, observation, target_node="Outcome", threshold=0.5):
        """
        Prédit la classe du nœud cible
        
        Args:
            observation (dict): Observations
            target_node (str): Nœud à prédire (défaut: Outcome)
            threshold (float): Seuil de décision (défaut: 0.5)
            
        Returns:
            tuple: (predicted_value, probability)
        """
        
        # Pour le nœud Outcome avec valeurs 0 et 1
        states = self._get_node_states(target_node)
        
        probabilities = {}
        for state in states:
            prob = self.posterior_probability(observation, target_node, state)
            probabilities[state] = prob
        
        # Déterminer la prédiction
        if "1" in probabilities and "0" in probabilities:
            prob_1 = probabilities["1"]
            if prob_1 >= threshold:
                prediction = "1"
            else:
                prediction = "0"
        else:
            # Pour les autres nœuds, choisir l'état avec la plus grande probabilité
            prediction = max(probabilities, key=probabilities.get)
        
        return prediction, probabilities
    
    def explain_prediction(self, observation, target_node="Outcome"):
        """
        Explique la prédiction localement
        
        Args:
            observation (dict): Observations
            target_node (str): Nœud cible
            
        Returns:
            dict: Explication avec effets des parents
        """
        
        parents = self.network.get_parents(target_node)
        
        explanation = {
            "prediction": self.predict(observation, target_node)[0],
            "target_node": target_node,
            "parents_effects": {}
        }
        
        # Pour chaque parent, analyser son effet
        for parent in parents:
            parent_value = observation.get(parent)
            if not parent_value:
                continue
            
            # Probabilité avec la valeur observée
            obs_with_parent = observation.copy()
            obs_with_parent[parent] = parent_value
            
            prob_observed = self.posterior_probability(
                obs_with_parent, target_node, "1"
            )
            
            # Probabilités avec autres valeurs
            alternative_probs = {}
            for alt_value in self._get_node_states(parent):
                if alt_value == parent_value:
                    continue
                
                obs_with_alt = observation.copy()
                obs_with_alt[parent] = alt_value
                
                alt_prob = self.posterior_probability(
                    obs_with_alt, target_node, "1"
                )
                
                effect = alt_prob - prob_observed
                alternative_probs[alt_value] = {
                    "probability": alt_prob,
                    "effect": effect
                }
            
            explanation["parents_effects"][parent] = {
                "observed_value": parent_value,
                "observed_prob": prob_observed,
                "alternatives": alternative_probs
            }
        
        return explanation
    
    def _get_node_states(self, node):
        """Récupère les états possibles d'un nœud"""
        if node not in self.cpt_estimator.cpts:
            return ["Faible", "Normal", "Eleve"]
        
        cpt = self.cpt_estimator.cpts[node]["cpt"]
        
        # Récupérer les états d'une des entrées de la CPT
        for parent_values, node_cpt in cpt.items():
            return sorted(list(node_cpt.keys()))
        
        return []
    
    def _get_default_value(self, node):
        """Récupère la valeur par défaut d'un nœud"""
        states = self._get_node_states(node)
        if states:
            return states[0]  # Première valeur
        return "Faible"
