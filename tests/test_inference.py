"""
Tests unitaires pour le module d'inférence
"""


def test_joint_probability():
    """Test le calcul de la probabilité jointe"""
    
    from src.network import BayesianNetwork
    from src.cpt import CPTEstimator
    from src.inference import BayesianInference
    
    # Données simples
    data = [
        {"A": "Faible", "B": "Faible", "C": "1"},
        {"A": "Faible", "B": "Normal", "C": "1"},
        {"A": "Eleve", "B": "Eleve", "C": "0"},
    ]
    
    # Construire le réseau et les CPT
    network = BayesianNetwork()
    cpt_estimator = CPTEstimator()
    cpt_estimator.build_all_cpts(data, ["A", "B", "C"], [("A", "C"), ("B", "C")])
    
    # Créer l'inférence
    inference = BayesianInference(cpt_estimator, network)
    
    # Calculer la probabilité jointe
    observation = {"A": "Faible", "B": "Faible", "C": "1"}
    prob = inference.joint_probability(observation)
    
    # Vérifier que la probabilité est positive
    assert prob > 0, "La probabilité jointe doit être positive"
    assert prob <= 1, "La probabilité jointe doit être ≤ 1"
    
    print("✓ Test joint probability passed")


def test_posterior_probability():
    """Test le calcul de la probabilité postérieure"""
    
    from src.network import BayesianNetwork
    from src.cpt import CPTEstimator
    from src.inference import BayesianInference
    
    # Données
    data = [
        {"A": "Faible", "B": "Faible", "C": "1"},
        {"A": "Faible", "B": "Normal", "C": "1"},
        {"A": "Eleve", "B": "Eleve", "C": "0"},
        {"A": "Eleve", "B": "Eleve", "C": "0"},
    ]
    
    network = BayesianNetwork()
    cpt_estimator = CPTEstimator()
    cpt_estimator.build_all_cpts(data, ["A", "B", "C"], [("A", "C"), ("B", "C")])
    
    inference = BayesianInference(cpt_estimator, network)
    
    # Calculer P(C=1 | A=Faible)
    observation = {"A": "Faible"}
    prob = inference.posterior_probability(observation, "C", "1")
    
    # Vérifier que c'est une probabilité valide
    assert 0 <= prob <= 1, f"Probabilité invalide: {prob}"
    
    print("✓ Test posterior probability passed")


if __name__ == "__main__":
    test_joint_probability()
    test_posterior_probability()
    print("\n✓ Tous les tests d'inférence passed")
