"""
Tests unitaires pour le module CPT
"""


def test_cpt_estimation():
    """Test l'estimation des CPT"""
    
    from src.cpt import CPTEstimator
    
    # Données synthétiques simples
    data = [
        {"A": "Faible", "B": "Faible", "C": 1},
        {"A": "Faible", "B": "Faible", "C": 1},
        {"A": "Eleve", "B": "Normal", "C": 0},
        {"A": "Eleve", "B": "Eleve", "C": 0},
    ]
    
    estimator = CPTEstimator()
    cpt = estimator.estimate_cpt(data, "C", ["A", "B"])
    
    # Vérifier que la CPT a été construite
    assert cpt is not None
    assert len(cpt) > 0
    
    print("✓ Test CPT estimation passed")


def test_laplace_smoothing():
    """Test le lissage de Laplace"""
    
    from src.cpt import CPTEstimator
    
    # Données avec une configuration absente
    data = [
        {"A": "Faible", "B": "Normal"},
        {"A": "Faible", "B": "Normal"},
        {"A": "Eleve", "B": "Eleve"},
    ]
    
    estimator = CPTEstimator(laplace_smoothing=True)
    cpt = estimator.estimate_cpt(data, "B", ["A"])
    
    # Vérifier qu'aucune probabilité n'est exactement 0
    for parent_values, node_cpt in cpt.items():
        for state, prob in node_cpt.items():
            assert prob > 0, f"Probabilité zéro trouvée: {prob}"
    
    print("✓ Test Laplace smoothing passed")


if __name__ == "__main__":
    test_cpt_estimation()
    test_laplace_smoothing()
    print("\n✓ Tous les tests CPT passed")
