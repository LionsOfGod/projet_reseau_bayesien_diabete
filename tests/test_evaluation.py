"""
Tests unitaires pour le module d'évaluation
"""


def test_confusion_matrix():
    """Test le calcul de la matrice de confusion"""
    
    from src.evaluation import Evaluator
    
    predictions = ["1", "0", "1", "1", "0"]
    true_labels = ["1", "0", "1", "0", "0"]
    
    evaluator = Evaluator()
    evaluator.evaluate(predictions, true_labels)
    
    # Vérifier les valeurs
    assert evaluator.tp == 2, f"TP incorrect: {evaluator.tp}"
    assert evaluator.fp == 1, f"FP incorrect: {evaluator.fp}"
    assert evaluator.tn == 2, f"TN incorrect: {evaluator.tn}"
    assert evaluator.fn == 0, f"FN incorrect: {evaluator.fn}"
    
    print("✓ Test confusion matrix passed")


def test_precision_recall():
    """Test le calcul de la précision et du rappel"""
    
    from src.evaluation import Evaluator
    
    predictions = ["1", "0", "1", "1", "0"]
    true_labels = ["1", "0", "1", "0", "0"]
    
    evaluator = Evaluator()
    evaluator.evaluate(predictions, true_labels)
    
    # Précision = TP / (TP + FP) = 2 / 3
    expected_precision = 2 / 3
    assert abs(evaluator.precision() - expected_precision) < 0.01
    
    # Rappel = TP / (TP + FN) = 2 / 2 = 1
    expected_recall = 1.0
    assert abs(evaluator.recall() - expected_recall) < 0.01
    
    print("✓ Test precision recall passed")


def test_f1_score():
    """Test le calcul du F1-score"""
    
    from src.evaluation import Evaluator
    
    predictions = ["1", "0", "1", "1", "0"]
    true_labels = ["1", "0", "1", "0", "0"]
    
    evaluator = Evaluator()
    evaluator.evaluate(predictions, true_labels)
    
    f1 = evaluator.f1_score()
    
    # F1 = 2 * (P * R) / (P + R) = 2 * (2/3 * 1) / (2/3 + 1) = 4/5 = 0.8
    expected_f1 = 0.8
    assert abs(f1 - expected_f1) < 0.01
    
    print("✓ Test F1-score passed")


def test_accuracy():
    """Test le calcul de l'exactitude"""
    
    from src.evaluation import Evaluator
    
    predictions = ["1", "0", "1", "1", "0"]
    true_labels = ["1", "0", "1", "0", "0"]
    
    evaluator = Evaluator()
    evaluator.evaluate(predictions, true_labels)
    
    # Accuracy = (TP + TN) / Total = 4 / 5 = 0.8
    expected_accuracy = 0.8
    assert abs(evaluator.accuracy() - expected_accuracy) < 0.01
    
    print("✓ Test accuracy passed")


if __name__ == "__main__":
    test_confusion_matrix()
    test_precision_recall()
    test_f1_score()
    test_accuracy()
    print("\n✓ Tous les tests d'évaluation passed")
