"""
Tests unitaires pour le module data_split
"""


def test_train_test_split():
    """Test le partitionnement train/test"""
    
    from src.data_split import DataSplitter
    
    # Données synthétiques
    data = [
        {"A": 1, "B": "x"},
        {"A": 2, "B": "y"},
        {"A": 3, "B": "z"},
        {"A": 4, "B": "x"},
        {"A": 5, "B": "y"},
        {"A": 6, "B": "z"},
        {"A": 7, "B": "x"},
        {"A": 8, "B": "y"},
        {"A": 9, "B": "z"},
        {"A": 10, "B": "x"},
    ]
    
    splitter = DataSplitter()
    train, test = splitter.train_test_split(data, test_size=0.2, random_state=42)
    
    # Vérifier les tailles
    assert len(train) == 8, f"Train size incorrect: {len(train)}"
    assert len(test) == 2, f"Test size incorrect: {len(test)}"
    
    # Vérifier qu'il n'y a pas de chevauchement
    train_ids = set(d["A"] for d in train)
    test_ids = set(d["A"] for d in test)
    assert len(train_ids & test_ids) == 0, "Chevauchement entre train et test"
    
    print("✓ Test train/test split passed")


def test_stratified_split():
    """Test le partitionnement stratifié"""
    
    from src.data_split import DataSplitter
    
    # Données déséquilibrées (2x plus de classe 0)
    data = [
        {"target": "0", "value": i} for i in range(100)
    ] + [
        {"target": "1", "value": i} for i in range(50)
    ]
    
    splitter = DataSplitter()
    train, test = splitter.stratified_split(
        data, 
        test_size=0.2, 
        target_col="target",
        random_state=42
    )
    
    # Calculer les proportions
    train_pos_ratio = sum(1 for d in train if d["target"] == "1") / len(train)
    test_pos_ratio = sum(1 for d in test if d["target"] == "1") / len(test)
    
    # Les proportions doivent être similaires (2/3 négatif, 1/3 positif)
    expected_ratio = 50 / 150
    
    assert abs(train_pos_ratio - expected_ratio) < 0.05, \
        f"Train ratio {train_pos_ratio} not close to {expected_ratio}"
    assert abs(test_pos_ratio - expected_ratio) < 0.05, \
        f"Test ratio {test_pos_ratio} not close to {expected_ratio}"
    
    print("✓ Test stratified split passed")


def test_cross_validation_split():
    """Test la validation croisée"""
    
    from src.data_split import DataSplitter
    
    # Créer 100 données
    data = [{"id": i} for i in range(100)]
    
    splitter = DataSplitter()
    folds = splitter.cross_validation_split(data, n_folds=5, random_state=42)
    
    # Vérifier qu'il y a 5 folds
    assert len(folds) == 5, f"Nombre de folds incorrect: {len(folds)}"
    
    # Vérifier que chaque fold a les bonnes tailles
    for fold_idx, (train, test) in enumerate(folds):
        # Chaque fold test devrait avoir ~20 observations
        assert 18 <= len(test) <= 22, \
            f"Fold {fold_idx}: test size incorrect: {len(test)}"
        
        # Chaque fold train devrait avoir ~80 observations
        assert 78 <= len(train) <= 82, \
            f"Fold {fold_idx}: train size incorrect: {len(train)}"
        
        # Pas de chevauchement
        train_ids = set(d["id"] for d in train)
        test_ids = set(d["id"] for d in test)
        assert len(train_ids & test_ids) == 0, \
            f"Fold {fold_idx}: chevauchement détecté"
    
    # Vérifier que toutes les données sont utilisées exactement une fois en test
    all_test_ids = set()
    for train, test in folds:
        test_ids = set(d["id"] for d in test)
        all_test_ids |= test_ids
    
    assert len(all_test_ids) == 100, \
        f"Pas toutes les données utilisées: {len(all_test_ids)}/100"
    
    print("✓ Test cross-validation split passed")


if __name__ == "__main__":
    test_train_test_split()
    test_stratified_split()
    test_cross_validation_split()
    print("\n✓ Tous les tests data_split passed")
