"""
Module de partitionnement des données
Division train/test avec contrôle de randomness
"""


class DataSplitter:
    """Classe pour partitionner les données"""
    
    @staticmethod
    def train_test_split(data, test_size=0.2, random_state=None):
        """
        Divise les données en ensemble d'entraînement et de test
        
        Args:
            data (list): Liste des observations
            test_size (float): Proportion du test set (défaut: 0.2)
            random_state (int): Seed pour la reproductibilité
            
        Returns:
            tuple: (train_data, test_data)
        """
        
        n = len(data)
        n_test = int(n * test_size)
        n_train = n - n_test
        
        # Initialiser le générateur aléatoire si random_state est fourni
        if random_state is not None:
            import random
            random.seed(random_state)
        
        # Mélanger les indices
        indices = list(range(n))
        _shuffle(indices)
        
        # Partitionner
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
        
        train_data = [data[i] for i in train_indices]
        test_data = [data[i] for i in test_indices]
        
        return train_data, test_data
    
    @staticmethod
    def stratified_split(data, test_size=0.2, target_col="Outcome", random_state=None):
        """
        Divise les données en maintenant la distribution des classes
        (Stratified train/test split)
        
        Args:
            data (list): Liste des observations
            test_size (float): Proportion du test set
            target_col (str): Colonne cible
            random_state (int): Seed
            
        Returns:
            tuple: (train_data, test_data)
        """
        
        if random_state is not None:
            import random
            random.seed(random_state)
        
        # Séparer par classe
        class_groups = {}
        for observation in data:
            target = observation[target_col]
            if target not in class_groups:
                class_groups[target] = []
            class_groups[target].append(observation)
        
        train_data = []
        test_data = []
        
        # Pour chaque classe, diviser avec la même proportion
        for class_value, class_data in class_groups.items():
            n = len(class_data)
            n_test = int(n * test_size)
            
            # Mélanger
            indices = list(range(n))
            _shuffle(indices)
            
            # Ajouter aux ensembles
            for i in indices[:n_test]:
                test_data.append(class_data[i])
            
            for i in indices[n_test:]:
                train_data.append(class_data[i])
        
        return train_data, test_data
    
    @staticmethod
    def cross_validation_split(data, n_folds=5, random_state=None):
        """
        Divise les données en n_folds pour la validation croisée
        
        Args:
            data (list): Liste des observations
            n_folds (int): Nombre de folds
            random_state (int): Seed
            
        Returns:
            list: Liste de tuples (train, test) pour chaque fold
        """
        
        if random_state is not None:
            import random
            random.seed(random_state)
        
        n = len(data)
        fold_size = n // n_folds
        
        # Mélanger
        indices = list(range(n))
        _shuffle(indices)
        
        folds = []
        for fold_idx in range(n_folds):
            start = fold_idx * fold_size
            end = start + fold_size
            
            if fold_idx == n_folds - 1:
                # Dernier fold prend les restants
                end = n
            
            test_indices = indices[start:end]
            train_indices = indices[:start] + indices[end:]
            
            train_data = [data[i] for i in train_indices]
            test_data = [data[i] for i in test_indices]
            
            folds.append((train_data, test_data))
        
        return folds


def _shuffle(lst):
    """
    Mélange une liste in-place (Fisher-Yates)
    
    Args:
        lst (list): Liste à mélanger
    """
    n = len(lst)
    for i in range(n - 1, 0, -1):
        j = _random_int(0, i)
        lst[i], lst[j] = lst[j], lst[i]


def _random_int(a, b):
    """Génère un entier aléatoire entre a et b inclus"""
    import random
    return random.randint(a, b)
