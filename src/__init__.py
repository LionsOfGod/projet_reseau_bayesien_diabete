"""
Réseaux Bayésiens pour le Diagnostic du Diabète
Package principal

Modules disponibles:
- preprocessing: Prétraitement des données
- network: Structure du réseau bayésien (DAG)
- cpt: Estimation des tables de probabilités conditionnelles
- inference: Inférence probabiliste
- data_split: Partitionnement train/test
- evaluation: Évaluation des performances
- main: Pipeline complet
"""

__version__ = "1.0.0"
__author__ = "Henri AYAMA"
__email__ = "henri.ayama@univ-amu.fr"

from .preprocessing import DataPreprocessor
from .network import BayesianNetwork
from .cpt import CPTEstimator
from .inference import BayesianInference
from .data_split import DataSplitter
from .evaluation import Evaluator

__all__ = [
    "DataPreprocessor",
    "BayesianNetwork",
    "CPTEstimator",
    "BayesianInference",
    "DataSplitter",
    "Evaluator",
]
