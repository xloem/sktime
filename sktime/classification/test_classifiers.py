# -*- coding: utf-8 -*-
""" Basic blackbox testing of the functionality and correctness of sktime classifers
"""
from sktime.classification.dictionary_based import (
    BOSSEnsemble,
    ContractableBOSS,
    TemporalDictionaryEnsemble,
    WEASEL,
    MUSE,
)
from sktime.classification.distance_based import (
    ElasticEnsemble,
    ProximityForest,
    ProximityTree,
    ProximityStump,
    KNeighborsTimeSeriesClassifier,
    ShapeDTW,
)
from sktime.classification.interval_based import{
    TimeSeriesForest,
    RandomIntervalSpectralForest,
    SupervisedTimeSeriesForest,
    CanonicalIntervalForest,
    DrCIF,
}

    ._cif import CanonicalIntervalForest
from sktime.classification.interval_based._drcif import DrCIF
from sktime.classification.shapelet_based import MrSEQLClassifier, ROCKETClassifier
from sktime.classification.shapelet_based import ShapeletTransformClassifier



from sktime.classification.hybrid import HIVECOTEV1
from sktime.classification.hybrid._catch22_forest_classifier import (
    Catch22ForestClassifier,
)
from sktime.classification.interval_based import (
    TimeSeriesForest,
    RandomIntervalSpectralForest,
)


import from dictionary_based
"IndividualBOSS",
"BOSSEnsemble",
"ContractableBOSS",
"TemporalDictionaryEnsemble",
"IndividualTDE",
"WEASEL",
"MUSE",

"ElasticEnsemble",
"ProximityTree",
"ProximityForest",
"ProximityStump",
"KNeighborsTimeSeriesClassifier",
"ShapeDTW",


def test_capabilities()
"""
this tests whether the classifier.capabilities are correct. These are boolean flags
    capabilities = {
        "multivariate"
        "unequal_length"
        "missing_values"
    }
if the variable is set to False, then the classifier should raise an exception. If it is 
set to True, it should be able to handle a problem of this kind
"""

def test_dictionary_based()

def test_distance_based()

def test_interval_based()

def test_shapelet_based()

def test_hybrid()


