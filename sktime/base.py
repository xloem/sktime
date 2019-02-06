from sklearn.base import BaseEstimator


class BaseClassifier(BaseEstimator):
    """
    Base class for classifiers, necessary for default scoring behaviour.
    """
    pass


class BaseRegressor(BaseEstimator):
    """
    Base class for regressors, necessary for default scoring behaviour.
    """
    pass
