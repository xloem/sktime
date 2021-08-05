from datasets._data_io import _load_dataset


def load_gunpoint(split=None, return_X_y=False):
    """
    Load the GunPoint time series classification problem and returns X and y.

    Parameters
    ----------
    split: None or str{"train", "test"}, optional (default=None)
        Whether to load the train or test partition of the problem. By
        default it loads both.
    return_X_y: bool, optional (default=False)
        If True, returns (features, target) separately instead of a single
        dataframe with columns for
        features and the target.

    Returns
    -------
    X: pandas DataFrame with m rows and c columns
        The time series data for the problem with m cases and c dimensions
    y: numpy array
        The class labels for each case in X
    Details
    -------
    Dimensionality:     univariate
    Series length:      150
    Train cases:        50
    Test cases:         150
    Number of classes:  2
    This dataset involves one female actor and one male actor making a
    motion with their
    hand. The two classes are: Gun-Draw and Point: For Gun-Draw the actors
    have their
    hands by their sides. They draw a replicate gun from a hip-mounted
    holster, point it
    at a target for approximately one second, then return the gun to the
    holster, and
    their hands to their sides. For Point the actors have their gun by their
    sides.
    They point with their index fingers to a target for approximately one
    second, and
    then return their hands to their sides. For both classes, we tracked the
    centroid
    of the actor's right hands in both X- and Y-axes, which appear to be highly
    correlated. The data in the archive is just the X-axis.

    Dataset details: http://timeseriesclassification.com/description.php
    ?Dataset=GunPoint
    """
    name = "GunPoint"
    return _load_dataset(name, split, return_X_y)