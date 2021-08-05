# -*- coding: utf-8 -*-
"""Utilities for loading datasets."""
import itertools
import os
import shutil
import tempfile
import textwrap
import zipfile
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

__all__ = [
    "load_airline",
    "load_arrow_head",
    "load_osuleaf",
    "load_italy_power_demand",
    "load_basic_motions",
    "load_japanese_vowels",
    "load_shampoo_sales",
    "load_longley",
    "load_lynx",
    "load_acsf1",
    "load_uschange",
    "load_UCR_UEA_dataset",
    "load_PBS_dataset",
    "load_unit_test",
    "load_from_tsfile_to_dataframe",
    "load_from_arff_to_dataframe",
]

__author__ = [
    "Markus Löning",
    "Sajay Ganesh",
    "@big-o",
    "Sebastiaan Koel",
    "Emilia Rose",
    "Tony Bagnall",
]

DIRNAME = "data"
MODULE = os.path.dirname(__file__)


# time series classification data sets
def _download_and_extract(url, extract_path=None):
    """
    Download and unzip datasets (helper function).

    This code was modified from
    https://github.com/tslearn-team/tslearn/blob
    /775daddb476b4ab02268a6751da417b8f0711140/tslearn/datasets.py#L28

    Parameters
    ----------
    url : string
        Url pointing to file to download
    extract_path : string, optional (default: None)
        path to extract downloaded zip to, None defaults
        to sktime/datasets/data

    Returns
    -------
    extract_path : string or None
        if successful, string containing the path of the extracted file, None
        if it wasn't succesful

    """
    file_name = os.path.basename(url)
    dl_dir = tempfile.mkdtemp()
    zip_file_name = os.path.join(dl_dir, file_name)
    urlretrieve(url, zip_file_name)

    if extract_path is None:
        extract_path = os.path.join(MODULE, "data/%s/" % file_name.split(".")[0])
    else:
        extract_path = os.path.join(extract_path, "%s/" % file_name.split(".")[0])

    try:
        if not os.path.exists(extract_path):
            os.makedirs(extract_path)
        zipfile.ZipFile(zip_file_name, "r").extractall(extract_path)
        shutil.rmtree(dl_dir)
        return extract_path
    except zipfile.BadZipFile:
        shutil.rmtree(dl_dir)
        if os.path.exists(extract_path):
            shutil.rmtree(extract_path)
        raise zipfile.BadZipFile(
            "Could not unzip dataset. Please make sure the URL is valid."
        )


def _list_downloaded_datasets(extract_path):
    """
    Return a list of all the currently downloaded datasets.

    Modified version of
    https://github.com/tslearn-team/tslearn/blob
    /775daddb476b4ab02268a6751da417b8f0711140/tslearn/datasets.py#L250

    Returns
    -------
    datasets : List
        List of the names of datasets downloaded

    """
    if extract_path is None:
        data_dir = os.path.join(MODULE, DIRNAME)
    else:
        data_dir = extract_path
    datasets = [
        path
        for path in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, path))
    ]
    return datasets


def load_UCR_UEA_dataset(name, split=None, return_X_y=False, extract_path=None):
    """
    Load dataset from UCR UEA time series archive.

    Datasets to be found here: http://www.timeseriesclassification.com/dataset.php
    Downloads and extracts dataset if not already downloaded.

    Parameters
    ----------
    name : str
        Name of data set.
        Possible strings can be found at:
        http://www.timeseriesclassification.com/dataset.php .
    split: None or str{"train", "test"}, optional (default=None)
        Whether to load the train or test partition of the problem. By
        default it loads both.
    return_X_y: bool, optional (default=False)
        If True, returns (features, target) separately instead of a single
        dataframe with columns for
        features and the target.
    extract_path : str, optional (default=None)
        Default extract path is `sktime/datasets/data/`

    Returns
    -------
    X: pandas DataFrame with m rows and c columns
        The time series data for the problem with m cases and c dimensions
    y: numpy array
        The class labels for each case in X
    """
    return _load_dataset(name, split, return_X_y, extract_path)


def _load_dataset(name, split, return_X_y, extract_path=None):
    """Load time series classification datasets (helper function)."""
    # Allow user to have non standard extract path
    if extract_path is not None:
        local_module = os.path.dirname(extract_path)
        local_dirname = extract_path
    else:
        local_module = MODULE
        local_dirname = DIRNAME

    if not os.path.exists(os.path.join(local_module, local_dirname)):
        os.makedirs(os.path.join(local_module, local_dirname))
    if name not in _list_downloaded_datasets(extract_path):
        url = "http://timeseriesclassification.com/Downloads/%s.zip" % name
        # This also tests the validitiy of the URL, can't rely on the html
        # status code as it always returns 200
        try:
            _download_and_extract(
                url,
            )
        except zipfile.BadZipFile as e:
            raise ValueError(
                "Invalid dataset name. ",
                extract_path,
                "Please make sure the dataset "
                + "is available on http://timeseriesclassification.com/.",
            ) from e
    if isinstance(split, str):
        split = split.upper()

    if split in ("TRAIN", "TEST"):
        fname = name + "_" + split + ".ts"
        abspath = os.path.join(local_module, local_dirname, name, fname)
        X, y = load_from_tsfile_to_dataframe(abspath)
    # if split is None, load both train and test set
    elif split is None:
        X = pd.DataFrame(dtype="object")
        y = pd.Series(dtype="object")
        for split in ("TRAIN", "TEST"):
            fname = name + "_" + split + ".ts"
            abspath = os.path.join(local_module, local_dirname, name, fname)
            result = load_from_tsfile_to_dataframe(abspath)
            X = pd.concat([X, pd.DataFrame(result[0])])
            y = pd.concat([y, pd.Series(result[1])])
    else:
        raise ValueError("Invalid `split` value =", split)

    # Return appropriately
    if return_X_y:
        return X, y
    else:
        X["class_val"] = pd.Series(y)
        return X


def load_osuleaf(split=None, return_X_y=False):
    """
    Load the OSULeaf time series classification problem and returns X and y.

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
    Series length:      427
    Train cases:        200
    Test cases:         242
    Number of classes:  6

    The OSULeaf data set consist of one dimensional outlines of leaves.
    The series were obtained by color image segmentation and boundary
    extraction (in the anti-clockwise direction) from digitized leaf images
    of six classes: Acer Circinatum, Acer Glabrum, Acer Macrophyllum,
    Acer Negundo, Quercus Garryanaand Quercus Kelloggii for the MSc thesis
    "Content-Based Image Retrieval: Plant Species Identification" by A Grandhi.

    Dataset details: http://www.timeseriesclassification.com/description.php
    ?Dataset=OSULeaf
    """
    name = "OSULeaf"
    return _load_dataset(name, split, return_X_y)


def load_italy_power_demand(split=None, return_X_y=False):
    """
    Load ItalyPowerDemand time series classification problem.

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
    Series length:      24
    Train cases:        67
    Test cases:         1029
    Number of classes:  2

    The data was derived from twelve monthly electrical power demand time series from
    Italy and first used in the paper "Intelligent Icons: Integrating Lite-Weight Data
    Mining and Visualization into GUI Operating Systems". The classification task is to
    distinguish days from Oct to March (inclusive) from April to September.
    Dataset details:
    http://timeseriesclassification.com/description.php?Dataset=ItalyPowerDemand
    """
    name = "ItalyPowerDemand"
    return _load_dataset(name, split, return_X_y)


def load_unit_test(split=None, return_X_y=False):
    """
    Load UnitTest time series classification problem.

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
    This is the Chinatown problem with a smaller test set, useful for rapid tests. See
    http://timeseriesclassification.com/description.php?Dataset=Chinatown
    for the full dataset
    Dimensionality:     univariate
    Series length:      24
    Train cases:        20
    Test cases:         22 (full dataset has 345)
    Number of classes:  2
    """
    name = "UnitTest"
    return _load_dataset(name, split, return_X_y)


def load_japanese_vowels(split=None, return_X_y=False):
    """
    Load the JapaneseVowels time series classification problem.

    Parameters
    ----------
    split: None or str{"train", "test"}, optional (default=None)
        Whether to load the train or test partition of the problem. By
    default it loads both.
    return_X_y: bool, optional (default=False)
        If True, returns (features, target) separately instead of a
        single dataframe with columns for
        features and the target.

    Returns
    -------
    X: pandas DataFrame with m rows and c columns
        The time series data for the problem with m cases and c dimensions
    y: numpy array
        The class labels for each case in X

    Details
    -------
    Dimensionality:     multivariate, 12
    Series length:      29
    Train cases:        270
    Test cases:         370
    Number of classes:  9

    A UCI Archive dataset. 9 Japanese-male speakers were recorded saying
    the vowels 'a' and 'e'. A '12-degree
    linear prediction analysis' is applied to the raw recordings to
    obtain time-series with 12 dimensions, a
    originally a length between 7 and 29. In this dataset, instances
    have been padded to the longest length,
    29. The classification task is to predict the speaker. Therefore,
    each instance is a transformed utterance,
    12*29 values with a single class label attached, [1...9]. The given
    training set is comprised of 30
    utterances for each speaker, however the test set has a varied
    distribution based on external factors of
    timing and experimental availability, between 24 and 88 instances per
    speaker. Reference: M. Kudo, J. Toyama
    and M. Shimbo. (1999). "Multidimensional Curve Classification Using
    Passing-Through Regions". Pattern
    Recognition Letters, Vol. 20, No. 11--13, pages 1103--1111.
    Dataset details: http://timeseriesclassification.com/description.php
    ?Dataset=JapaneseVowels
    """
    name = "JapaneseVowels"
    return _load_dataset(name, split, return_X_y)


def load_arrow_head(split=None, return_X_y=False):
    """
    Load the ArrowHead time series classification problem and returns X and y.

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
    Series length:      251
    Train cases:        36
    Test cases:         175
    Number of classes:  3

    The arrowhead data consists of outlines of the images of arrowheads. The
    shapes of the
    projectile points are converted into a time series using the angle-based
    method. The
    classification of projectile points is an important topic in
    anthropology. The classes
    are based on shape distinctions such as the presence and location of a
    notch in the
    arrow. The problem in the repository is a length normalised version of
    that used in
    Ye09shapelets. The three classes are called "Avonlea", "Clovis" and "Mix"."

    Dataset details: http://timeseriesclassification.com/description.php
    ?Dataset=ArrowHead
    """
    name = "ArrowHead"
    return _load_dataset(name, split, return_X_y)


def load_acsf1(split=None, return_X_y=False):
    """
    Load dataset on power consumption of typical appliances.

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
    Series length:      1460
    Train cases:        100
    Test cases:         100
    Number of classes:  10

    The dataset contains the power consumption of typical appliances.
    The recordings are characterized by long idle periods and some high bursts
    of energy consumption when the appliance is active.
    The classes correspond to 10 categories of home appliances;
    mobile phones (via chargers), coffee machines, computer stations
    (including monitor), fridges and freezers, Hi-Fi systems (CD players),
    lamp (CFL), laptops (via chargers), microwave ovens, printers, and
    televisions (LCD or LED)."

    Dataset details: http://www.timeseriesclassification.com/description.php?Dataset
    =ACSF1
    """
    name = "ACSF1"
    return _load_dataset(name, split, return_X_y)


def load_basic_motions(split=None, return_X_y=False):
    """
    Load the  BasicMotions time series classification problem and returns X and y.

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
    Series length:      100
    Train cases:        40
    Test cases:         40
    Number of classes:  4

    The data was generated as part of a student project where four students performed
    four activities whilst wearing a smart watch. The watch collects 3D accelerometer
    and a 3D gyroscope It consists of four classes, which are walking, resting,
    running and badminton. Participants were required to record motion a total of
    five times, and the data is sampled once every tenth of a second, for a ten second
    period.

    Dataset details: http://www.timeseriesclassification.com/description.php?Dataset
    =BasicMotions
    """
    name = "BasicMotions"
    return _load_dataset(name, split, return_X_y)


# forecasting data sets
def load_shampoo_sales():
    """
    Load the shampoo sales univariate time series dataset for forecasting.

    Returns
    -------
    y : pandas Series/DataFrame
        Shampoo sales dataset

    Details
    -------
    This dataset describes the monthly number of sales of shampoo over a 3
    year period.
    The units are a sales count.

    Dimensionality:     univariate
    Series length:      36
    Frequency:          Monthly
    Number of cases:    1


    References
    ----------
    .. [1] Makridakis, Wheelwright and Hyndman (1998) Forecasting: methods
    and applications,
        John Wiley & Sons: New York. Chapter 3.
    """
    name = "ShampooSales"
    fname = name + ".csv"
    path = os.path.join(MODULE, DIRNAME, name, fname)
    y = pd.read_csv(path, index_col=0, squeeze=True, dtype={1: np.float})
    y.index = pd.PeriodIndex(y.index, freq="M", name="Period")
    y.name = "Number of shampoo sales"
    return y


def load_longley(y_name="TOTEMP"):
    """
    Load the Longley dataset for forecasting with exogenous variables.

    Parameters
    ----------
    y_name: str, optional (default="TOTEMP")
        Name of target variable (y)

    Returns
    -------
    y: pandas.Series
        The target series to be predicted.
    X: pandas.DataFrame
        The exogenous time series data for the problem.

    Details
    -------
    This mulitvariate time series dataset contains various US macroeconomic
    variables from 1947 to 1962 that are known to be highly collinear.

    Dimensionality:     multivariate, 6
    Series length:      16
    Frequency:          Yearly
    Number of cases:    1

    Variable description:

    TOTEMP - Total employment
    GNPDEFL - Gross national product deflator
    GNP - Gross national product
    UNEMP - Number of unemployed
    ARMED - Size of armed forces
    POP - Population

    References
    ----------
    .. [1] Longley, J.W. (1967) "An Appraisal of Least Squares Programs for the
        Electronic Computer from the Point of View of the User."  Journal of
        the American Statistical Association.  62.319, 819-41.
        (https://www.itl.nist.gov/div898/strd/lls/data/LINKS/DATA/Longley.dat)
    """
    name = "Longley"
    fname = name + ".csv"
    path = os.path.join(MODULE, DIRNAME, name, fname)
    data = pd.read_csv(path, index_col=0)
    data = data.set_index("YEAR")
    data.index = pd.PeriodIndex(data.index, freq="Y", name="Period")
    data = data.astype(np.float)

    # Get target series
    y = data.pop(y_name)
    return y, data


def load_lynx():
    """
    Load the lynx univariate time series dataset for forecasting.

    Returns
    -------
    y : pandas Series/DataFrame
        Lynx sales dataset

    Details
    -------
    The annual numbers of lynx trappings for 1821–1934 in Canada. This
    time-series records the number of skins of
    predators (lynx) that were collected over several years by the Hudson's
    Bay Company. The dataset was
    taken from Brockwell & Davis (1991) and appears to be the series
    considered by Campbell & Walker (1977).

    Dimensionality:     univariate
    Series length:      114
    Frequency:          Yearly
    Number of cases:    1

    Notes
    -----
    This data shows aperiodic, cyclical patterns, as opposed to periodic,
    seasonal patterns.

    References
    ----------
    .. [1] Becker, R. A., Chambers, J. M. and Wilks, A. R. (1988). The New S
    Language. Wadsworth & Brooks/Cole.

    .. [2] Campbell, M. J. and Walker, A. M. (1977). A Survey of statistical
    work on the Mackenzie River series of
    annual Canadian lynx trappings for the years 1821–1934 and a new
    analysis. Journal of the Royal Statistical Society
    series A, 140, 411–431.
    """
    name = "Lynx"
    fname = name + ".csv"
    path = os.path.join(MODULE, DIRNAME, name, fname)
    y = pd.read_csv(path, index_col=0, squeeze=True, dtype={1: np.float})
    y.index = pd.PeriodIndex(y.index, freq="Y", name="Period")
    y.name = "Number of Lynx trappings"
    return y


def load_airline():
    """
    Load the airline univariate time series dataset [1].

    Returns
    -------
    y : pd.Series
     Time series

    Details
    -------
    The classic Box & Jenkins airline data. Monthly totals of international
    airline passengers, 1949 to 1960.

    Dimensionality:     univariate
    Series length:      144
    Frequency:          Monthly
    Number of cases:    1

    Notes
    -----
    This data shows an increasing trend, non-constant (increasing) variance
    and periodic, seasonal patterns.

    References
    ----------
    ..[1] Box, G. E. P., Jenkins, G. M. and Reinsel, G. C. (1976) Time Series
          Analysis, Forecasting and Control. Third Edition. Holden-Day.
          Series G.
    """
    name = "Airline"
    fname = name + ".csv"
    path = os.path.join(MODULE, DIRNAME, name, fname)
    y = pd.read_csv(path, index_col=0, squeeze=True, dtype={1: np.float})

    # make sure time index is properly formatted
    y.index = pd.PeriodIndex(y.index, freq="M", name="Period")
    y.name = "Number of airline passengers"
    return y


def load_uschange(y_name="Consumption"):
    """
    Load MTS dataset for forecasting Growth rates of personal consumption and income.

    Returns
    -------
    y : pandas Series
        selected column, default consumption
    X : pandas Dataframe
        columns with explanatory variables

    Details
    -------
    Percentage changes in quarterly personal consumption expenditure,
    personal disposable income, production, savings and the
    unemployment rate for the US, 1960 to 2016.


    Dimensionality:     multivariate
    Columns:            ['Quarter', 'Consumption', 'Income', 'Production',
                         'Savings', 'Unemployment']
    Series length:      188
    Frequency:          Quarterly
    Number of cases:    1

    Notes
    -----
    This data shows an increasing trend, non-constant (increasing) variance
    and periodic, seasonal patterns.

    References
    ----------
    ..fpp2: Data for "Forecasting: Principles and Practice" (2nd Edition)
    """
    name = "Uschange"
    fname = name + ".csv"
    path = os.path.join(MODULE, DIRNAME, name, fname)
    data = pd.read_csv(path, index_col=0, squeeze=True)

    # Sort by Quarter then set simple numeric index
    # TODO add support for period/datetime indexing
    # data.index = pd.PeriodIndex(data.index, freq='Y')
    data = data.sort_values("Quarter")
    data = data.reset_index(drop=True)
    data.index = pd.Int64Index(data.index)
    data.name = name
    y = data[y_name]
    if y_name != "Quarter":
        data = data.drop("Quarter", axis=1)
    X = data.drop(y_name, axis=1)
    return y, X


def load_PBS_dataset():
    """
    Load the Pharmaceutical Benefit Scheme univariate time series dataset [1].

    Returns
    -------
    y : pd.Series
     Time series

    Details
    -------
    The Pharmaceutical Benefits Scheme (PBS) is the Australian government drugs
    subsidy scheme.
    Data comprises of the numbers of scripts sold each month for immune sera
    and immunoglobulin products in Australia.


    Dimensionality:     univariate
    Series length:      204
    Frequency:          Monthly
    Number of cases:    1

    Notes
    -----
    The time series is intermittent, i.e contains small counts,
    with many months registering no sales at all,
    and only small numbers of items sold in other months.

    References
    ----------
    ..fpp3: Data for "Forecasting: Principles and Practice" (3rd Edition)
    """
    name = "PBS_dataset"
    fname = name + ".csv"
    path = os.path.join(MODULE, DIRNAME, name, fname)
    y = pd.read_csv(path, index_col=0, squeeze=True, dtype={1: np.float})

    # make sure time index is properly formatted
    y.index = pd.PeriodIndex(y.index, freq="M", name="Period")
    y.name = "Number of scripts"
    return y


def load_from_tsfile_to_dataframe(
    full_file_path_and_name,
    return_separate_X_and_y=True,
    replace_missing_vals_with="NaN",
):
    """Load data from a .ts file into a Pandas DataFrame.

    Parameters
    ----------
    full_file_path_and_name: str
        The full pathname of the .ts file to read.
    return_separate_X_and_y: bool
        true if X and Y values should be returned as separate Data Frames (
        X) and a numpy array (y), false otherwise.
        This is only relevant for data that
    replace_missing_vals_with: str
       The value that missing values in the text file should be replaced
       with prior to parsing.

    Returns
    -------
    DataFrame, ndarray
        If return_separate_X_and_y then a tuple containing a DataFrame and a
        numpy array containing the relevant time-series and corresponding
        class values.
    DataFrame
        If not return_separate_X_and_y then a single DataFrame containing
        all time-series and (if relevant) a column "class_vals" the
        associated class values.
    """
    # Initialize flags and variables used when parsing the file
    metadata_started = False
    data_started = False

    has_problem_name_tag = False
    has_timestamps_tag = False
    has_univariate_tag = False
    has_class_labels_tag = False
    has_data_tag = False

    previous_timestamp_was_int = None
    prev_timestamp_was_timestamp = None
    num_dimensions = None
    is_first_case = True
    instance_list = []
    class_val_list = []
    line_num = 0

    # Parse the file
    # print(full_file_path_and_name)
    with open(full_file_path_and_name, "r", encoding="utf-8") as file:
        for line in file:
            # Strip white space from start/end of line and change to
            # lowercase for use below
            line = line.strip().lower()
            # Empty lines are valid at any point in a file
            if line:
                # Check if this line contains metadata
                # Please note that even though metadata is stored in this
                # function it is not currently published externally
                if line.startswith("@problemname"):
                    # Check that the data has not started
                    if data_started:
                        raise TsFileParseException("metadata must come before data")
                    # Check that the associated value is valid
                    tokens = line.split(" ")
                    token_len = len(tokens)

                    if token_len == 1:
                        raise TsFileParseException(
                            "problemname tag requires an associated value"
                        )

                    # problem_name = line[len("@problemname") + 1:]
                    has_problem_name_tag = True
                    metadata_started = True

                elif line.startswith("@timestamps"):

                    # Check that the data has not started

                    if data_started:
                        raise TsFileParseException("metadata must come before data")

                    # Check that the associated value is valid

                    tokens = line.split(" ")
                    token_len = len(tokens)

                    if token_len != 2:
                        raise TsFileParseException(
                            "timestamps tag requires an associated Boolean " "value"
                        )

                    elif tokens[1] == "true":
                        timestamps = True

                    elif tokens[1] == "false":
                        timestamps = False

                    else:
                        raise TsFileParseException("invalid timestamps value")

                    has_timestamps_tag = True
                    metadata_started = True

                elif line.startswith("@univariate"):

                    # Check that the data has not started

                    if data_started:
                        raise TsFileParseException("metadata must come before data")

                    # Check that the associated value is valid

                    tokens = line.split(" ")
                    token_len = len(tokens)

                    if token_len != 2:
                        raise TsFileParseException(
                            "univariate tag requires an associated Boolean  " "value"
                        )

                    elif tokens[1] == "true":
                        # univariate = True
                        pass

                    elif tokens[1] == "false":
                        # univariate = False
                        pass

                    else:
                        raise TsFileParseException("invalid univariate value")

                    has_univariate_tag = True
                    metadata_started = True

                elif line.startswith("@classlabel"):

                    # Check that the data has not started

                    if data_started:
                        raise TsFileParseException("metadata must come before data")

                    # Check that the associated value is valid

                    tokens = line.split(" ")
                    token_len = len(tokens)

                    if token_len == 1:
                        raise TsFileParseException(
                            "classlabel tag requires an associated Boolean  " "value"
                        )

                    if tokens[1] == "true":
                        class_labels = True

                    elif tokens[1] == "false":
                        class_labels = False

                    else:
                        raise TsFileParseException("invalid classLabel value")

                    # Check if we have any associated class values

                    if token_len == 2 and class_labels:
                        raise TsFileParseException(
                            "if the classlabel tag is true then class values "
                            "must be supplied"
                        )

                    has_class_labels_tag = True
                    class_label_list = [token.strip() for token in tokens[2:]]
                    metadata_started = True

                # Check if this line contains the start of data

                elif line.startswith("@data"):

                    if line != "@data":
                        raise TsFileParseException(
                            "data tag should not have an associated value"
                        )

                    if data_started and not metadata_started:
                        raise TsFileParseException("metadata must come before data")

                    else:
                        has_data_tag = True
                        data_started = True

                # If the 'data tag has been found then metadata has been
                # parsed and data can be loaded

                elif data_started:

                    # Check that a full set of metadata has been provided

                    if (
                        not has_problem_name_tag
                        or not has_timestamps_tag
                        or not has_univariate_tag
                        or not has_class_labels_tag
                        or not has_data_tag
                    ):
                        raise TsFileParseException(
                            "a full set of metadata has not been provided "
                            "before the data"
                        )

                    # Replace any missing values with the value specified

                    line = line.replace("?", replace_missing_vals_with)

                    # Check if we dealing with data that has timestamps

                    if timestamps:

                        # We're dealing with timestamps so cannot just split
                        # line on ':' as timestamps may contain one

                        has_another_value = False
                        has_another_dimension = False

                        timestamp_for_dim = []
                        values_for_dimension = []

                        this_line_num_dim = 0
                        line_len = len(line)
                        char_num = 0

                        while char_num < line_len:

                            # Move through any spaces

                            while char_num < line_len and str.isspace(line[char_num]):
                                char_num += 1

                            # See if there is any more data to read in or if
                            # we should validate that read thus far

                            if char_num < line_len:

                                # See if we have an empty dimension (i.e. no
                                # values)

                                if line[char_num] == ":":
                                    if len(instance_list) < (this_line_num_dim + 1):
                                        instance_list.append([])

                                    instance_list[this_line_num_dim].append(
                                        pd.Series(dtype="object")
                                    )
                                    this_line_num_dim += 1

                                    has_another_value = False
                                    has_another_dimension = True

                                    timestamp_for_dim = []
                                    values_for_dimension = []

                                    char_num += 1

                                else:

                                    # Check if we have reached a class label

                                    if line[char_num] != "(" and class_labels:

                                        class_val = line[char_num:].strip()

                                        if class_val not in class_label_list:
                                            raise TsFileParseException(
                                                "the class value '"
                                                + class_val
                                                + "' on line "
                                                + str(line_num + 1)
                                                + " is not "
                                                "valid"
                                            )

                                        class_val_list.append(class_val)
                                        char_num = line_len

                                        has_another_value = False
                                        has_another_dimension = False

                                        timestamp_for_dim = []
                                        values_for_dimension = []

                                    else:

                                        # Read in the data contained within
                                        # the next tuple

                                        if line[char_num] != "(" and not class_labels:
                                            raise TsFileParseException(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " does "
                                                "not "
                                                "start "
                                                "with a "
                                                "'('"
                                            )

                                        char_num += 1
                                        tuple_data = ""

                                        while (
                                            char_num < line_len
                                            and line[char_num] != ")"
                                        ):
                                            tuple_data += line[char_num]
                                            char_num += 1

                                        if (
                                            char_num >= line_len
                                            or line[char_num] != ")"
                                        ):
                                            raise TsFileParseException(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " does "
                                                "not end"
                                                " with a "
                                                "')'"
                                            )

                                        # Read in any spaces immediately
                                        # after the current tuple

                                        char_num += 1

                                        while char_num < line_len and str.isspace(
                                            line[char_num]
                                        ):
                                            char_num += 1

                                        # Check if there is another value or
                                        # dimension to process after this tuple

                                        if char_num >= line_len:
                                            has_another_value = False
                                            has_another_dimension = False

                                        elif line[char_num] == ",":
                                            has_another_value = True
                                            has_another_dimension = False

                                        elif line[char_num] == ":":
                                            has_another_value = False
                                            has_another_dimension = True

                                        char_num += 1

                                        # Get the numeric value for the
                                        # tuple by reading from the end of
                                        # the tuple data backwards to the
                                        # last comma

                                        last_comma_index = tuple_data.rfind(",")

                                        if last_comma_index == -1:
                                            raise TsFileParseException(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " contains a tuple that has "
                                                "no comma inside of it"
                                            )

                                        try:
                                            value = tuple_data[last_comma_index + 1 :]
                                            value = float(value)

                                        except ValueError:
                                            raise TsFileParseException(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " contains a tuple that does "
                                                "not have a valid numeric "
                                                "value"
                                            )

                                        # Check the type of timestamp that
                                        # we have

                                        timestamp = tuple_data[0:last_comma_index]

                                        try:
                                            timestamp = int(timestamp)
                                            timestamp_is_int = True
                                            timestamp_is_timestamp = False

                                        except ValueError:
                                            timestamp_is_int = False

                                        if not timestamp_is_int:
                                            try:
                                                timestamp = timestamp.strip()
                                                timestamp_is_timestamp = True

                                            except ValueError:
                                                timestamp_is_timestamp = False

                                        # Make sure that the timestamps in
                                        # the file (not just this dimension
                                        # or case) are consistent

                                        if (
                                            not timestamp_is_timestamp
                                            and not timestamp_is_int
                                        ):
                                            raise TsFileParseException(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " contains a tuple that "
                                                "has an invalid timestamp '"
                                                + timestamp
                                                + "'"
                                            )

                                        if (
                                            previous_timestamp_was_int is not None
                                            and previous_timestamp_was_int
                                            and not timestamp_is_int
                                        ):
                                            raise TsFileParseException(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " contains tuples where the "
                                                "timestamp format is "
                                                "inconsistent"
                                            )

                                        if (
                                            prev_timestamp_was_timestamp is not None
                                            and prev_timestamp_was_timestamp
                                            and not timestamp_is_timestamp
                                        ):
                                            raise TsFileParseException(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " contains tuples where the "
                                                "timestamp format is "
                                                "inconsistent"
                                            )

                                        # Store the values

                                        timestamp_for_dim += [timestamp]
                                        values_for_dimension += [value]

                                        #  If this was our first tuple then
                                        #  we store the type of timestamp we
                                        #  had

                                        if (
                                            prev_timestamp_was_timestamp is None
                                            and timestamp_is_timestamp
                                        ):
                                            prev_timestamp_was_timestamp = True
                                            previous_timestamp_was_int = False

                                        if (
                                            previous_timestamp_was_int is None
                                            and timestamp_is_int
                                        ):
                                            prev_timestamp_was_timestamp = False
                                            previous_timestamp_was_int = True

                                        # See if we should add the data for
                                        # this dimension

                                        if not has_another_value:
                                            if len(instance_list) < (
                                                this_line_num_dim + 1
                                            ):
                                                instance_list.append([])

                                            if timestamp_is_timestamp:
                                                timestamp_for_dim = pd.DatetimeIndex(
                                                    timestamp_for_dim
                                                )

                                            instance_list[this_line_num_dim].append(
                                                pd.Series(
                                                    index=timestamp_for_dim,
                                                    data=values_for_dimension,
                                                )
                                            )
                                            this_line_num_dim += 1

                                            timestamp_for_dim = []
                                            values_for_dimension = []

                            elif has_another_value:
                                raise TsFileParseException(
                                    "dimension " + str(this_line_num_dim + 1) + " on "
                                    "line "
                                    + str(line_num + 1)
                                    + " ends with a ',' that "
                                    "is not followed by "
                                    "another tuple"
                                )

                            elif has_another_dimension and class_labels:
                                raise TsFileParseException(
                                    "dimension " + str(this_line_num_dim + 1) + " on "
                                    "line "
                                    + str(line_num + 1)
                                    + " ends with a ':' while "
                                    "it should list a class "
                                    "value"
                                )

                            elif has_another_dimension and not class_labels:
                                if len(instance_list) < (this_line_num_dim + 1):
                                    instance_list.append([])

                                instance_list[this_line_num_dim].append(
                                    pd.Series(dtype=np.float32)
                                )
                                this_line_num_dim += 1
                                num_dimensions = this_line_num_dim

                            # If this is the 1st line of data we have seen
                            # then note the dimensions

                            if not has_another_value and not has_another_dimension:
                                if num_dimensions is None:
                                    num_dimensions = this_line_num_dim

                                if num_dimensions != this_line_num_dim:
                                    raise TsFileParseException(
                                        "line "
                                        + str(line_num + 1)
                                        + " does not have the "
                                        "same number of "
                                        "dimensions as the "
                                        "previous line of "
                                        "data"
                                    )

                        # Check that we are not expecting some more data,
                        # and if not, store that processed above

                        if has_another_value:
                            raise TsFileParseException(
                                "dimension "
                                + str(this_line_num_dim + 1)
                                + " on line "
                                + str(line_num + 1)
                                + " ends with a ',' that is "
                                "not followed by another "
                                "tuple"
                            )

                        elif has_another_dimension and class_labels:
                            raise TsFileParseException(
                                "dimension "
                                + str(this_line_num_dim + 1)
                                + " on line "
                                + str(line_num + 1)
                                + " ends with a ':' while it "
                                "should list a class value"
                            )

                        elif has_another_dimension and not class_labels:
                            if len(instance_list) < (this_line_num_dim + 1):
                                instance_list.append([])

                            instance_list[this_line_num_dim].append(
                                pd.Series(dtype="object")
                            )
                            this_line_num_dim += 1
                            num_dimensions = this_line_num_dim

                        # If this is the 1st line of data we have seen then
                        # note the dimensions

                        if (
                            not has_another_value
                            and num_dimensions != this_line_num_dim
                        ):
                            raise TsFileParseException(
                                "line " + str(line_num + 1) + " does not have the same "
                                "number of dimensions as the "
                                "previous line of data"
                            )

                        # Check if we should have class values, and if so
                        # that they are contained in those listed in the
                        # metadata

                        if class_labels and len(class_val_list) == 0:
                            raise TsFileParseException(
                                "the cases have no associated class values"
                            )

                    else:
                        dimensions = line.split(":")

                        # If first row then note the number of dimensions (
                        # that must be the same for all cases)

                        if is_first_case:
                            num_dimensions = len(dimensions)

                            if class_labels:
                                num_dimensions -= 1

                            for _dim in range(0, num_dimensions):
                                instance_list.append([])

                            is_first_case = False

                        # See how many dimensions that the case whose data
                        # in represented in this line has

                        this_line_num_dim = len(dimensions)

                        if class_labels:
                            this_line_num_dim -= 1

                        # All dimensions should be included for all series,
                        # even if they are empty

                        if this_line_num_dim != num_dimensions:
                            raise TsFileParseException(
                                "inconsistent number of dimensions. "
                                "Expecting "
                                + str(num_dimensions)
                                + " but have read "
                                + str(this_line_num_dim)
                            )

                        # Process the data for each dimension

                        for dim in range(0, num_dimensions):
                            dimension = dimensions[dim].strip()

                            if dimension:
                                data_series = dimension.split(",")
                                data_series = [float(i) for i in data_series]
                                instance_list[dim].append(pd.Series(data_series))

                            else:
                                instance_list[dim].append(pd.Series(dtype="object"))

                        if class_labels:
                            class_val_list.append(dimensions[num_dimensions].strip())

            line_num += 1

    # Check that the file was not empty

    if line_num:
        # Check that the file contained both metadata and data

        if metadata_started and not (
            has_problem_name_tag
            and has_timestamps_tag
            and has_univariate_tag
            and has_class_labels_tag
            and has_data_tag
        ):
            raise TsFileParseException("metadata incomplete")

        elif metadata_started and not data_started:
            raise TsFileParseException("file contained metadata but no data")

        elif metadata_started and data_started and len(instance_list) == 0:
            raise TsFileParseException("file contained metadata but no data")

        # Create a DataFrame from the data parsed above

        data = pd.DataFrame(dtype=np.float32)

        for dim in range(0, num_dimensions):
            data["dim_" + str(dim)] = instance_list[dim]

        # Check if we should return any associated class labels separately

        if class_labels:
            if return_separate_X_and_y:
                return data, np.asarray(class_val_list)

            else:
                data["class_vals"] = pd.Series(class_val_list)
                return data
        else:
            return data

    else:
        raise TsFileParseException("empty file")


class TsFileParseException(Exception):
    """Should be raised when parsing a .ts file and the format is incorrect."""

    pass


class LongFormatDataParseException(Exception):
    """Should be raised when parsing a .csv file with long-formatted data."""

    pass


def load_from_arff_to_dataframe(
    full_file_path_and_name,
    has_class_labels=True,
    return_separate_X_and_y=True,
    replace_missing_vals_with="NaN",
):
    """Load data from a .ts file into a Pandas DataFrame.

    Parameters
    ----------
    full_file_path_and_name: str
        The full pathname of the .ts file to read.
    has_class_labels: bool
        true then line contains separated strings and class value contains
        list of separated strings, check for 'return_separate_X_and_y'
        false otherwise.
    return_separate_X_and_y: bool
        true then X and Y values should be returned as separate Data Frames (
        X) and a numpy array (y), false otherwise.
        This is only relevant for data.
    replace_missing_vals_with: str
       The value that missing values in the text file should be replaced
       with prior to parsing.

    Returns
    -------
    DataFrame, ndarray
        If return_separate_X_and_y then a tuple containing a DataFrame and a
        numpy array containing the relevant time-series and corresponding
        class values.
    DataFrame
        If not return_separate_X_and_y then a single DataFrame containing
        all time-series and (if relevant) a column "class_vals" the
        associated class values.
    """
    instance_list = []
    class_val_list = []

    data_started = False
    is_multi_variate = False
    is_first_case = True

    # Parse the file
    # print(full_file_path_and_name)
    with open(full_file_path_and_name, "r") as f:
        for line in f:

            if line.strip():
                if (
                    is_multi_variate is False
                    and "@attribute" in line.lower()
                    and "relational" in line.lower()
                ):
                    is_multi_variate = True

                if "@data" in line.lower():
                    data_started = True
                    continue

                # if the 'data tag has been found, the header information
                # has been cleared and now data can be loaded
                if data_started:
                    line = line.replace("?", replace_missing_vals_with)

                    if is_multi_variate:
                        if has_class_labels:
                            line, class_val = line.split("',")
                            class_val_list.append(class_val.strip())
                        dimensions = line.split("\\n")
                        dimensions[0] = dimensions[0].replace("'", "")

                        if is_first_case:
                            for _d in range(len(dimensions)):
                                instance_list.append([])
                            is_first_case = False

                        for dim in range(len(dimensions)):
                            instance_list[dim].append(
                                pd.Series(
                                    [float(i) for i in dimensions[dim].split(",")]
                                )
                            )

                    else:
                        if is_first_case:
                            instance_list.append([])
                            is_first_case = False

                        line_parts = line.split(",")
                        if has_class_labels:
                            instance_list[0].append(
                                pd.Series(
                                    [
                                        float(i)
                                        for i in line_parts[: len(line_parts) - 1]
                                    ]
                                )
                            )
                            class_val_list.append(line_parts[-1].strip())
                        else:
                            instance_list[0].append(
                                pd.Series(
                                    [float(i) for i in line_parts[: len(line_parts)]]
                                )
                            )

    x_data = pd.DataFrame(dtype=np.float32)
    for dim in range(len(instance_list)):
        x_data["dim_" + str(dim)] = instance_list[dim]

    if has_class_labels:
        if return_separate_X_and_y:
            return x_data, np.asarray(class_val_list)
        else:
            x_data["class_vals"] = pd.Series(class_val_list)

    return x_data


def write_dataframe_to_tsfile(
    data,
    path,
    problem_name="sample_data",
    timestamp=False,
    univariate=True,
    class_label=None,
    class_value_list=None,
    equal_length=False,
    series_length=-1,
    missing_values="NaN",
    comment=None,
):
    """
    Output a dataset in dataframe format to .ts file.

    Parameters
    ----------
    data: pandas dataframe
        the dataset in a dataframe to be written as a ts file
        which must be of the structure specified in the documentation
        examples/loading_data.ipynb
        index |   dim_0   |   dim_1   |    ...    |  dim_c-1
           0  | pd.Series | pd.Series | pd.Series | pd.Series
           1  | pd.Series | pd.Series | pd.Series | pd.Series
          ... |    ...    |    ...    |    ...    |    ...
           n  | pd.Series | pd.Series | pd.Series | pd.Series
    path: str
        The full path to output the ts file
    problem_name: str
        The problemName to print in the header of the ts file
        and also the name of the file.
    timestamp: {False, bool}, optional
        Indicate whether the data contains timestamps in the header.
    univariate: {True, bool}, optional
        Indicate whether the data is univariate or multivariate in the header.
        If univariate, only the first dimension will be written to file
    class_label: {list, None}, optional
        Provide class label to show the possible class values
        for classification problems in the header.
    class_value_list: {list/ndarray, []}, optional
        ndarray containing the class values for each case in classification problems
    equal_length: {False, bool}, optional
        Indicate whether each series has equal length. It only write to file if true.
    series_length: {-1, int}, optional
        Indicate each series length if they are of equal length.
        It only write to file if true.
    missing_values: {NaN, str}, optional
        Representation for missing value, default is NaN.
    comment: {None, str}, optional
        Comment text to be inserted before the header in a block.

    Returns
    -------
    None

    Notes
    -----
    This version currently does not support writing timestamp data.

    References
    ----------
    The code for writing series data into file is adopted from
    https://stackoverflow.com/questions/37877708/
    how-to-turn-a-pandas-dataframe-row-into-a-comma-separated-string
    """
    if class_value_list is None:
        class_value_list = []
    # ensure data provided is a dataframe
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data provided must be a DataFrame")
    # ensure number of cases is same as the class value list
    if len(data.index) != len(class_value_list) and len(class_value_list) > 0:
        raise IndexError(
            "The number of cases is not the same as the number of given " "class values"
        )

    if equal_length and series_length == -1:
        raise ValueError(
            "Please specify the series length for equal length time series data."
        )

    # create path if not exist
    dirt = f"{str(path)}/{str(problem_name)}/"
    try:
        os.makedirs(dirt)
    except os.error:
        pass  # raises os.error if path already exists

    # create ts file in the path
    file = open(f"{dirt}{str(problem_name)}_transform.ts", "w")

    # write comment if any as a block at start of file
    if comment:
        file.write("\n# ".join(textwrap.wrap("# " + comment)))
        file.write("\n")
    # begin writing header information
    file.write(f"@problemName {problem_name}\n")
    file.write(f"@timeStamps {str(timestamp).lower()}\n")
    file.write(f"@univariate {str(univariate).lower()}\n")

    # write equal length or series length if provided
    if equal_length:
        file.write(f"@equalLength {str(equal_length).lower()}\n")
    if series_length > 0:
        file.write(f"@seriesLength {series_length}\n")

    # write class label line
    if class_label:
        space_separated_class_label = " ".join(str(label) for label in class_label)
        file.write(f"@classLabel true {space_separated_class_label}\n")
    else:
        file.write("@class_label false\n")

    # begin writing the core data for each case
    # which are the series and the class value list if there is any
    file.write("@data\n")
    for case, value in itertools.zip_longest(data.iterrows(), class_value_list):
        for dimension in case[1:]:  # start from the first dimension
            # split the series observation into separate token
            # ignoring the header and index
            series = (
                dimension[0]
                .to_string(index=False, header=False, na_rep=missing_values)
                .split("\n")
            )
            # turn series into comma-separated row
            series = ",".join(obsv for obsv in series)
            file.write(str(series))
            # continue with another dimension for multivariate case
            if not univariate:
                file.write(":")
        if value is not None:
            file.write(f":{value}")  # write the case value if any
        file.write("\n")  # open a new line

    file.close()