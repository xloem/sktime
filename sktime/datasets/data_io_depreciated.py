# -*- coding: utf-8 -*-
"""Functions for the input and output of data and results."""

import os

import numpy as np
import pandas as pd

from datasets._data_io import LongFormatDataParseException
from sktime.datatypes._panel._convert import _make_column_names, from_long_to_nested


def load_from_ucr_tsv_to_dataframe(
    full_file_path_and_name, return_separate_X_and_y=True
):
    """Load data from a .tsv file into a Pandas DataFrame.

    Parameters
    ----------
    full_file_path_and_name: str
        The full pathname of the .tsv file to read.
    return_separate_X_and_y: bool
        true then X and Y values should be returned as separate Data Frames (
        X) and a numpy array (y), false otherwise.
        This is only relevant for data.

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
    df = pd.read_csv(full_file_path_and_name, sep="\t", header=None)
    y = df.pop(0).values
    df.columns -= 1
    X = pd.DataFrame()
    X["dim_0"] = [pd.Series(df.iloc[x, :]) for x in range(len(df))]
    if return_separate_X_and_y is True:
        return X, y
    X["class_val"] = y
    return X


def load_from_long_to_dataframe(full_file_path_and_name, separator=","):
    """Load data from a long format file into a Pandas DataFrame.

    Parameters
    ----------
    full_file_path_and_name: str
        The full pathname of the .csv file to read.
    separator: str
        The character that the csv uses as a delimiter

    Returns
    -------
    DataFrame
        A dataframe with sktime-formatted data
    """
    data = pd.read_csv(full_file_path_and_name, sep=separator, header=0)
    # ensure there are 4 columns in the long_format table
    if len(data.columns) != 4:
        raise LongFormatDataParseException("dataframe must contain 4 columns of data")

    # ensure that all columns contain the correct data types
    if (
        not data.iloc[:, 0].dtype == "int64"
        or not data.iloc[:, 1].dtype == "int64"
        or not data.iloc[:, 2].dtype == "int64"
        or not data.iloc[:, 3].dtype == "float64"
    ):
        raise LongFormatDataParseException(
            "one or more data columns contains data of an incorrect type"
        )

    data = from_long_to_nested(data)
    return data


# left here for now, better elsewhere later perhaps
def generate_example_long_table(num_cases=50, series_len=20, num_dims=2):
    """Generate example from long table format file.

    Parameters
    ----------
    num_cases: int
        Number of cases.
    series_len: int
        Length of the series.
    num_dims: int
        Number of dimensions.

    Returns
    -------
    DataFrame
    """
    rows_per_case = series_len * num_dims
    total_rows = num_cases * series_len * num_dims

    case_ids = np.empty(total_rows, dtype=np.int)
    idxs = np.empty(total_rows, dtype=np.int)
    dims = np.empty(total_rows, dtype=np.int)
    vals = np.random.rand(total_rows)

    for i in range(total_rows):
        case_ids[i] = int(i / rows_per_case)
        rem = i % rows_per_case
        dims[i] = int(rem / series_len)
        idxs[i] = rem % series_len

    df = pd.DataFrame()
    df["case_id"] = pd.Series(case_ids)
    df["dim_id"] = pd.Series(dims)
    df["reading_id"] = pd.Series(idxs)
    df["value"] = pd.Series(vals)
    return df


def make_multi_index_dataframe(n_instances=50, n_columns=3, n_timepoints=20):
    """Generate example multi-index DataFrame.

    Parameters
    ----------
    n_instances : int
        Number of instances.

    n_columns : int
        Number of columns (series) in multi-indexed DataFrame.

    n_timepoints : int
        Number of timepoints per instance-column pair.

    Returns
    -------
    mi_df : pd.DataFrame
        The multi-indexed DataFrame with
        shape (n_instances*n_timepoints, n_column).
    """
    # Make long DataFrame
    long_df = generate_example_long_table(
        num_cases=n_instances, series_len=n_timepoints, num_dims=n_columns
    )
    # Make Multi index DataFrame
    mi_df = long_df.set_index(["case_id", "reading_id"]).pivot(columns="dim_id")
    mi_df.columns = _make_column_names(n_columns)

    return mi_df


def write_results_to_uea_format(
    estimator_name,
    dataset_name,
    y_pred,
    output_path,
    full_path=True,
    y_true=None,
    predicted_probs=None,
    split="TEST",
    resample_seed=0,
    second_line="No Parameter Info",
    third_line="N/A",
):
    """Write the predictions for an experiment in the standard format used by sktime.

    Parameters
    ----------
    estimator_name : str,
        Name of the object that made the predictions, written to file and can
        deterimine file structure of output_root is True
    dataset_name : str
        name of the problem the classifier was built on
    y_pred : np.array
        predicted values
    output_path : str
        Path where to put results. Either a root path, or a full path
    full_path : boolean, default = True
        If False, then the standard file structure is created. If false, results are
        written directly to the directory passed in output_path
    y_true : np.array, default = None
        Actual values, written to file with the predicted values if present
    predicted_probs :  np.ndarray, default = None
        Estimated class probabilities. If passed, these are written after the
        predicted values. Regressors should not pass anything
    split : str, default = "TEST"
        Either TRAIN or TEST, depending on the results, influences file name.
    resample_seed : int, default = 0
        Indicates what data
    second_line : str
        unstructured, used for predictor parameters
    third_line : str
        summary performance information (see comment below)
    """
    if len(y_true) != len(y_pred):
        raise IndexError(
            "The number of predicted values is not the same as the "
            "number of actual class values"
        )
    # If the full directory path is not passed, make the standard structure
    if not full_path:
        output_path = (
            str(output_path)
            + "/"
            + str(estimator_name)
            + "/Predictions/"
            + str(dataset_name)
            + "/"
        )
    try:
        os.makedirs(output_path)
    except os.error:
        pass  # raises os.error if path already exists, so just ignore this

    if split == "TRAIN" or split == "train":
        train_or_test = "train"
    elif split == "TEST" or split == "test":
        train_or_test = "test"
    else:
        raise ValueError("Unknown 'split' value - should be TRAIN/train or TEST/test")

    file = open(
        str(output_path)
        + "/"
        + str(train_or_test)
        + "Resample"
        + str(resample_seed)
        + ".csv",
        "w",
    )

    # the first line of the output file is in the form of:
    # <classifierName>,<datasetName>,<train/test>
    file.write(
        str(estimator_name) + "," + str(dataset_name) + "," + str(train_or_test) + "\n"
    )

    # the second line of the output is free form and estimator-specific; usually this
    # will record info such as build time, paramater options used, any constituent model
    # names for ensembles, etc.
    file.write(str(second_line) + "\n")

    # the third line of the file is the accuracy (should be between 0 and 1
    # inclusive). If this is a train output file then it will be a training estimate
    # of the classifier on the training data only (e.g. 10-fold cv, leave-one-out cv,
    # etc.). If this is a test output file, it should be the output of the estimator
    # on the test data (likely trained on the training data for a-priori parameter
    # optimisation)
    file.write(str(third_line) + "\n")

    # from line 4 onwards each line should include the actual and predicted class
    # labels (comma-separated). If present, for each case, the probabilities of
    # predicting every class value for this case should also be appended to the line (
    # a space is also included between the predicted value and the predict_proba). E.g.:
    #
    # if predict_proba data IS provided for case i:
    #   y_true[i], y_pred[i],,prob_class_0[i],
    #   prob_class_1[i],...,prob_class_c[i]
    #
    # if predict_proba data IS NOT provided for case i:
    #   y_true[i], y_pred[i]
    # If y_true is None (if clustering), y_true[i] is replaced with ? to indicate
    # missing
    if y_true is None:
        for i in range(0, len(y_pred)):
            file.write("?," + str(y_pred[i]))
            if predicted_probs is not None:
                file.write(",")
                for j in predicted_probs[i]:
                    file.write("," + str(j))
            file.write("\n")
    else:
        for i in range(0, len(y_pred)):
            file.write(str(y_true[i]) + "," + str(y_pred[i]))
            if predicted_probs is not None:
                file.write(",")
                for j in predicted_probs[i]:
                    file.write("," + str(j))
            file.write("\n")
    file.close()


