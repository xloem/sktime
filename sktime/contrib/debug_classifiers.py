# -*- coding: utf-8 -*-
"""
Working area to test classifiers
"""

from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier

import sktime.contrib.experiments as exp


def knn_debugging():
    """ Problem with the distance functions and the data alignment"""
    dataset = "UnitTest"
    dataset = "BirdChicken"
    dir = "C:/Code/sktime/sktime/datasets/data/"
    dir = "Z:/ArchiveData/Univariate_ts/"
    met = "euclidean"
    results = "C:/Temp/sktimeDebug/"
    print(" problem = " + dataset + " writing to " + results)
    ed = KNeighborsTimeSeriesClassifier(metric=met)
    exp.run_experiment(
        overwrite=True,
        problem_path=dir,
        results_path=results,
        cls_name="1-NN-ED",
        classifier=ed,
        dataset=dataset,
        train_file=False,
    )


if __name__ == "__main__":
    knn_debugging()



