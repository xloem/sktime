import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sktime.utils.plotting import plot_series
from sktime.utils.data_io import load_from_tsfile_to_dataframe
#from sktime.classification.distance_based import ProximityForest
from sktime.classification.kernel_based import Arsenal
from sktime.utils.data_processing import from_nested_to_2d_array
from sklearn.metrics import accuracy_score
#from sktime.clustering import (TimeSeriesKMeans,TimeSeriesKMedoids)

def clusteringExample():
    clusterer1 = TimeSeriesKMeans(n_clusters=5, max_iter=50,
                                  averaging_algorithm="mean")
    clusterer2 = TimeSeriesKMedoids()
    X, y = load_from_tsfile_to_dataframe("C:\\Temp\\NOAR\\crp.ts");
    clusterer1.fit(X)
    c = clusterer1.predict(X)





def classificationExample():
    trainX, trainY= load_from_tsfile_to_dataframe("C:\\Temp\\NOAR\\crp_TRAIN.ts");
    testX, testY= load_from_tsfile_to_dataframe("C:\\Temp\\NOAR\\crp_TEST.ts");
    afc = Arsenal()
    afc.fit(trainX,trainY)
    preds = afc.predict(testX)
    print(" Test set accuracy the Arsenal",accuracy_score(preds,testY))

    npTrain=from_nested_to_2d_array(trainX, return_numpy=True)
    npTest=from_nested_to_2d_array(testX, return_numpy=True)
    rf = RandomForestClassifier(n_estimators=50)

    rf.fit(npTrain,trainY)
    preds = rf.predict(npTest)
    print(" Test set accuracy Random Forest",accuracy_score(preds,testY))


if __name__ == "__main__":
    #clusteringExample()
    classificationExample()






