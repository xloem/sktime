import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

X, y = load_iris(return_X_y=True)
tr = tree.DecisionTreeClassifier()
tr.fit(X, y)
tree.plot_tree(tr)
plt.show()

if __name__ == "__main__":
    """
    Example simple usage of scikit learn
    """
    print("I hate python")
    dummy = DummyClassifier()
    # 6 cases with 2 features
""" trainX = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2], [3, 1]])
    # 6 labels
    trainY = np.array([1, 1, 1, 2, 2, 2, 2])
    dummy.fit(trainX, trainY)
    testX = np.array([[-1, 2],[100, 200],[-333,555]])
    testPredict = dummy.predict(testX)
    testActual = np.array([1,2,1])
    proba = dummy.predict_proba(testX)
    ac = accuracy_score(testActual, testPredict)
    print(f" prediction  = {testPredict} \n actual = {testActual} \n test accuracy "
          f"={ac}")
    cm = confusion_matrix(testActual, testPredict)
    print(f" Contingency = \n {cm}")
    dt = DecisionTreeClassifier()
    dt.fit(trainX,trainY)
"""




