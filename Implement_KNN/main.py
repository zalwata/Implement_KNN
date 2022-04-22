"""
- Program Name: cs486Fall2021AryumJeonImplementKNN.py
- Written By  : Aryum Jeon
- Date        : 04/07/2022
"""

#library
import numpy as np
from numpy.linalg import norm
import pandas as pd

#function
def euclidean(a,b):
    distance = np.linalg.norm(a-b)
    return distance

#class
class KNN:

    def __init__(self, k, random_state=None):
        self.k = k
        self.random_state = random_state
        self.labels_ = None
        self.distance = euclidean
        self.df = ''

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        try:
            dataFrame = []
            if  (len(features) == len(labels)):
                featuresList = features.tolist()
                labelsList = labels.tolist()

                for i , j in zip(featuresList, labelsList):
                    dataFrame.append((i,j))

                self.df = list(dataFrame)
        except:
            print("error occurred in 'fit' function")


    def predict(self, features: np.ndarray) -> np.array:
        """
        -pseudocode-
        foreach t in test_x:
            points = sort(nearest_points(t))
            neighbours = get_top(k, points)
            hyp = m(neighbours)
            hypotheses.append(hyp)
        :param features:
        :return: hypotheses
        """
        dataDistances = {}
        for dataPoint in self.df:
            dataPoints = dataPoint[0]
            dataDistance = self.distance(dataPoints, features)
            dataDistances[dataDistance] = dataPoint
        hypotheses = []
        k_Points = sorted(dataDistances.keys())[:self.k]
        for nearest_points in k_Points:
            hypotheses.append(dataDistances[nearest_points][-1])
        return hypotheses

#import csv file
df = pd.read_csv('titanic.csv')
print(df)










"""
- Program Output

"""

"""
- Logic_Code_Output Issues

"""

