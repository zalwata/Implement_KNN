"""
- Program Name: cs486Fall2021AryumJeonImplementKNN.py
- Written By  : Aryum Jeon
- Date        : 04/07/2022
"""

#library
import numpy as np
from numpy.linalg import norm

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
        '''
        Predict the labels for the input features given the
        training instances.
        '''
    # YOUR CODE HERE





#application driver








"""
- Program Output

"""

"""
- Logic_Code_Output Issues

"""

