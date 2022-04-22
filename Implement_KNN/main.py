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

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

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
#print(df)

#cleaning data
##checking null values in dataframe
#print(df.isna().sum())

##filling null values in 'Age' with median
median = df['Age'].median()
df['Age'] = df['Age'].fillna(median)

##drop columns which does not provide enough correlation with target
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)
#print(df)

##switching column position of the target ('Survived') as the last column in the dataframe
dataTitles = list(df.columns)
dataTitles[0],dataTitles[-1] = dataTitles[-1],dataTitles[0]
df = df[dataTitles]
#print(df)

#split the data into train and test set
train_set, test_set = split_train_test(df, 0.2)
#print(len(train_set), 'train instances and', len(test_set), 'test instances.')

#train_x, train_y
train_features = train_set.drop(columns = ['Survived'])
test_features = test_set.drop(columns = ['Survived'])
train_x = train_features.iloc[:, :-1].values
test_x = test_features.iloc[:, :-1].values

train_labels = train_set['Survived']
test_labels = test_set['Survived']
train_y = train_features.iloc[:, -1].values
test_y = test_features.iloc[:, -1].values

#applying same k value as sklearn implementation
KNNAlgorithm = KNN(3)
KNNAlgorithm.fit(train_x, train_y)














"""
- Program Output

"""

"""
- Logic_Code_Output Issues

"""

