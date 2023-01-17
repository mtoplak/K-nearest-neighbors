import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from scipy.stats import mode


print("starting...")

# your path to csv file
dataset = pd.read_csv("D:\MOS\stres.csv")

#print(dataset)
print(dataset.head())

# values of our parameters without class
x = dataset.drop('Stress Level', axis=1)
# actual classes of each instance
y = dataset['Stress Level']

#print(x)
#print(y)


# evklidska razdalja
def euclideanDistance(point1, point2):
    disance = 0
    for i in range(len(point1)):
        disance += (point1[i] - point2[i]) ** 2

    return np.sqrt(disance)

"""
# example distance
a = np.array([3,4,1])
b = np.array([10,2,3])
print(euclideanDistance(a, b))
"""


def KNN(x_train, x_test, y_train, y_test, k_val):
    y_hat = [] # predictions

    # go over each line of test instances
    for test_pt in x_test.to_numpy(): # for each test instance in the test set (change it to array)
        distances = [] # storing distances to training instances

        # go over each line of learning instances
        for i in range(len(x_train)):
            # calculate the distance of the new (test) instance to each instance of the training set
            distances.append(euclideanDistance(np.array(x_train.iloc[i]), test_pt)) # iloc is a property from the pandas library... with iloc[index] we get a row of the training set

        # convert it to pandas so that we can then sort... the index will be the same as before, the distance will be calculated
        distance_data = pd.DataFrame(data=distances, columns=['distance'], index=y_train.index)

        # sort by distance, because we need the nearest ones, k_val - we get that many neighbors
        k_neighbors_list = distance_data.sort_values(by=["distance"], axis=0)[:k_val]

        # class names
        labels = y_train.loc[k_neighbors_list.index]

        # mode - 1st the most common class
        most_common_class = mode(labels, keepdims=True).mode[0]

        y_hat.append(most_common_class)
    # return predictions
    print("predicted classes: ")
    print(*y_hat)
    return y_hat

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)

y_hat_test = KNN(x_train, x_test, y_train, y_test, k_val=4)

"""
# prediction with different number of neighbors - accuracy
accuracy_vals = []
for i in range(1, 8):
    y_hat_test = KNN(x_train, x_test, y_train, y_test, k_val=i)
    accuracy_vals.append(accuracy_score(y_test, y_hat_test))

print(accuracy_vals)
"""

# accuracy with different K's
# plt.plot(range(1, 8), accuracy_vals, color = "blue", marker = 'x', linestyle= 'dashed')


# prints the true and predicted classes for the data we tested
df = pd.DataFrame({'Real Values':y_test, 'Predicted Values':y_hat_test})
print(df)



# accuracy
y_hat_test = KNN(x_train, x_test, y_train, y_test, k_val=4)
print("accuracy: ")
print(accuracy_score(y_test, y_hat_test))

# confusion matrix
cm = confusion_matrix(y_test, y_hat_test)
print("Confusion matrix: ")
print(cm)

# sensitivity for class 0
# TP / P
# let's assume that class 0 is positive
# new confusion matrix
print("[["+str(cm[0][0]) +" "+str(cm[0][1]+cm[0][2])+"]")
print("["+str(cm[1][0]+cm[2][0])+" "+str(cm[1][1]+cm[1][2]+cm[2][1]+cm[2][2])+"]]")

sensitivity = cm[0][0] / (cm[0][0]+cm[0][1]+cm[0][2])
print("sensitivity:")
print(sensitivity)

# specificity for class 0
# TN / N
# let's assume that class 0 is positive
# new confusion matrix
print("[["+str(cm[0][0]) +" "+str(cm[0][1]+cm[0][2])+"]")
print("["+str(cm[1][0]+cm[2][0])+" "+str(cm[1][1]+cm[1][2]+cm[2][1]+cm[2][2])+"]]")

specificity = (cm[1][1]+cm[1][2]+cm[2][1]+cm[2][2])/(cm[1][0]+cm[2][0]+cm[1][1]+cm[1][2]+cm[2][1]+cm[2][2])

print("specificity:")
print(specificity)


# recall, precision
report = classification_report(y_test, y_hat_test)
print(report)