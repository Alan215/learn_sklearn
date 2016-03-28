import  numpy as np
from sklearn import datasets
from sklearn.neighbors import  KNeighborsClassifier

np.random.seed(0)
iris = datasets.load_iris()
indices = np.random.permutation(len(iris.data))
iris_X_train = iris.data[indices[:-10]]
iris_y_train = iris.target[indices[:-10]]
iris_X_test = iris.data[indices[-10:]]
iris_y_test = iris.target[indices[-10:]]

knn = KNeighborsClassifier()
knn.fit(iris_X_train, iris_y_train)
print knn
print knn.predict(iris_X_test)
print iris_y_test