from knn import KNN
from sklearn import datasets
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as matplot

cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# matplot.figure()
# matplot.scatter(X[:,2],X[:,3], c=y, cmap=cmap, edgecolor='k', s=20)
# matplot.show()


k = 3
classifier = KNN(k=k)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
print(predictions)
