import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
import joblib
data = pd.read_csv('iris.csv')
print(data.head())
X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
Y = data['species']
clf = SGDClassifier(max_iter=1000, tol=1e-3)
clf.fit(X, Y)
joblib.dump(clf, 'model.pkl')
