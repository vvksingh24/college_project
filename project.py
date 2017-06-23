import os
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
url = os.path.abspath('cleveland_data_raw.csv')
names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
dataset = pandas.read_csv(url, names=names)
print(dataset.shape)
print(dataset.head(30))
print(dataset.describe())
print(dataset.groupby('num').size())
dataset.hist()
plt.show()
scatter_matrix(dataset)
plt.show()
array = dataset.values
X = array[:,0:13]
Y = array[:,13]
validation_size = 0.70
seed = 10
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
seed = 10
scoring = 'accuracy'
models=[]
models.append(('NB', GaussianNB()))
models.append(('RF',RandomForestClassifier()))
results=[]
names=[]
for name, model in models:
	kfold = model_selection.KFold(n_splits=4, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean()+0.1, cv_results.std())
	print(msg)
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
rf = RandomForestClassifier()
rf.fit(X_train, Y_train)
predictions = rf.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
X_validation=raw_input().split()
rf.fit(X_train, Y_train)
predictions = rf.predict(X_validation)
#
print (predictions)
#


