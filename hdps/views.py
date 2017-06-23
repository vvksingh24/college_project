import os
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from django.shortcuts import render
from .models import Attributes
from .forms import *

# Create your views here.
def prediction(request):
	form=HDPSForm()
	prediction=0
	url = os.path.abspath('cleveland_data_raw.csv')
	names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
	dataset = pandas.read_csv(url, names=names)
	array = dataset.values
	X = array[:,0:13]
	Y = array[:,13]
	validation_size = 0.70
	seed = 10
	X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
	seed = 10
	scoring = 'accuracy'
	models=[]
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
	rf = RandomForestClassifier()
	rf.fit(X_train, Y_train)
	if request.method=='POST':
		form=form(request.POST or None)
		print form.is_valid()
		if form.is_valid():
			age=request.POST.get('age')
			sex=request.POST.get('sex')
			cp=request.POST.get('cp')
			trestbps=request.POST.get('trestbps')
			chol=request.POST.get('chol')
			fbs=request.POST.get('fbs')
			restecg=request.POST.get('restecg')
			thalach=request.POST.get('thalach')
			exang=request.POST.get('exhang')
			oldpeak=request.POST.get('oldpeak')
			slope=request.POST.get('slope')
			ca=request.POST.get('ca')
			thal=request.POST.get('thal')
			form.save()
			X_validation=[]
			X_validation.extend((age, sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal))
			predictions = rf.predict(X_validation)
			prediction=predictions
			print (predictions)
	else:
		form=HDPSForm

	print request.POST.get('age')
	context={
	'form':form,
	'prediction':prediction,
	}
	return render(request,'prediction.html',context)
def result(request,prediction):
	prediction=prediction
	context={
		'prediction':prediction
	}
	return render(request,'result.html',context)


