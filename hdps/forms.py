from django import forms
from .models import  Attributes
 
class HDPSForm(forms.ModelForm):
 	class Meta:
 		model=Attributes
 		fields=['age','sex','cp','tresbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
 		

 	
