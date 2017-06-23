from django.db import models

# Create your models here.
class Attributes(models.Model):
	age=models.IntegerField(blank=False)
	sex=models.IntegerField(blank=False)
	cp=models.IntegerField(blank=False)
	tresbps=models.IntegerField(blank=False)
	chol=models.IntegerField(blank=False)
	fbs=models.IntegerField(blank=False)
	restecg=models.IntegerField(blank=False)
	thalach=models.IntegerField(blank=False)
	exang=models.IntegerField(blank=False)
	oldpeak=models.FloatField(blank=False)
	slope=models.FloatField(blank=False)
	ca=models.IntegerField(blank=False)
	thal=models.IntegerField(blank=False)
	num=models.IntegerField()
