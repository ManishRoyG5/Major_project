from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)
    gender= models.CharField(max_length=30)
    address= models.CharField(max_length=30)


class traffic_incident_prediction(models.Model):

    Fid= models.CharField(max_length=30000)
    lat= models.CharField(max_length=30000)
    lng= models.CharField(max_length=30000)
    road_desc= models.CharField(max_length=30000)
    zip= models.CharField(max_length=30000)
    timeStamp= models.CharField(max_length=30000)
    twp= models.CharField(max_length=30000)
    addr= models.CharField(max_length=30000)
    traffic_status= models.CharField(max_length=30000)
    traffic_occured_by= models.CharField(max_length=30000)
    area_accident_occured= models.CharField(max_length=30000)
    Prediction= models.CharField(max_length=300)

class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_ratio(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)



