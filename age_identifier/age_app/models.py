from django.db import models

class UploadedImage(models.Model):
    image = models.ImageField(upload_to='uploads/')
    predicted_age = models.CharField(max_length=100)
    predicted_gender = models.CharField(max_length=100)