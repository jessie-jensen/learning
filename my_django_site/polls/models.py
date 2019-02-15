from django.db import models
from django.utils import timezone

import datetime as dt

# Create your models here.

class Question(models.Model):
    question_s = models.CharField(max_length=200)
    pub_dt = models.DateTimeField('date published')

    def __str__(self):
        return self.question_s

    def was_published_recently(self):
        return self.pub_dt >= (timezone.now() - dt.timedelta(days=1))



class Choice(models.Model):
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    choice_s = models.CharField(max_length=200)
    votes = models.IntegerField(default=0)

    def __str__(self):
        return self.choice_s