from sqlalchemy import Column, Integer, String, Float
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy()

class TestingModel(db.Model):
    __tablename__ = 'testing'

    id = Column(Integer, primary_key=True, autoincrement=True, unique=True)
    tweet = Column(String(255))
    slug = Column(String(255), unique=True)

class TrainingModel(db.Model):
    __tablename__ = 'training'

    id = Column(Integer, primary_key=True, autoincrement=True, unique=True)
    tweet = Column(String(255))
    slug = Column(String(255), unique=True)
    label = Column(String(25))
    timestamp = Column(String(100))


class CrawlingTweet(db.Model):
    __tablename__ = 'crawling_tweet'

    id = Column(Integer, primary_key=True, autoincrement=True, unique=True)
    tweet = Column(String(255))
    preprocessing_result = Column(String(255))  
    label = Column(String(25))  

class PreprocessingTraining(db.Model):
    __tablename__ = 'preprocessing_training'

    id = Column(Integer, primary_key=True, autoincrement=True, unique=True)
    tweet = Column(String(255))
    result = Column(String(255))
    slug = Column(String(255), unique=True)
    label = Column(String(25))

class ClassificationTraining(db.Model):
    __tablename__ = 'classification_training'

    id = Column(Integer, primary_key=True, autoincrement=True, unique=True)
    tweet = Column(String(255))
    label = Column(String(25))
    result_classification = Column(String(25))
    slug = Column(String(255), unique=True)

class PreprocessingTesting(db.Model):
    __tablename__ = 'preprocessing_testing'

    id = Column(Integer, primary_key=True, autoincrement=True, unique=True)
    tweet = Column(String(255))
    result = Column(String(255))
    slug = Column(String(255), unique=True)

class ClassificationTesting(db.Model):
    __tablename__ = 'classification_testing'

    id = Column(Integer, primary_key=True, autoincrement=True, unique=True)
    tweet = Column(String(255))
    result_classification = Column(String(25))
    slug = Column(String(255), unique=True)

class UserModel(db.Model, UserMixin):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, autoincrement=True, unique=True)
    name = Column(String(50), unique=True)
    email = Column(String(100), unique=True)
    password = Column(String(255))