
import numpy as np
import pandas as pd
import os
import re

#classifiers
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
#models
from sklearn.ensemble import RandomForestClassifier
#from sklearn.linear_model import LogisticRegression
#from sklearn.svm import LinearSVC

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords

#plots
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

base_path = 'aclImdb'
labels = {'pos':1,'neg':0}
df_train=pd.DataFrame(columns=['data_train','label'])
df_test = pd.DataFrame(columns=['data_test','label'])
df_target_train = pd.DataFrame(columns=['label'])

for k in ('train','test'):
	for i in ('neg','pos'):
		if k==('train'):
			path = os.path.join(base_path,k,i)

			for filename in os.listdir(path):
				with open(os.path.join(path,filename),'r',encoding = 'utf-8') as infile:
					text = infile.read()
					df_train = df_train.append({'data_train':text,'label':labels[i]},ignore_index=True)
					df_target_train = df_target_train.append({'label':labels[i]},ignore_index=True)
		elif k==('test'):
			path = os.path.join(base_path,k,i)

			for filename in os.listdir(path):
				with open(os.path.join(path,filename),'r',encoding = 'utf-8') as infile:
					text1 = infile.read()
					df_test = df_test.append({'data_test':text1,'label':labels[i]},ignore_index=True)

dataset_train = df_train['data_train']
dataset_train_target = df_train['label']
dataset_test = df_test['data_test']
dataset_test_target = df_test['label']

#import re

REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
NO_SPACE = ""
SPACE = " "
target = [1 if i < 12500 else 0 for i in range(25000)]
def preprocess_reviews(reviews):
    
    reviews = [REPLACE_NO_SPACE.sub(NO_SPACE, line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(SPACE, line) for line in reviews]
    
    return reviews

reviews_train_clean = preprocess_reviews(dataset_train)
reviews_test_clean = preprocess_reviews(dataset_test)

def confusion_matrix(ypred, y_test):
    from sklearn.metrics import classification_report, confusion_matrix
    #fig, ax=plt.subplots(figsize=(12,12))
    #print(sns.heatmap(confusion_matrix(y_test, ypred, labels=np.unique(y_test)),annot=True, 
              #fmt="d",xticklabels=np.unique(y_test),yticklabels=np.unique(y_test), ax=ax))
    print(confusion_matrix(y_test, ypred))
    print(classification_report(y_test, ypred))
    print(accuracy_score(y_test, ypred))
    accuracy=accuracy_score(y_test, ypred)
    return accuracy

#random forest classifier
#import time
#count vectorizer
stop_words=stopwords.words('english')
vect = CountVectorizer(binary=True, ngram_range=(1,3), stop_words=stop_words)
vect.fit(reviews_train_clean)
X_d = vect.transform(reviews_train_clean)
X_test_d= vect.transform(reviews_test_clean)

X_train, X_val, y_train, y_val = train_test_split(
   X_d,target, train_size = 0.75)

alpha=[0.01, 0.05, 0.25, 0.5, 1]
parameter = {'random_state' : [None]}
dt_clf= GridSearchCV(RandomForestClassifier(), param_grid=parameter, cv=5)
dt_clf.fit(X_train, y_train)
print("cross-val score: {:.2f}".format(dt_clf.best_score_))
print("best param:", dt_clf.best_params_)
print("best estimator:", dt_clf.best_estimator_)
dt_clf=dt_clf.best_estimator_
#start=time.time()
dt_clf= RandomForestClassifier( random_state= 0)
dt_clf.fit(X_train,y_train)
ypred_val_dt=dt_clf.predict(X_val)
y_pred_test_dt=dt_clf.predict(X_d)
#end= time.time()
#print('time taken to train and predict in dt clf: {}'.format(end-start))
confusion_matrix(ypred_val_dt, y_val)
    
final_model=RandomForestClassifier(random_state= 0)
final_model.fit(X_d, target)
print ("Final Accuracy b: %s" 
       % accuracy_score(target, final_model.predict(X_d)))
