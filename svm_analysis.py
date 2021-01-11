
import numpy as np
import pandas as pd
import os
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer
#plots
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score

import warnings
warnings.filterwarnings("ignore")
import time

base_path ='aclImdb'
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

def confusion_matrix(ypred, ytest):
    from sklearn.metrics import classification_report, confusion_matrix
    fig, ax=plt.subplots(figsize=(12,12))
    print(sns.heatmap(confusion_matrix(ytest, ypred, labels=np.unique(ytest)),annot=True, 
               fmt="d",xticklabels=np.unique(ytest),yticklabels=np.unique(ytest), ax=ax))
    print(confusion_matrix(ytest, ypred))
    print(classification_report(ytest, ypred))
    print(accuracy_score(ytest, ypred))
    accuracy=accuracy_score(ytest, ypred)
    return accuracy

#linear SVC
vectorizer = CountVectorizer(analyzer='word', min_df=2,max_df= 0.9, ngram_range=(1, 3))  
X_train_vec = vectorizer.fit_transform(reviews_train_clean) 
X_test_vec = vectorizer.transform(reviews_test_clean) 

tfidftransformer = TfidfTransformer(sublinear_tf=True, use_idf =True, norm='l2')  
X_train_tfidf = tfidftransformer.fit_transform(X_train_vec)
X_test_tfidf = tfidftransformer.transform(X_test_vec)

Xtrain, Xtest, ytrain, ytest = train_test_split(X_train_tfidf, target, test_size=0.16, random_state=42) 


Cs = [0.01, 0.1, 0.5, 1, 1.5,2, 2.5, 3 ]
svc = GridSearchCV(LinearSVC(), param_grid=dict(C=Cs), cv=10)
start = time.time()
svc.fit(Xtrain, ytrain)    
print("Best cross-validation score: {:.2f}".format(svc.best_score_))
print("Best parameters: ", svc.best_params_)
print("Best estimator: ", svc.best_estimator_)
svc = svc.best_estimator_
ypredval_svc = svc.predict(Xtest)
y_pred_test_svc = svc.predict(X_test_tfidf)
end =  time.time()
print('Time to train and predict in Linear SVC Model: {}'.format(end-start))
confusion_matrix(ypredval_svc, ytest)

#Logistic Regression
vectorizer = CountVectorizer(analyzer='word', min_df=2,max_df= 0.9, ngram_range=(1, 3))  
X_train_vec = vectorizer.fit_transform(reviews_train_clean) 
X_test_vec = vectorizer.transform(reviews_test_clean) 

tfidfconverter = TfidfTransformer(sublinear_tf=True, use_idf =True, norm='l2')  
X_train_tfidf = tfidfconverter.fit_transform(X_train_vec)
X_test_tfidf = tfidfconverter.transform(X_test_vec)


Xtrain, Xtest, ytrain, ytest = train_test_split(X_train_tfidf,target, test_size=0.16, random_state=42) 

param_grid = {'C': [1, 10, 100, 1000 ]}
grid = GridSearchCV(LogisticRegression(solver='lbfgs',multi_class='multinomial',random_state=0), param_grid, cv=10)
start = time.time()
grid.fit(Xtrain, ytrain)
print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print("Best parameters: ", grid.best_params_)
print("Best estimator: ", grid.best_estimator_)
lr = grid.best_estimator_
ypred_val_lr = lr.predict(Xtest)
y_pred_test_lr=lr.predict(X_test_tfidf)
end =  time.time()
print('Time to train and predict in Logistic Regression Model: {}'.format(end-start))
confusion_matrix(ypred_val_lr, ytest)






