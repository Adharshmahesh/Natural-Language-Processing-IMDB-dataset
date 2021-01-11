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
from matplotlib import pyplot
import matplotlib.pyplot as plt
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

from sklearn.metrics import f1_score
from sklearn.metrics import auc

import warnings
warnings.filterwarnings("ignore")
import time

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
    print(confusion_matrix(y_test, ypred))
    print(classification_report(y_test, ypred))
    print(accuracy_score(y_test, ypred))
    accuracy=accuracy_score(y_test, ypred)
    return accuracy
'''
#linear SVC
#Count Vectorization with tfidf transformer:
vectorizer = CountVectorizer(analyzer='word', min_df=2,max_df= 0.9, ngram_range=(1, 3))  
X_train_vect = vectorizer.fit_transform(reviews_train_clean) 
X_test_vect = vectorizer.transform(reviews_test_clean) 

tfidfconverter = TfidfTransformer(sublinear_tf=True, use_idf =True, norm='l2')  
X_train_tfidf = tfidfconverter.fit_transform(X_train_vect)
X_test_tfidf = tfidfconverter.transform(X_test_vect)


#Spliting the dataset for training and validation:
X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, target, test_size=0.16, random_state=42) 

#Linear SVC Model:
Cs = [0.01, 0.1, 0.5, 1, 1.5,2, 2.5, 3 ]
svc = GridSearchCV(LinearSVC(), param_grid=dict(C=Cs), cv=10)
start = time.time()
svc.fit(X_train, y_train)    
print("Best cross-validation score: {:.2f}".format(svc.best_score_))
print("Best parameters: ", svc.best_params_)
print("Best estimator: ", svc.best_estimator_)
svc= svc.best_estimator_
ypred_val_svc = svc.predict(X_test)
y_pred_test_svc = svc.predict(X_test_tfidf)
end =  time.time()
print('Time to train and predict in Linear SVC Model: {}'.format(end-start))
#confusion_matrix(ypred_val_svc, y_test)
'''
#logistic regretion
#Count Vectorization with tfidf transformer:
vectorizer = CountVectorizer(analyzer='word', min_df=2,max_df= 0.9, ngram_range=(1, 3))  
X_train_vect = vectorizer.fit_transform(reviews_train_clean) 
X_test_vect = vectorizer.transform(reviews_test_clean) 

tfidfconverter = TfidfTransformer(sublinear_tf=True, use_idf =True, norm='l2')  
X_train_tfidf = tfidfconverter.fit_transform(X_train_vect)
X_test_tfidf = tfidfconverter.transform(X_test_vect)


#Spliting the dataset for training and validation:
X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf,target, test_size=0.16, random_state=42) 

#Logistic Regression
param_grid = {'C': [1, 10, 100, 1000 ]}
grid = GridSearchCV(LogisticRegression(solver='lbfgs',multi_class='multinomial',random_state=0), param_grid, cv=10)
start = time.time()
grid.fit(X_train, y_train)
print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print("Best parameters: ", grid.best_params_)
print("Best estimator: ", grid.best_estimator_)
lr = grid.best_estimator_
ypred_val_lr = lr.predict(X_test)
y_pred_test_lr=lr.predict(X_test_tfidf)
end =  time.time()
print('Time to train and predict in Logistic Regression Model: {}'.format(end-start))
#confusion_matrix(ypred_val_lr, y_test)
'''

from sklearn.ensemble import VotingClassifier
estimators=[('LinearSVC', svc), ('Logistic_Regression', lr)]
ensemble = VotingClassifier(estimators, voting='hard')
start = time.time()
#Fitting the model on traininf data
ensemble.fit(X_train, y_train)
ypred_val_en   = ensemble.predict(X_test)
y_pred_test_en = ensemble.predict(X_test_tfidf)
end =  time.time()
#test our model on the test data            
ensemble.score(X_test, y_test)
print('Time to train and predict in Ensemble model: {}'.format(end-start))
confusion_matrix(ypred_val_en, y_test)

'''
# predict probabilities
probs = lr.predict_proba(X_test)
probs = probs[:,1]
auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' %auc)
fpr, tpr, thresholds = roc_curve(y_test, probs)
pyplot.plot([0, 1], [0, 1], linestyle='--')
pyplot.plot(fpr, tpr, marker='.')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC curve for LogisticRegretion')                     
pyplot.show()



