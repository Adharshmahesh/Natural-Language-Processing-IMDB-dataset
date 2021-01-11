
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from matplotlib import pyplot
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import sys
print(sys.executable)
print(sys.path)

import os
import re

#plots
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
#plots
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
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
#Word Cloud 


plt.style.use('fivethirtyeight')
from wordcloud import WordCloud, STOPWORDS
stop_words = ['movie','film','scene','make','really','story','made','might','even','one','character'] + list(STOPWORDS)
negative_words = reviews_train_clean[12500:]
negative_string = []
for t in negative_words:
    negative_string.append(t)
negative_string = pd.Series(negative_string).str.cat(sep=' ')
wordcloud = WordCloud(width=1500, height=750,max_font_size=190, stopwords=stop_words).generate(negative_string)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()



positive_words = reviews_train_clean[:12500]
positive_string = []
for t in positive_words:
    positive_string.append(t)
positive_string = pd.Series(positive_string).str.cat(sep=' ')
wordcloud = WordCloud(width=1500, height=750,max_font_size=190,stopwords=stop_words,colormap='magma').generate(positive_string) 
plt.figure(figsize=(12,10)) 
plt.imshow(wordcloud, interpolation="bilinear") 
plt.axis("off") 
plt.show()