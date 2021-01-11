
import numpy as np
import pandas as pd
import os
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords

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

import re

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

#baseline
vectorizer = CountVectorizer(binary=True)
vectorizer.fit(reviews_train_clean)
X_b = vectorizer.transform(reviews_train_clean)
X_test_b= vectorizer.transform(reviews_test_clean)

Xtrain, Xval, ytrain, yval = train_test_split(
   X_b, target, train_size = 0.75)

for alpha in [0.01, 0.05, 0.25, 0.5, 1]:
    
    lr = LogisticRegression(C=alpha)
    lr.fit(Xtrain, ytrain)
    print ("Accuracy for C=%s: %s" 
           % (alpha, accuracy_score(yval, lr.predict(Xval))))
    
#     Accuracy for C=0.01: 0.87472
#     Accuracy for C=0.05: 0.88368
#     Accuracy for C=0.25: 0.88016
#     Accuracy for C=0.5: 0.87808
#     Accuracy for C=1: 0.87648

final_model = LogisticRegression(C=0.05)
final_model.fit(X_b, target)
print ("Final Accuracy b: %s" 
       % accuracy_score(target, final_model.predict(X_test_b)))
# Final Accuracy: 0.88168


#stopwords
stop_words = stopwords.words('english')
def remove_stop_words(corpus):
    removed_stop_words = []
    for review in corpus:
        removed_stop_words.append(
            ' '.join([word for word in review.split() 
                      if word not in stop_words])
        )
    return removed_stop_words

no_stop_words_train = remove_stop_words(reviews_train_clean)
no_stop_words_test = remove_stop_words(reviews_test_clean)

cv_s = CountVectorizer(binary=True)
cv_s.fit(no_stop_words_train)
X_s = cv_s.transform(no_stop_words_train)
X_test_s = cv_s.transform(no_stop_words_test)

X_train, X_val, y_train, y_val = train_test_split(
    X_s, target, train_size = 0.75
)

for alpha in [0.01, 0.05, 0.25, 0.5, 1]:
    
    lr = LogisticRegression(C=alpha)
    lr.fit(X_train, y_train)
    print ("Accuracy for C=%s: %s" 
           % (alpha, accuracy_score(y_val, lr.predict(X_val))))

final_model = LogisticRegression(C=0.05)
final_model.fit(X_s, target)
print ("Final Accuracy s: %s" 
       % accuracy_score(target, final_model.predict(X_test_s)))

#stemming
def get_stemmed_text(corpus):
    from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer()

    return [' '.join([stemmer.stem(word) for word in review.split()]) for review in corpus]

stemmed_reviews_train = get_stemmed_text(reviews_train_clean)
stemmed_reviews_test = get_stemmed_text(reviews_test_clean)

cv_l = CountVectorizer(binary=True)
cv_l.fit(stemmed_reviews_train)
X_l = cv_l.transform(stemmed_reviews_train)
X_test_l = cv_l.transform(stemmed_reviews_test)

X_train, X_val, y_train, y_val = train_test_split(
    X_l, target, train_size = 0.75
)

for alpha in [0.01, 0.05, 0.25, 0.5, 1]:
    
    lr = LogisticRegression(C=alpha)
    lr.fit(X_train, y_train)
    print ("Accuracy for C=%s: %s" 
           % (alpha, accuracy_score(y_val, lr.predict(X_val))))
    
final_stemmed = LogisticRegression(C=0.05)
final_stemmed.fit(X_l, target)
print ("Final Accuracy l: %s" 
       % accuracy_score(target, final_stemmed.predict(X_test_l)))
#lematizer

def get_lemmatized_text(corpus):
    
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in corpus]

lemmatized_reviews_train = get_lemmatized_text(reviews_train_clean)
lemmatized_reviews_test = get_lemmatized_text(reviews_test_clean)

cv_lm = CountVectorizer(binary=True)
cv_lm.fit(lemmatized_reviews_train)
X_lm = cv_lm.transform(lemmatized_reviews_train)
X_test_lm = cv_lm.transform(lemmatized_reviews_test)

X_train, X_val, y_train, y_val = train_test_split(
    X_lm, target, train_size = 0.75
)

for alpha in [0.01, 0.05, 0.25, 0.5, 1]:
    
    lr = LogisticRegression(C=alpha)
    lr.fit(X_train, y_train)
    print ("Accuracy for C=%s: %s" 
           % (alpha, accuracy_score(y_val, lr.predict(X_val))))
    
final_lemmatized = LogisticRegression(C=0.25)
final_lemmatized.fit(X_lm, target)
print ("Final Accuracy lm: %s" 
       % accuracy_score(target, final_lemmatized.predict(X_test_lm)))

#ngram
ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
ngram_vectorizer.fit(reviews_train_clean)
X_n = ngram_vectorizer.transform(reviews_train_clean)
X_test_n = ngram_vectorizer.transform(reviews_test_clean)

X_train, X_val, y_train, y_val = train_test_split(
    X_n, target, train_size = 0.75
)

for alpha in [0.01, 0.05, 0.25, 0.5, 1]:
    
    lr = LogisticRegression(C=alpha)
    lr.fit(X_train, y_train)
    print ("Accuracy for C=%s: %s" 
           % (alpha, accuracy_score(y_val, lr.predict(X_val))))
    
# Accuracy for C=0.01: 0.88416
# Accuracy for C=0.05: 0.892
# Accuracy for C=0.25: 0.89424
# Accuracy for C=0.5: 0.89456
# Accuracy for C=1: 0.8944
    
final_ngram = LogisticRegression(C=0.5)
final_ngram.fit(X_n, target)
print ("Final Accuracy n: %s" 
       % accuracy_score(target, final_ngram.predict(X_test_n)))

# Final Accuracy: 0.898
#word count

#from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import accuracy_score
#from sklearn.model_selection import train_test_split

wc_vectorizer = CountVectorizer(binary=False)
wc_vectorizer.fit(reviews_train_clean)
X_wc = wc_vectorizer.transform(reviews_train_clean)
X_test_wc = wc_vectorizer.transform(reviews_test_clean)

X_train, X_val, y_train, y_val = train_test_split(
    X_wc, target, train_size = 0.75, 
)

for alpha in [0.01, 0.05, 0.25, 0.5, 1]:
    
    lr = LogisticRegression(C=alpha)
    lr.fit(X_train, y_train)
    print ("Accuracy for C=%s: %s" 
           % (alpha, accuracy_score(y_val, lr.predict(X_val))))
    
# Accuracy for C=0.01: 0.87456
# Accuracy for C=0.05: 0.88016
# Accuracy for C=0.25: 0.87936
# Accuracy for C=0.5: 0.87936
# Accuracy for C=1: 0.87696
    
final_wc = LogisticRegression(C=0.05)
final_wc.fit(X_wc, target)
print ("Final Accuracy wc: %s" 
       % accuracy_score(target, final_wc.predict(X_test_wc)))

# Final Accuracy: 0.88184

#tfidf


tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(reviews_train_clean)
X_tf = tfidf_vectorizer.transform(reviews_train_clean)
X_test_tf = tfidf_vectorizer.transform(reviews_test_clean)

X_train, X_val, y_train, y_val = train_test_split(
    X_tf, target, train_size = 0.75
)

for alpha in [0.01, 0.05, 0.25, 0.5, 1]:
    
    lr = LogisticRegression(C=alpha)
    lr.fit(X_train, y_train)
    print ("Accuracy for C=%s: %s" 
           % (alpha, accuracy_score(y_val, lr.predict(X_val))))

# Accuracy for C=0.01: 0.79632
# Accuracy for C=0.05: 0.83168
# Accuracy for C=0.25: 0.86768
# Accuracy for C=0.5: 0.8736
# Accuracy for C=1: 0.88432
    
final_tfidf = LogisticRegression(C=1)
final_tfidf.fit(X_tf, target)
print ("Final Accuracy tfidf: %s" 
       % accuracy_score(target, final_tfidf.predict(X_test_tf)))
#accuracy= 0.882
#svm model
stop_words = ['in', 'of', 'at', 'a', 'the']
ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 1),stop_words=stop_words)
ngram_vectorizer.fit(reviews_train_clean)
X_svm= ngram_vectorizer.transform(reviews_train_clean)
X_test_svm = ngram_vectorizer.transform(reviews_test_clean)

X_train, X_val, y_train, y_val = train_test_split(
    X_svm, target, train_size = 0.75
)

for alpha in [0.01, 0.05, 0.25, 0.5, 1]:
    
    svm = LinearSVC(C=alpha)
    svm.fit(X_train, y_train)
    print ("Accuracy for C=%s: %s" 
           % (alpha, accuracy_score(y_val, svm.predict(X_val))))
    
# Accuracy for C=0.01: 0.89104
# Accuracy for C=0.05: 0.88736
# Accuracy for C=0.25: 0.8856
# Accuracy for C=0.5: 0.88608
# Accuracy for C=1: 0.88592
    
final_svm_ngram = LinearSVC(C=0.01)
final_svm_ngram.fit(X_svm, target)
print ("Final Accuracy: %s" 
       % accuracy_score(target, final_svm_ngram.predict(X_test_svm)))

#final accuracy 0.90064








