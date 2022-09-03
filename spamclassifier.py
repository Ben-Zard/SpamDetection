# -*- coding: utf-8 -*-

files.upload()

import pandas as pd
df = pd.read_csv('SMSSpamCollection.txt', sep='\t', names=["label", "message"])

print(df.head())

print(df.tail())

import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0, len(df)):
    sms_message = re.sub('[^a-zA-Z]', ' ', df['message'][i])
    sms_message = sms_message.lower()
    sms_words = sms_message.split()

    stemmed_sms_words = []
    for word in sms_words:
      if word not in stopwords.words('english'):
        stemmed_sms_words.append(word)
    stemmed_sms_message = ' '.join(stemmed_sms_words) # diffrent stem or lemmers
    corpus.append(stemmed_sms_message)
print(corpus)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()

print(X.shape)

y=pd.get_dummies(df['label']) # ham or spam into 1 or 0 
y=y.iloc[:,1].values
print(y)

# Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)
y_pred=spam_detect_model.predict(X_test)

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
score = recall_score(y_test, y_pred)
print(score)



from sklearn.tree import DecisionTreeClassifier
decision_tree_clf = DecisionTreeClassifier(max_depth=5)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(decision_tree_clf, X_train, y_train, cv=5, scoring='precision')
spam_detect_model = MultinomialNB().fit(X_train, y_train)
y_pred=spam_detect_model.predict(X_test)
print(scores.mean())
print(scores)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(decision_tree_clf, X_train, y_train, cv=5, scoring='accuracy')
print(scores.mean())
print(scores)

from sklearn.ensemble import RandomForestClassifier
random_forest_clf = RandomForestClassifier(max_depth=2, random_state=0)
scores = cross_val_score(decision_tree_clf, X_train, y_train, cv=5, scoring='precision')
print(scores.mean())
print(scores)