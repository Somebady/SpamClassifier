import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np


mails = pd.read_csv(
    'C:/Users/akshason/OneDrive - AMDOCS/Backup Folders/Desktop/spam.csv', encoding='ISO-8859-1')
# print(mails.columns)
mails = mails[['v1', 'v2']]


ps = PorterStemmer()
wnet = WordNetLemmatizer()

corpus = []
for i in range(0, len(mails)):
    mail = re.sub('[^a-zA-Z]', ' ', mails['v2'][i])
    mail = mail.lower()
    mail = mail.split()

    mail = [wnet.lemmatize(word) for word in mail if word not in set(
        stopwords.words('english'))]
    mail = ' '.join(mail)
    corpus.append(mail)

print(corpus)


tv = TfidfVectorizer(max_features=4500)
X = tv.fit_transform(corpus)
print(X)
print(mails['v1'])
Y = pd.get_dummies(mails['v1'], drop_first=True)
print(Y)

print(X.shape, Y.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0)

spam_model = MultinomialNB()
spam_model.fit(X_train, y_train)
y_pred = spam_model.predict(X_test)
print(spam_model.score(X_test, y_test))
confusion = confusion_matrix(y_test, y_pred)
print(confusion)

accuracy = accuracy_score(y_pred, y_test)
print(accuracy)

# Using Stemming we are getting below accuradcy scores
# 0.965311004784689
# [[1433    1]
#  [57  181]]


# Accuracy we are getting useing Lamatization
# 0.9688995215311005
# [[1434    0]
#  [52  186]]
print(X_test.shape)

# # open a file, where we want to store data
file = open('spam_classifier.pkl', 'wb')
# dump information to that file
pickle.dump(spam_model, file)

# Dumping Vectorize Pickle file
pickle.dump(tv, open("tv_vector.pkl", "wb"))


# Testing

mail = '''
            Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's
'''

emp_mail = []
mail = re.sub('[^a-zA-Z]', ' ', mail)
mail = mail.lower()
mail = mail.split()
mail_words = [wnet.lemmatize(word)
              for word in mail if word not in set(stopwords.words('english'))]
mail = ' '.join(mail_words)

emp_mail.append(mail)

mail_vector = tv.transform(emp_mail)
print(X_test.shape, mail_vector.shape)
prediction = spam_model.predict(mail_vector)

if prediction == 0:
    print("It's a geniune Mail")
else:
    print("Alert!!--It's a Spam Message, Please be secure while opening it.")
