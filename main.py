import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.preprocessing import normalize
from functions_cleaning import *
import nltk


def apply_grades(row):
    if row in [5, 4]:
        return "Positive"
    elif row in [1, 2]:
        return "Negative"
    return "Neutral"

df = pd.read_csv('alexa_reviews.csv', sep=";", encoding='cp1252')
df['rating'] = df['rating'].apply(lambda x:apply_grades(x))
df = df[:1000]
print(df)
print(df['rating'])
print(df["rating"].value_counts())
print(df["verified_reviews"].value_counts())

plot(df)

string = ""
for i in tqdm(range(len(df['verified_reviews']))):
    string += df['verified_reviews'].iloc[i] + " "
print(len(string))

tekst = text_tokenizer(string)
bow = bag_of_words(tekst)
wordcloud(bow)
print('\n')
print(tekst)





vectorizer = CountVectorizer(tokenizer=text_tokenizer)
X_transform = vectorizer.fit_transform(df['verified_reviews'])
print(vectorizer.get_feature_names_out())
print(X_transform.toarray())
#print(X_transform)
print(X_transform.shape)


plot_most_important(top_tokens(X_transform.toarray().sum(axis=0), vectorizer.get_feature_names_out(), 10),
                    bow, "Tokeny występujące najszęściej wg ilości")

print(prettytable_most_important(top_tokens(X_transform.toarray().sum(axis=0), vectorizer.get_feature_names_out(), 10),
                            bow, "Tokeny występujące najszęściej wg ilości"))

vectorizer_tfidf = TfidfVectorizer(tokenizer=text_tokenizer)
transform_tfidf = vectorizer_tfidf.fit_transform(df['verified_reviews'])
columns = vectorizer_tfidf.get_feature_names_out()
weights = transform_tfidf.toarray().mean(axis=0)

key_plot(columns, weights)

print(prettytable_key(columns, weights))

x_train, x_test, y_train, y_test = train_test_split(X_transform, df['verified_reviews'], test_size=0.3, random_state=42)


#classifiers = [LinearSVC()]
#for clf in classifiers:
 #   fig, ax = plt.subplots(1,1)
 #   clf.fit(x_train, y_train)
 #   y_pred = clf.predict(x_test)
  #  cm = confusion_matrix(y_test, y_pred)
 #   cm = normalize(cm, axis=0, norm='l1')
 #   disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
  #  ax.title.set_text(f"{clf}")
  #  disp.plot(ax=ax)
  #  plt.show()
  #  cr = classification_report(y_test, y_pred, target_names=clf.classes_)
  #  print(cr)