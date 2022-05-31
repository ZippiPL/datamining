from lab_2 import (
    text_tokenizer,
    top_tokens,
    top_documents,
    )
import pandas as pd
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    CountVectorizer
    )


df = pd.read_csv(r"D:\pythonuczelniasem2\datamining\files\True.csv")

vectorizer = CountVectorizer(tokenizer=text_tokenizer)
X_transform = vectorizer.fit_transform(df['title'])

vectorizer_tfidf = TfidfVectorizer(tokenizer=text_tokenizer)
transform_tfidf = vectorizer_tfidf.fit_transform(df['title'])
print(X_transform.toarray())
print(vectorizer.get_feature_names_out())
# Top 10 occurring tokens
print("Top 10 occurring tokens")
print(top_tokens(X_transform.toarray().sum(axis=0), vectorizer.get_feature_names_out(), 10))

# Top 10 most important tokens
print("Top 10 most important tokens")
print(top_tokens(transform_tfidf.toarray().sum(axis=0), vectorizer_tfidf.get_feature_names_out(), 10))

# Top 10 documents
print("Top 10 documents")
print(top_documents(X_transform.toarray().sum(axis=1), 10))