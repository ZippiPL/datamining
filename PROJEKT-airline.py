# Zespół: Chela Laura, Duda Denis

from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.preprocessing import normalize
from functions_cleaning import *

df = pd.read_csv(r"Tweets-airline.csv")
df = df[:14000]
print(df)

print(df['text'][1])

show_plot(df)

print(df["airline_sentiment"].value_counts())

string = ""
for i in tqdm(range(len(df['text']))):
    string += df['text'].iloc[i] + " "
print(len(string))

tekst = text_tokenizer(string)
bow = bag_of_words(tekst)
wordcloud(bow)

vectorizer = CountVectorizer(tokenizer=text_tokenizer)
X_transform = vectorizer.fit_transform(df['text'])
# print(vectorizer.get_feature_names_out())
print(X_transform.toarray())
# print(X_transform)
print(X_transform.shape)

show_plot_most_important(top_10_najczesciej_tokeny(X_transform.toarray().sum(axis=0), vectorizer.get_feature_names_out(), 10),
                         bow, "Tokeny występujące najszęściej wg ilości")

print(prettytable_most_important(top_10_najczesciej_tokeny(X_transform.toarray().sum(axis=0), vectorizer.get_feature_names_out(), 10),
                                 bow, "Tokeny występujące najszęściej wg ilości"))

vectorizer_tfidf = TfidfVectorizer(tokenizer=text_tokenizer)
transform_tfidf = vectorizer_tfidf.fit_transform(df['text'])
columns = vectorizer_tfidf.get_feature_names_out()
weights = transform_tfidf.toarray().mean(axis=0)

show_key_plot(columns, weights)

print('prettytable_key')
print(prettytable_tfidf_key(columns, weights))

x_train, x_test, y_train, y_test = train_test_split(X_transform, df['airline_sentiment'], test_size=0.3, random_state=42)

classifiers = [LinearSVC(), AdaBoostClassifier(), BaggingClassifier(), DecisionTreeClassifier(), RandomForestClassifier()]
for classif in classifiers:
    fig, ax = plt.subplots(1, 1)
    classif.fit(x_train, y_train)
    y_pred = classif.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    cm = normalize(cm, axis=0, norm='l1')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classif.classes_)
    ax.title.set_text(f"{classif}")
    disp.plot(ax=ax)
    plt.show()
    cr = classification_report(y_test, y_pred, target_names=classif.classes_)
    print(cr)

# dokładność - Accuracy=(TP+TN)/Total
# Wysoka wartość wskaźnika recall oznacza, że bardzo niewiele wyników pozytywnych jest błędnie klasyfikowanych jako wyniki negatywne.
# Wysoka precyzja oznacza, że bardzo niewiele wyników negatywnych jest błędnie klasyfikowanych jako pozytywne.
# Istnieje tu pewien kompromis. Jeśli model jest ukierunkowany na pozytywy, uzyskamy wysoki współczynnik przywołania, ale niską precyzję.
# Jeśli model faworyzuje negatywy, uzyskamy niski współczynnik przywoływania i wysoką precyzję.
# Dokładność to ogólna miara poprawności przewidywań, niezależnie od klasy (pozytywna lub negatywna).

# INTERPRETACJA najlepszego classifiera
# Najlepszym z klasifikatorów okazał się RandomForestClassifier(), którego accuracy(dokładność) wyniosła 0.76,
# czyli zaklasyfikował on poprawnie (zgodnie z klasą: negativ, neutral, positiv) 76% przypadków poddanych predykcji.

#              precision    recall  f1-score   support
#    negative       0.80      0.92      0.85      2676
#     neutral       0.60      0.42      0.49       859
#    positive       0.71      0.57      0.64       665

# Precision (precyzja) w przypadku klasy "negative" wyniosła 0.80, czyli 80% przypadków zostało poprwanie zaklasyfikowane jako "negative".
# Recall w przypadku klasy "negative" wyniosła 0.92, co oznacza, że 92% spośród wszystkich poprawnych klasyfikacji jako "negative"
#   zostało prawidłowo przewidzianych (bardzo niewiele wyników pozytywnych jest błędnie klasyfikowanych jako inne klasy).
# F1-score F-score jest metryką wydajności modelu uczenia maszynowego, która daje równą wagę zarówno Precision, jak i Recall
#   do pomiaru jego wydajności pod względem dokładności, co czyni ją alternatywą dla metryki Accuracy
#   i w przypadku klasy "negative" wynosi 0.85.
# Support - liczba przypadków użytych do testowania
