import numpy as np
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import re

nltk.download("stopwords")
nltk.download('punkt')

path = "data"
stop_words = set(stopwords.words("english"))

def preprocess(sentence):
    stemer = PorterStemmer()
    sentence = sentence.lower()
    sentence = re.sub(r"http\S+", "", sentence)
    sentence = re.sub(r"<.*?>", "", sentence)
    sentence = re.sub(r"\s+", " ", sentence).strip()
    sentence = re.sub(r'[^\w\s]', '', sentence)

    sentence = word_tokenize(sentence)
    tokens=[]
    for word in sentence:
        if (word not in stop_words) or (word == "not"):
            word = stemer.stem(word)
            tokens.append(word)
    return " ".join(tokens)

if __name__ == "__main__":
    train_data = pd.read_csv(path + "/train.csv", encoding="latin1")
    test_data = pd.read_csv(path + "/test.csv", encoding="latin1")

    train_data = train_data[(train_data["sentiment"] == "positive") | (train_data["sentiment"] == "negative")]
    test_data = test_data[(test_data["sentiment"] == "positive") | (test_data["sentiment"] == "negative")]

    sentiment_map = {"positive":1,
                    "negative":0}
    x_train = np.array(train_data["text"])
    y_train = np.array([sentiment_map[i] for i in train_data["sentiment"]])
    x_test = np.array(test_data["text"])
    y_test = np.array([sentiment_map[i] for i in test_data["sentiment"]])


    tfidf = TfidfVectorizer(ngram_range=(1,2),min_df=2,max_df=0.95)

    processed = np.array([preprocess(i) for i in x_train])
    x_train_tfidf = tfidf.fit_transform(processed)
    processed_test = np.array([preprocess(i) for i in x_test])
    x_test_tfidf = tfidf.transform(processed_test)

    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(max_iter=1000)

    model.fit(x_train_tfidf,y_train)
    y_predict = model.predict(x_test_tfidf)

    accuracy = np.sum(y_predict == y_test) / len(y_test)
    print(accuracy)