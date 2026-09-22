import numpy as np
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
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
        if word not in stop_words:
            word = stemer.stem(word)
            tokens.append(word)
    return tokens


def get_freqs_dict(x_train , y_train,preprocess = preprocess):
    freqs = {}

    for sentence,y in zip(x_train,y_train):
        sentence = preprocess(sentence)
        for word in sentence:
            if word not in freqs.keys():
                word_vec = np.zeros(3)
                word_vec[0] = 1
                word_vec[2 - y] = 1 
                freqs[word] = word_vec
            else:
                freqs[word][2 - y] +=1
    return freqs


def get_setence_vector(sentence,freqs,preprocess = preprocess):
    sentence_vector = np.zeros(3)
    for word in preprocess(sentence):
        sentence_vector += freqs.get(word,0)
    sentence_vector[0] = 1
    return sentence_vector

def sigmoid(z):

    return 1/(1 + np.exp(-z))


def gradientDescent(x, y, theta, alpha, num_iterations):
    m = len(y)
    for i in range(0, num_iterations):
        z = np.dot(x,theta)       
        h = sigmoid(z)
        J = (-1 / m) * ((np.dot(y.T, np.log(h)))  +  (np.dot(((1 - y).T), np.log(1 - h))))
        theta = theta - (alpha / m) * (np.dot(x.T , (h - y)))
    return J, theta

def predict_sentiment(tweet, freqs, theta):
    x = get_setence_vector(tweet,freqs)
    y_pred = sigmoid(np.dot(x , theta))
    
    return y_pred

def test_logistic_regression(test_x, test_y, freqs, theta, predict=predict_sentiment):
    y_hat = []
    
    for tweet in test_x:
        y_pred = predict(tweet,freqs,theta)
        
        if y_pred > 0.5:
            y_hat.append(1.0)
        else:
            y_hat.append(0.0)

    accuracy = sum(np.array(y_hat) == test_y.flatten()) / len(y_hat)    
    return accuracy


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

    freqs = get_freqs_dict(x_train,y_train)

    train_vectors = np.zeros((len(y_train), 3))
    for i,sentence in enumerate(x_train):
        train_vectors[i] = get_setence_vector(sentence,freqs)

    y_train= y_train.reshape(-1,1)

    j , theta = gradientDescent(train_vectors,y_train,np.zeros((3,1)),0.0000001,2000)

    accuracy = test_logistic_regression(x_test,y_test,freqs,theta)
    print(accuracy)

