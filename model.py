import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("fake_news.csv")


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z ]', '', text)  
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df['text'] = df['text'].apply(clean_text)


# Your labels are ALREADY numeric (0/1)
X = df['text']
y = df['label']

# TF-IDF 
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000,
    min_df=2
)

X_tfidf = vectorizer.fit_transform(X)

# MODEL 
model = LogisticRegression(max_iter=1000)
model.fit(X_tfidf, y)

# PREDICTION
def predict_news(news_text):
    news_text = clean_text(news_text)
    vector = vectorizer.transform([news_text])
    prediction = model.predict(vector)[0]
    return "REAL NEWS" if prediction == 1 else "FAKE NEWS"
