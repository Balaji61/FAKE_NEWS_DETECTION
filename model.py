import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# =========================
# 1. Load dataset
# =========================
df = pd.read_csv("fake_news.csv")

# Check columns
print(df.columns)

# =========================
# 2. Split dataset
# =========================
X = df["text"]
y = df["label"]

# =========================
# 3. Create TF-IDF Vectorizer
# =========================
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)

X_vec = vectorizer.fit_transform(X)

# =========================
# 4. Train Logistic Regression Model
# =========================
model = LogisticRegression(max_iter=200)
model.fit(X_vec, y)

print("Training complete!")

# =========================
# 5. Save model + vectorizer
# =========================
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("model.pkl and vectorizer.pkl saved successfully!")
