import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# ── Load ──────────────────────────────────────────────
folder = r"E:\mental_health"
dfs = []
for f in os.listdir(folder):
    if f.endswith('.csv'):
        temp = pd.read_csv(os.path.join(folder, f))
        temp['Label'] = temp['Label'].str.lower().str.strip()
        dfs.append(temp)

df = pd.concat(dfs, ignore_index=True)
df['selftext'] = df['selftext'].astype(str)

# ── Fix NaN ───────────────────────────────────────────
df = df.dropna(subset=['Label', 'selftext'])
print("Total rows after cleanup:", len(df))

# ── EDA ───────────────────────────────────────────────
df['Label'].value_counts().plot(
    kind='bar',
    color=['#7c6af7','#4fc3f7','#4caf82','#f5a623']
)
plt.title('Label Distribution')
plt.tight_layout()
plt.savefig('label_dist.png')
plt.show()

text = ' '.join(df[df['Label']=='trauma and stress']['selftext'])
wc = WordCloud(width=800, height=400, background_color='white').generate(text)
plt.figure(figsize=(10,5))
plt.imshow(wc)
plt.axis('off')
plt.title('Trauma and Stress - Word Cloud')
plt.savefig('wordcloud.png')
plt.show()

# ── Model ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df['selftext'], df['Label'], test_size=0.2, random_state=42
)

tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec  = tfidf.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# ── Confusion Matrix ──────────────────────────────────
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=model.classes_,
            yticklabels=model.classes_,
            cmap='Purples')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# ── Test ──────────────────────────────────────────────
def predict(text):
    vec = tfidf.transform([text])
    return model.predict(vec)[0]

sample = "I've been feeling really anxious and stressed about my relationships"
print("Prediction:", predict(sample))

import numpy as np

feature_names = tfidf.get_feature_names_out()
importances = model.feature_importances_
top_idx = np.argsort(importances)[::-1][:20]

plt.figure(figsize=(10,6))
plt.barh(
    [feature_names[i] for i in top_idx][::-1],
    importances[top_idx][::-1],
    color='#7c6af7'
)
plt.title('Top 20 Most Important Words')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()