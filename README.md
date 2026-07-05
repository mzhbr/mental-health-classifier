# 🧠 Reddit Mental Health Post Classifier

A multi-class text classification pipeline that identifies the mental health 
topic of Reddit posts using NLP and machine learning.

## What It Does

Given a Reddit post, the model predicts which mental health category it 
belongs to:
- 💊 Drug and Alcohol
- 👶 Early Life
- 🧬 Personality
- 😰 Trauma and Stress

## Why It Matters

Mental health discussions on social media contain valuable signals for:
- Early detection of at-risk individuals
- Understanding language patterns across different mental health domains
- Building NLP tools that support clinical research

This project sits at the intersection of **NLP** and 
**computational neuroscience** — using language as a window into 
mental states.

## Results

| Category | Precision | Recall | F1 |
|----------|-----------|--------|----|
| Drug and Alcohol | 0.79 | 0.73 | 0.76 |
| Early Life | 0.61 | 0.81 | 0.70 |
| Personality | 0.66 | 0.60 | 0.63 |
| Trauma and Stress | 0.54 | 0.44 | 0.49 |
| **Overall Accuracy** | | | **0.64** |

## Visualizations

### Label Distribution
![Label Distribution](label_dist.png)

### Word Cloud — Trauma & Stress
![Word Cloud](wordcloud.png)

### Confusion Matrix
![Confusion Matrix](confusion_matrix.png)

### Top 20 Most Important Words
![Feature Importance](feature_importance.png)

## Key Finding

The model learned clinically meaningful signals:
- **weed, drugs, alcohol, cocaine** → Drug & Alcohol category
- **childhood, school, kid, abused** → Early Life category  
- **anxiety, sober, feel** → shared across categories

Overlap between Trauma & Stress and other categories explains 
the lower F1 score — reflecting real-world complexity of mental 
health language.

## Pipeline


Raw Reddit Posts (CSV)
↓
Data Cleaning & Normalization
↓
TF-IDF Vectorization (5000 features)
↓
Random Forest Classifier (100 trees)
↓
Evaluation + Feature Importance

## Tech Stack

| Tool | Purpose |
|------|---------|
| pandas | Data loading and cleaning |
| scikit-learn | TF-IDF + Random Forest + evaluation |
| matplotlib / seaborn | Visualization |
| wordcloud | Word Cloud generation |

## How to Run

```bash
git clone https://github.com/mzhbr/mental-health-classifier
cd mental-health-classifier
pip install -r requirements.txt
python main.py
```

## Dataset

Reddit Mental Health Dataset — available on Kaggle.
Contains posts from mental health subreddits, labeled by category.

---
