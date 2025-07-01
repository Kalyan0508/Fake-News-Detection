# 🚀 **Fake News Detection**  
*Automatically classify news articles as real or fake using NLP and machine learning.*

---

## 📝 **Overview**  
In today’s digital age, misinformation spreads rapidly across social media and news outlets. **Fake News Detection** applies natural language processing and supervised learning to automatically flag deceptive content, helping platforms and readers identify credible journalism.

---

## ✨ **Features**  
- ✅ **Text Cleaning & Preprocessing**  
  - Tokenization  
  - Lemmatization  
  - Stop‑word removal  
- ✅ **Feature Extraction**  
  - TF‑IDF vectors  
  - Word embeddings (GloVe, Word2Vec)  
- ✅ **Model Comparison**  
  - Logistic Regression  
  - Random Forest  
  - Support Vector Machine (SVM)  
  - LSTM (Keras)  
- ✅ **Performance Metrics**  
  - Accuracy  
  - Precision  
  - Recall  
  - F1‑Score  
  - ROC AUC  
- ✅ **Jupyter Notebook** walkthrough of all steps

---

## ⚙️ **Installation**  
1. **Clone the repo**  
   ```bash
   git clone https://github.com/Kalyan0508/fake-news-detection.git  
   cd fake-news-detection
2. **Create a virtual environment**

   python3 -m venv venv  
   source venv/bin/activate      # Windows: venv\Scripts\activate
## 📂 **Dataset**
## Source: Kaggle Fake News Dataset

Description: ~20,000 news articles labeled FAKE or REAL.

Files:

train.csv — training set with text & labels

test.csv — unseen articles for evaluation

Preprocessing Steps:

Lowercasing & punctuation removal

Tokenization & lemmatization

Stop‑word filtering

## 🔬 Modeling

### 🧹 Data Preprocessing
- Text normalization (lowercase, strip HTML, remove URLs)  
- Tokenization with NLTK / spaCy  
- Lemmatization & stop‑word removal  

### 🔧 Feature Engineering
- TF‑IDF vectors capturing word importance  
- Word embeddings (optional: GloVe, Word2Vec)  

### 🧠 Algorithms & Architecture

| Model                 | Key Parameters             | Val. F1‑Score |
|-----------------------|----------------------------|---------------|
| Logistic Regression   | `C=1.0`, solver=`lbfgs`    | 0.92          |
| Random Forest         | `n_estimators=100`, `max_depth=10` | 0.90  |
| Linear SVM            | `C=0.5`, kernel=`linear`   | 0.91          |
| LSTM (Keras)          | 2 layers, 128‑unit, dropout=0.3 | 0.93    |

---
## 📊 Results & Evaluation
- **Accuracy:** 93.1%  
- **Precision / Recall / F1‑Score** for each class  
- **ROC AUC:** 0.96  

*(See `assets/roc_curve.png` & `assets/confusion_matrix.png`)*

---

## 🌟 Future Work
- 🔍 Hyperparameter tuning with **GridSearchCV**  
- 🧬 Explore Transformer‑based models (BERT, RoBERTa)  
- 🌐 Deploy as a **REST API** (FastAPI or Flask)  
- 📈 Build a web dashboard for real‑time predictions  

---

## 🤝 Contributing
1. Fork the repo  
2. Create a branch:  
   ```bash
   git checkout -b feature/YourFeature
3.Commit changes:
     git commit -m "Add new feature"
4.Push & open PR:
     git push origin feature/YourFeature
     
##📄 License
Distributed under the MIT License.

##📬 Contact
Kalyan Ram — kalyanr0508@gmail.com

