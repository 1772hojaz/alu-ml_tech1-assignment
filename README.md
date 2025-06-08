
# Sentiment Analysis Assignment – Group 1

Welcome to the **Sentiment Analysis Assignment** repository by **Group 1**!
This project focuses on classifying **IMDB movie reviews** as **positive** or **negative** using both **traditional machine learning** (Logistic Regression) and **deep learning** (LSTM) techniques.

---

## Repository Structure

```
.
├── Dataset
│   └── IMDB Dataset.csv
├── models
│   ├── best_logistic_model.pkl
│   ├── best_lstm_model.keras
│   ├── tokenizer.pkl
│   └── vectorizer.pkl
├── notebooks
│   └── Sentiment_Classification_Notebook.ipynb
└── README.md
```

### Descriptions:

* **Dataset/**: Contains the IMDB Dataset with 50,000 labeled movie reviews.
* **models/**: Stores trained models and preprocessing tools:

  * `best_logistic_model.pkl`: Trained Logistic Regression model
  * `best_lstm_model.keras`: Trained LSTM deep learning model
  * `tokenizer.pkl`: Keras tokenizer used for LSTM preprocessing
  * `vectorizer.pkl`: TF-IDF or CountVectorizer used for Logistic Regression
* **notebooks/**:

  * `Sentiment_Classification_Notebook.ipynb`: End-to-end Jupyter notebook with preprocessing, model training, and evaluation.
* **README.md**: You're reading it now!

---

## Project Overview

We use the **IMDB Dataset** of 50,000 balanced reviews to train models for binary sentiment classification (positive = 1, negative = 0). The goal is to compare:

* A **Logistic Regression** model with TF-IDF/Count vectorization.
* An **LSTM** model capturing sequential context from reviews.

The project includes:

* Data preprocessing
* Feature engineering
* Hyperparameter tuning
* Model training and evaluation

---

## Group Members

* **Humphrey Jones Gabarinocheka Nyahoja**
* **Samuel Dushime**
* **Audry Ashleen Chivanga**
* **Jules Gatete**

---

## ⚙️ Setup Instructions

### Prerequisites

Ensure you have **Python 3.x** and install the required libraries:

```bash
pip install numpy pandas scikit-learn tensorflow keras nltk matplotlib seaborn joblib
```

### Installation

```bash
git clone https://github.com/1772hojaz/alu-ml_tech1-assignment.git
cd alu-ml_tech1-assignment
```

Ensure `Dataset/IMDB Dataset.csv` is in place.

---

## 🚀 Usage

1. Open the notebook:

   ```bash
   jupyter notebook notebooks/Sentiment_Classification_Notebook.ipynb
   ```
2. Run the cells to:

   * Load and clean the dataset
   * Preprocess text
   * Train and evaluate Logistic Regression and LSTM models
   * Visualize results with heatmaps and plots

Trained models and vectorizers/tokenizers are saved in the `models/` directory for reuse.

---

## Making Predictions

To predict sentiments on new text:

1. Load the saved model (`.pkl` or `.keras`) and associated vectorizer/tokenizer.
2. Apply the same preprocessing steps.
3. Use `.predict()` to get the sentiment label (0 = negative, 1 = positive).

---

## Dataset Description

* **Source**: [Kaggle IMDB Movie Reviews Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
* **Format**: CSV with 2 columns: `review` (text) and `sentiment` (0/1)
* **Balance**: Equal distribution of positive and negative reviews
* **Length**: Varies from short comments to long paragraphs

---

## Preprocessing Steps

### Text Processing

* Remove HTML tags (e.g., `<br />`)
* Convert to lowercase
* Remove punctuation

### Tokenization & Cleaning

* Tokenized using `nltk.word_tokenize`
* Remove stopwords (`nltk.corpus.stopwords`)

### Feature Engineering

* **TF-IDF Vectorizer**: 10,000 features with bigrams
* **Count Vectorizer**: Raw term frequencies (same config)

### LSTM Preprocessing

* Tokenized using **Keras Tokenizer** (vocab size = 10,000)
* Sequences padded to max length of 200

### Train/Test Split

* 80% training, 20% testing
* `random_state=42`

---

## Model Choices

### 1. **Logistic Regression**

* Simple and fast linear model
* Works with TF-IDF or Count Vectorized input
* Tuned with:

  * Regularization `C`
  * Solver: `liblinear`, `saga`

### 2. **LSTM (Long Short-Term Memory)**

* Deep sequential model for learning patterns in text
* Architecture:

  * Embedding Layer
  * LSTM Layer
  * Dropout Layer
  * Dense Output (sigmoid)
* Tuned with:

  * Embedding dim, LSTM units, dropout rate
  * Learning rate, batch size, optimizer (`Adam` or `Nadam`)

---

## Experiment Results

### Logistic Regression

| Exp | Vectorizer | C   | Solver    | Accuracy   | Precision | Recall | F1         |
| --- | ---------- | --- | --------- | ---------- | --------- | ------ | ---------- |
| 0   | TF-IDF     | 0.1 | liblinear | 0.8750     | 0.8615    | 0.8960 | 0.8784     |
| 1   | TF-IDF     | 1.0 | liblinear | **0.8949** | 0.8852    | 0.9093 | **0.8971** |
| 2   | TF-IDF     | 1.0 | saga      | 0.8949     | 0.8852    | 0.9093 | 0.8971     |
| 3   | Count      | 0.1 | liblinear | 0.8910     | 0.8853    | 0.9004 | 0.8928     |
| 4   | Count      | 1.0 | liblinear | 0.8746     | 0.8748    | 0.8766 | 0.8757     |

**Best configuration**: TF-IDF + C=1.0 + liblinear or saga
**Observation**: TF-IDF outperformed Count Vectorizer; solver choice had minimal impact.

---

### LSTM

| Exp | Embed Dim | LSTM Units | Dropout | LR     | Batch | Optim | Accuracy   | Precision  | Recall | F1         |
| --- | --------- | ---------- | ------- | ------ | ----- | ----- | ---------- | ---------- | ------ | ---------- |
| 0   | 50        | 64         | 0.3     | 0.001  | 32    | Adam  | 0.8685     | 0.8602     | 0.8825 | 0.8700     |
| 1   | 100       | 64         | 0.5     | 0.001  | 64    | Adam  | **0.8763** | 0.8796     | 0.8742 | **0.8770** |
| 2   | 100       | 128        | 0.5     | 0.0005 | 32    | Nadam | 0.8659     | 0.8790     | 0.8509 | 0.8640     |
| 3   | 50        | 64         | 0.3     | 0.001  | 32    | Nadam | 0.8631     | **0.8956** | 0.8243 | 0.8580     |

**Best configuration**: Experiment 1 (F1 = 0.8770)
**Observation**: Higher embedding dimension and batch size helped improve accuracy. Nadam boosted precision but lowered recall.

---

## Model Comparison

| Model               | Accuracy   | F1 Score   |
| ------------------- | ---------- | ---------- |
| Logistic Regression | **0.8949** | **0.8971** |
| LSTM                | 0.8763     | 0.8770     |

* **Logistic Regression** outperformed LSTM slightly on F1 and accuracy.
* **LSTM**, despite being more complex, showed competitive performance, benefiting from sequential data structure.

---

## Conclusion

* **Best Performing Model**: Logistic Regression with TF-IDF, C=1.0, solver=liblinear/saga
* **Best Deep Learning Model**: LSTM with embedding\_dim=100, units=64, dropout=0.5, optimizer=Adam
* This project demonstrates how both classic and deep learning models can effectively handle sentiment classification tasks.

---




