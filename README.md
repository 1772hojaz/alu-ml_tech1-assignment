#  Sentiment Analysis on IMDB Movie Reviews using Traditional Machine Learning Models

##  Project Overview

This project focuses on building a **sentiment classification system** using **traditional machine learning algorithms** on the [IMDB 50K Movie Reviews dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). The goal is to classify movie reviews as either **positive** or **negative** using models such as **Logistic Regression** and **Support Vector Machine (SVM)**.

The project is part of an academic assignment that aims to compare traditional ML models with deep learning approaches for natural language processing (NLP) tasks.

---

##  Aim

To develop, evaluate, and compare the performance of traditional machine learning models—specifically **Logistic Regression** and **SVM (LinearSVC)**—for **binary sentiment analysis** and LSTM models using text reviews from IMDB. This includes preprocessing, feature extraction, training, evaluation, and result visualization.

---

##  Dataset

- **Source**: [Kaggle - IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Size**: 50,000 labeled reviews
- **Classes**: Positive, Negative (binary classification)
- **Balance**: Dataset is evenly balanced between the two classes

---

##  Workflow

1. **Data Cleaning & Preprocessing**
   - Lowercasing
   - Removing HTML tags, punctuation, and special characters
   - Stopword removal
   - Stemming using `PorterStemmer`

2. **Feature Engineering**
   - Text vectorization using **TF-IDF**

3. **Model Training**
   - Logistic Regression
   - Support Vector Machine (LinearSVC)

4. **Evaluation**
   - Accuracy
   - Precision, Recall, F1-Score
   - Confusion Matrix
   - Training time measurement
   - Comparative analysis

5. **Visualization**
   - Confusion matrices for both models
   - Accuracy and F1-score comparisons

---

##  Experiments

The project includes multiple experiments to evaluate the effect of:
- Different model parameters (e.g., `C` for SVM, `max_iter` for Logistic Regression)
- Vectorizer configurations (e.g., `max_features`, `ngram_range`)

---

##  Results

| Model              | Accuracy | Precision | Recall | F1-Score |
|-------------------|----------|-----------|--------|----------|
| Logistic Regression | ~88.8%   | ~0.89      | ~0.89   | ~0.89     |
| SVM (LinearSVC)     | ~89.6%   | ~0.90      | ~0.89   | ~0.90     |

- The models achieved strong results.
- For traditional models, SVM slightly outperformed Logistic Regression in overall metrics.

---

##  Repository Structure

 Sentiment-Analysis-IMDB
|
├── data/
│ └── IMDB Dataset.csv
├── notebooks/
| └── DataExplorationNotebook.ipynb
│ └── TraditionalModels_Sentiment_Analysis.ipynb
| └── LSTMModels_Sentiment_Analysis.ipynb
├── models/
│ └── logistic_regression_model.pkl
│ └── svm_pipeline_model.pkl
├── README.md
├── requirements.txt


---

##  Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- NLTK
- Matplotlib, Seaborn
- Jupyter Notebook

---

## Team Contribution

- **Jules Gatete**: Data exploration, Data visualization + General preprocessing
- **Audry Ashleen Chivanga**: Traditional model implementation : Logistic Regression, SVM model
-- **Humphrey Nyahoja** LSTM  models  implementation + Visualizations
- **Samuel Dushime**: Result collection , Documentation + Report writing

---

##  Citation

For the dataset, please cite:
> Lakshmi Narayan Pathi, "IMDB Dataset of 50K Movie Reviews", Kaggle, 2018.

---

## Status

 Completed: 1. Traditional model implementation, evaluation, and documentation.  
               2. Deep learning model comparison (RNN, LSTM).

---

##  License

This project is for academic purposes only under African Leadership University . Dataset is publicly available from Kaggle under their license.

