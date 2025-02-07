# Sentiment Analysis on Tweets: Predictive Modeling

This project focuses on the analysis and classification of tweets based on sentiment labels (`Positive`, `Neutral`, `Negative`, and `Irrelevant`). It utilizes machine learning techniques to preprocess data, train classifiers, and evaluate models for optimal performance. The analysis also involves exploratory data visualization and sentiment-based insights for various companies.

---

## Project Overview

### Objectives:
1. Preprocess and clean tweet data for effective analysis.
2. Train and evaluate multiple machine learning classifiers to predict sentiment labels.
3. Provide actionable insights based on tweet sentiment distribution and company-specific analysis.

### Dataset:
- **Training Dataset**: 73,996 tweets with labels.
- **Validation Dataset**: 1,000 tweets with labels.

---

## Methodology

### Data Preprocessing:
- Removed missing values, URLs, hashtags, mentions, and special characters.
- Applied tokenization, stopword removal, and lemmatization for text cleaning.
- Extracted features using **TF-IDF Vectorizer** and **Count Vectorizer**.

### Exploratory Data Analysis (EDA):
- Distribution of sentiment labels and tweet lengths visualized.
- Analysis of top companies by tweet volume.
- Word cloud generated to highlight common terms.

### Machine Learning Models:
1. **Decision Tree Classifier**
2. **Random Forest Classifier**
3. **Gradient Boosting Classifier**
4. **AdaBoost Classifier**
5. **Multinomial Naive Bayes**

### Metrics Evaluated:
- Precision, Recall, F1-Score, Accuracy, AUC Score.
- Misclassifications for error analysis.

---

## Results

| Model                       | Train Accuracy | Test Accuracy | Validation Accuracy | F1 Score | AUC Score |
|-----------------------------|----------------|----------------|----------------------|----------|-----------|
| Decision Tree (TF)          | 97.27%         | 77.77%         | 91.7%               | 77.70%   | 86.50%    |
| Random Forest (TF)          | 97.27%         | 89.50%         | 97.2%               | 89.49%   | 98.19%    |
| Multinomial NB (TF)         | 75.85%         | 68.97%         | 78.6%               | 67.34%   | 90.96%    |
| Gradient Boosting (TF)      | 53.96%         | 51.27%         | 54.7%               | 48.26%   | 76.72%    |
| AdaBoost (TF)               | 46.33%         | 45.01%         | 44.8%               | 41.43%   | 67.44%    |

---

## Visualizations

1. **Sentiment Label Distribution**:
   ![Sentiment Distribution](image_path_here)

2. **Top 10 Companies by Tweet Volume**:
   ![Top Companies](image_path_here)

3. **Tweet Length Distribution**:
   ![Tweet Length](image_path_here)

4. **Word Cloud**:
   ![Word Cloud](image_path_here)

---

## Features

- **Pipeline Architecture**: Data preprocessing, feature extraction, and classification integrated into scikit-learn pipelines.
- **Validation**: Real-world tweet data used to validate model performance.
- **Interpretability**: Visualizations for both data and model results.

---

## Conclusion

- **Best Model**: Random Forest Classifier achieved the highest accuracy and AUC across test and validation datasets.
- **Business Insight**: Sentiment analysis provides valuable insights into public perception of various companies.
- **Future Work**:
  - Fine-tune hyperparameters for Gradient Boosting and AdaBoost.
  - Incorporate sentiment-specific keywords for improved feature extraction.
  - Expand analysis to more companies and diverse datasets.

---

## Dependencies

- Python (>=3.7)
- Pandas
- NumPy
- scikit-learn
- matplotlib
- seaborn
- wordcloud
- nltk
- yellowbrick

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/niranjankumarnk/Sentiment-Analysis-Tweets.git
