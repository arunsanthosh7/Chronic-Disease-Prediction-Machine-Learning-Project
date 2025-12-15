# Chronic Disease Prediction Using Machine Learning

## Project Overview
This project focuses on predicting chronic kidney disease outcomes using supervised machine learning techniques. The objective is to analyze a real-world healthcare dataset and build reliable classification models to assist in early disease identification and decision-making.

The project demonstrates an end-to-end machine learning workflow, including data preprocessing, exploratory data analysis, model training, evaluation, and comparison of multiple classification algorithms.

Dataset Source: Kaggle

---

## Tools & Technologies
- **Programming Language:** Python  
- **IDE:** VS Code  
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn  

---

## Dataset Description
The dataset contains patient clinical and laboratory information such as blood pressure, glucose levels, serum creatinine, haemoglobin, and comorbid conditions like hypertension and diabetes mellitus.  
The target variable indicates the presence or absence of chronic kidney disease.

---

## Project Workflow

### 1. Data Loading & Exploration
- Loaded dataset using Pandas
- Inspected structure, data types, and summary statistics
- Removed irrelevant identifier columns
- Renamed columns for clarity and consistency

### 2. Data Cleaning & Preprocessing
- Converted text-based numerical features to numeric format
- Handled missing values using:
  - Mean imputation for numerical features
  - Mode imputation for categorical features
- Cleaned inconsistent categorical labels
- Encoded categorical variables into numerical values

### 3. Exploratory Data Analysis (EDA)
- Performed correlation analysis using a heatmap
- Identified key clinical features strongly associated with the target variable
- Used insights to support model selection

### 4. Model Building
Implemented and trained the following classification models:
- Naive Bayes (GaussianNB)
- K-Nearest Neighbors (KNN)
- Random Forest Classifier
- Decision Tree Classifier
- Support Vector Machine (Linear SVC)

### 5. Model Evaluation
Models were evaluated using standard classification metrics:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## Model Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score |
|-----|---------|----------|--------|---------|
| Naive Bayes | 0.95 | 1.00 | 0.92 | 0.96 |
| KNN | 0.76 | 0.88 | 0.71 | 0.79 |
| Random Forest | **0.98** | 0.97 | **1.00** | **0.98** |
| Decision Tree | 0.97 | 0.97 | 0.98 | 0.98 |
| SVM (Linear) | 0.95 | 0.95 | 0.97 | 0.96 |

---

## Key Results & Insights
- Random Forest achieved the best overall performance with **98% accuracy and 100% recall**
- Tree-based models handled non-linear relationships effectively
- Correlation analysis validated the importance of medical features such as serum creatinine, blood urea, and haemoglobin

---

## Future Improvements
- Hyperparameter tuning for performance optimization
- Cross-validation for improved generalization
- Feature selection and dimensionality reduction
- Deployment using Flask or Streamlit

---

## Author
**Arun Santhosh M**  
Aspiring Data Analyst | Machine Learning Enthusiast  

---

## Acknowledgements
- Kaggle for the dataset
- Scikit-learn and open-source Python community

