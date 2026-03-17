# Telecom Customer Churn Prediction

End-to-end machine learning project focused on predicting customer churn in a telecom company using classification models and model explainability techniques.

---

## Project Overview

Customer churn is one of the most important business problems in subscription-based companies. Predicting which customers are likely to leave can help businesses improve retention strategies and reduce revenue loss.

In this project, I analyze a telecom customer dataset and build multiple machine learning models to predict churn based on customer demographics, contract information, and service usage patterns.

---

## Business Problem

Customer acquisition is usually more expensive than customer retention.  
A churn prediction model can help identify high-risk customers early and support targeted retention actions.

The objective of this project is to answer two key questions:

1. Can we predict which customers are likely to churn?
2. Which factors are most strongly associated with churn?

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## Project Structure

```text
telecom-customer-churn-ml/
│
├── data/
│   └── telco_churn.csv
│
├── notebooks/
│   ├── churn_model.ipynb
│   ├── churn_model.html
│   └── shap_summary.png
│
├── README.md
└── requirements.txt
```

---

## Workflow

### 1. Exploratory Data Analysis
- Inspect dataset structure and missing values
- Understand churn distribution
- Explore customer, contract, and billing variables

### 2. Data Preprocessing
- Clean and encode categorical features
- Prepare target variable
- Split data into training and test sets
- Scale features when required

### 3. Model Training
Three classification models were trained and compared:

- Logistic Regression
- Random Forest
- XGBoost

### 4. Model Evaluation
Models were evaluated using:

- Accuracy
- Precision
- Recall
- ROC-AUC
- Classification report
- ROC curve comparison

### 5. Model Interpretation
To understand the drivers of churn, the project includes:

- XGBoost feature importance
- SHAP explainability

---

## Model Performance

| Model | Accuracy | ROC-AUC |
|------|------|------|
| XGBoost | 0.80 | 0.83 |
| Logistic Regression | 0.79 | 0.83 |
| Random Forest | 0.78 | 0.82 |

XGBoost achieved the best overall performance among the tested models.

---

## Key Insights

- Customers with **short tenure** are more likely to churn
- **Month-to-month contracts** are strongly associated with higher churn risk
- **Higher monthly charges** are associated with increased churn probability
- **Longer contracts** are linked to lower churn risk
- **Fiber optic internet service** appears strongly associated with churn in this dataset

---

## Explainability

Feature importance and SHAP analysis were used to better understand model behavior.

These methods help move beyond prediction alone and answer a more practical business question:

**Why is a customer likely to churn?**

SHAP summary plot showing feature impact on churn prediction:

![SHAP Summary](notebooks/shap_summary.png)

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/juampymv/telecom-customer-churn-ml.git
cd telecom-customer-churn-ml
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Open the notebook
```bash
jupyter notebook
```

Then open:

```text
notebooks/churn_model.ipynb
```

---

## What This Project Demonstrates

- End-to-end machine learning workflow
- Model comparison using multiple classifiers
- Use of business-oriented evaluation metrics
- Ability to interpret model predictions with feature importance and SHAP
- Clear connection between predictive modeling and business retention strategy

---

## Next Steps

- Improve recall through threshold optimization
- Perform hyperparameter tuning
- Build a retention-focused business recommendation layer
- Deploy the model as an interactive application

---

## Author

Juan Pablo Moreno  
Data Scientist
