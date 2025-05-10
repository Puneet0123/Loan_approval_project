## Loan Approval Prediction Project

A full end-to-end machine learning project to predict loan approvals using applicant data. This project automates credit risk assessment and provides transparent insights via dashboards and model explainability techniques.

---

### Problem Statement

Loan officers in financial institutions need to make quick, accurate, and fair decisions about approving or rejecting loan applications. This project builds a machine learning model that predicts whether a loan should be approved based on applicant details such as income, credit history, loan amount, and more.

---


### Key Features

-  Clean and well-commented Jupyter notebooks
- In-depth EDA with feature distribution, correlation, and insights
- Robust preprocessing pipeline for missing values and encoding
- Multiple models tried: Logistic Regression, Random Forest, XGBoost
- Hyperparameter tuning with cross-validation (`GridSearchCV`)
- Threshold tuning to balance precision & recall based on business goals
- Feature importance interpretation
- Interactive dashboard for decision-makers
- Model serialized for deployment

---

### Final Model Performance

| Metric     | Value   |
|------------|---------|
| Accuracy   | 0.849   |
| Precision  | 0.85    |
| Recall     | 0.95    |
| F1-Score   | 0.897   |

Final model: **RandomForestClassifier**  
Precision prioritized to reduce bad loan approvals  
Threshold tuning applied for better business alignment

---

### Dashboard

Interactive dashboards are included in the `/dashboards/` folder to help stakeholders:
- Visualize approval trends
- Understand customer segments
- Monitor key risk features (e.g., Credit History, Income)

---

### Feature Importance

Top features influencing model decisions:
- `Credit_History`
- `ApplicantIncome`
- `LoanAmount`
- `Married`
- `Education`

These align with real-world banking criteria and provide model transparency.

---

### Tools & Libraries

- Python, Pandas, NumPy, Scikit-learn
- Seaborn & Matplotlib for data viz
- XGBoost, RandomForestClassifier
- Power BI for dashboards
- `joblib` for model serialization

---

### Dataset

- Source: [Dataset] (https://www.kaggle.com/datasets/ninzaami/loan-predication)
- Size: ~600 records, 13+ features

---

### Author

**Puneet Saini**  
Aspiring Data Scientist | Machine Learning Enthusiast  
sainipuneet471@gmail.com | [Linkedin](https://www.linkedin.com/in/puneet471/)

---

### Final Words

This project demonstrates how machine learning can drive smarter financial decisions. It combines strong modeling with explainability and business reasoning, making it suitable for real-world deployment and stakeholder presentation.



