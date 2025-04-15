# Heart Disease Prediction Using Logistic Regression

## Project Overview
This project aims to predict whether a patient has heart disease based on key medical parameters. We use a logistic regression model to classify patients as having heart disease or not, based on features like age, gender, cholesterol levels, and blood pressure. The project demonstrates data preprocessing, feature engineering, model training, and evaluation techniques in machine learning.

---

## Dataset
The dataset used in this project is `heart_disease.csv` and contains the following columns:

- **Age**: Patient's age
- **Gender**: Patient's gender (Male/Female)
- **Cholesterol**: Cholesterol levels (numeric)
- **Blood Pressure**: Systolic/Diastolic blood pressure (string, e.g., '120/80')
- **Heart Disease**: Target variable (1 for patients with heart disease, 0 for patients without)

---

## Tasks Performed
### 1. Data Loading and Cleaning
- Loaded the dataset using pandas.
- Checked for missing or inconsistent data.
- Handled missing values and removed duplicates.
- Converted categorical variables like gender into numerical values.
- Split the "Blood Pressure" column into two separate columns for systolic and diastolic pressures.
  
### 2. Feature Engineering
- Scaled the numerical features (Age, Cholesterol, Systolic BP, Diastolic BP) using `StandardScaler` to improve model performance.
  
### 3. Model Training
- Trained a logistic regression model to predict heart disease using the cleaned and preprocessed data.
- Used class weighting to handle class imbalance.

### 4. Model Evaluation
- Evaluated the model's performance using accuracy, precision, recall, and F1-score.
- Displayed the confusion matrix to show the model’s predictions compared to the actual values.

---

## Installation
### Requirements
- Python 3.x
- Pandas
- Scikit-learn
- Matplotlib (for visualization)

### linkedin profile link:- www.linkedin.com/in/s-berlin-samvel-pandian007
