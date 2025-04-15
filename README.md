# 🩺 Heart Disease Prediction & 📈 Time Series Sales Forecasting

## 🔍 Project Overview

This repository includes two major machine learning projects:

1. **Heart Disease Prediction using Logistic Regression**  
   Predict whether a patient has heart disease based on medical parameters.

2. **Sales Forecasting using Time Series Analysis (ARIMA)**  
   Analyze historical sales trends and forecast future sales using ARIMA.

---

## 📁 Project 1: Heart Disease Prediction

### 🎯 Objective
Predict the presence of heart disease in a patient using key features like age, gender, cholesterol level, and blood pressure.

### 📂 Dataset
- **Filename**: `heart_disease.csv`
- **Features**:
  - `Age`: Patient's age
  - `Gender`: Male or Female
  - `Cholesterol`: Cholesterol levels
  - `Blood Pressure`: Systolic/Diastolic pressure (e.g., 120/80)
  - `Heart Disease`: Target variable (0 = No, 1 = Yes)

### ✅ Tasks Performed
1. **Data Cleaning**
   - Checked and handled missing or inconsistent values.
   - Encoded categorical values.
   - Split blood pressure into `Systolic BP` and `Diastolic BP`.

2. **Feature Engineering**
   - Scaled numerical features using `StandardScaler`.

3. **Model Training**
   - Trained a **Logistic Regression** model with class balancing.

4. **Model Evaluation**
   - Used metrics:
     - Confusion Matrix
     - Accuracy
     - Precision
     - Recall
     - F1-score

### 📊 Deliverables
- Trained Logistic Regression Model
- Evaluation Report with performance metrics

---

## 📁 Project 2: Sales Forecasting with Time Series

### 🎯 Objective
Use time series analysis to forecast future sales values.

### 📂 Dataset
- Minimum two columns:
  - `Date`: Sales date
  - `Sales`: Revenue or number of units sold

### ✅ Tasks Performed
1. **Trend Analysis**
   - Visualized sales patterns over time
   - Identified trends and seasonality using line plots and moving averages

2. **Forecasting with ARIMA**
   - Modeled time series using **ARIMA**
   - Generated predictions for future sales
   - Evaluated performance using:
     - RMSE (Root Mean Squared Error)
     - MAPE (Mean Absolute Percentage Error)

### 📊 Deliverables
- Forecasted sales values
- Trend and forecast visualization plots

---

## 👨‍💻 Author

**Berlin Samvel Pandian S.**  
🎓 Artificial Intelligence & Machine Learning  
🔗 [LinkedIn Profile](https://www.linkedin.com/in/s-berlin-samvel-pandian007)

---

## 📌 How to Run

1. Clone the repository
```bash
git clone https://github.com/your-username/your-repo-name.git
