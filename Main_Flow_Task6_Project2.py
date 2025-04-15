import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
df = pd.read_csv(r'A:\PycharmProjects\Main__Flow_Task4\Main_flow_Task4\heartdisease.csv')

# Check for missing data
print("Missing data per column:")
print(df.isnull().sum())

# Clean the dataset
df.drop_duplicates(inplace=True)  # Remove duplicates
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})  # Encode Gender
df[['Systolic BP', 'Diastolic BP']] = df['Blood Pressure'].str.split('/', expand=True)
df[['Systolic BP', 'Diastolic BP']] = df[['Systolic BP', 'Diastolic BP']].apply(pd.to_numeric)
df.drop('Blood Pressure', axis=1, inplace=True)
df.fillna(df.mean(), inplace=True)  # Handle missing values

# Feature Engineering
numerical_columns = ['Age', 'Cholesterol', 'Systolic BP', 'Diastolic BP']
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])  # Scale features

# Model Training
X = df.drop('Heart Disease', axis=1)
y = df['Heart Disease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, zero_division=1)
f1 = f1_score(y_test, y_pred)

print("\nModel Evaluation Report:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")
