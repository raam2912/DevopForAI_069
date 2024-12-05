import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


data = pd.read_csv('credit_risk_dataset.csv')


data['debt_income_ratio'] = data['loan_amnt'] / data['person_income']


X = data[['person_income', 'debt_income_ratio', 'person_emp_length']]
y = data['cb_person_default_on_file']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

joblib.dump(model, 'credit_risk_model.pkl')