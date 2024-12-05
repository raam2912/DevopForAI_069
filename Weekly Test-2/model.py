import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

data_path = 'online_shopping_pref.csv' 
data = pd.read_csv(data_path)

data.columns = ['Timestamp', 'Shopping_Frequency', 'Age_Group', 'Electronics_Platform',
                'Fashion_Platform', 'Beauty_Platform', 'Grocery_Platform', 
                'Important_Factor', 'Trust_Reviews', 'Best_Return_Policy']
data = data.drop(columns=['Timestamp'])
data = data.dropna()


X = data.drop(columns=['Important_Factor'])
X = pd.get_dummies(X, drop_first=True)


y = data['Important_Factor']
y = pd.factorize(y)[0]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

lr_y_pred = lr_model.predict(X_test)


print("Random Forest Results:")
print(confusion_matrix(y_test, rf_y_pred))
print(classification_report(y_test, rf_y_pred))
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_y_pred):.4f}\n")

print("Logistic Regression Results:")
print(confusion_matrix(y_test, lr_y_pred))
print(classification_report(y_test, lr_y_pred))
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, lr_y_pred):.4f}\n")


joblib.dump(rf_model, 'random_forest_model.pkl')
joblib.dump(lr_model, 'logistic_regression_model.pkl')


