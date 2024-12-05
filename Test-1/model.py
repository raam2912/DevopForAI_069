import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib


data_path = 'online_shopping.csv' 
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
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


joblib.dump(model, 'online_shopping_model.pkl')