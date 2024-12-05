import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt 
import kaggle
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import  GridSearchCV

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.linear_model import  LinearRegression
from sklearn.preprocessing import StandardScaler 

import seaborn as sns
kaggle.api.authenticate()
df=pd.read_csv("Housing.csv")
df["date"] = pd.to_datetime(df["date"])
df=df.drop("id",axis=1)
x=df[["bathrooms","lat","long","sqft_lot15","grade","waterfront","sqft_lot","bedrooms","yr_built","yr_renovated","condition",
      "view","sqft_living"]]
y=df["price"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 101)
train_data =x_train.join(y_train)
forest  = RandomForestRegressor()
forest.fit(x_train,y_train)
forest.score(x_test,y_test)

param_grid={
      "n_estimators": [100,200,300],
      "max_depth": [10,20,30],
      "min_samples_split": [2,5,10],
}      
grid = GridSearchCV(forest,param_grid, cv=5,scoring="neg_mean_squared_error",return_train_score=True)
grid.fit(x_train,y_train)
grid.score(x_test,y_test)