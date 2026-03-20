import numpy as np

import pandas as pd
df = pd.read_csv("insurance.csv")

df["sex"] = df["sex"].map({"male": 1, "female": 0})
df["smoker"] = df["smoker"].map({"yes": 1, "no": 0})
df["region"] = df["region"].map({"southwest": 1, "southeast": 0})
df.fillna(3 , inplace=True)
# print(df.head())

X = df.drop("charges", axis=1)
Y = df["charges"]
 #print(X)


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

from sklearn.linear_model import LinearRegression as lr
from sklearn.linear_model import Ridge as ridge
from sklearn.tree import DecisionTreeRegressor as dt
from sklearn.ensemble import RandomForestRegressor as rf
from sklearn.ensemble import GradientBoostingRegressor as gb
from sklearn.neighbors import KNeighborsRegressor as knn 



models = {
    "lr": lr(),
    "ridge": ridge(),
    "dt": dt(),
    "rf": rf(),
    "gb": gb(),
    "knn": knn()
}


# training all models
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
 
trained_models = {}

for name, model in models.items():
    model.fit(X_train, Y_train)
    trained_models[name] = model

print("All models trained ✅")

from sklearn.metrics import r2_score

for name, model in trained_models.items():
    y_pred = model.predict(X_test)
    print(name, ":", r2_score(Y_test, y_pred))



import pickle
for name, model in trained_models.items():
    pickle.dump(model, open(f"{name}.pkl", "wb"))

print("Models saved ✅")

import pickle
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Save the scaler
pickle.dump(scaler, open("scaler.pickle", "wb"))