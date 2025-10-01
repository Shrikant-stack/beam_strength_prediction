import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

df = pd.read_csv("beam_dataset.csv")

X = df.drop("Max_Load_kN", axis=1)
y = df["Max_Load_kN"]

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)


joblib.dump(model, "model.pkl")
print("✅ Model saved as model.pkl")
