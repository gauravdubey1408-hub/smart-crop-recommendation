import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

def train_model():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "Crop.csv")

    df = pd.read_csv(file_path)

    X = df[['temperature','rainfall','humidity','N','P','K','ph']]
    y = df['crop']

    # Split data (important 🔥)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Better model 🚀
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)

    # Accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy