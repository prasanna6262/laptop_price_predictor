from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib
from .preprocess import load_and_preprocess_data
from .utils import save_model

def train_model():
    X_train, X_test, y_train, y_test, label_encoders = load_and_preprocess_data("data/laptops.csv")

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse=np.sqrt(mse)

    print(f" RMSE: ",rmse)
    
    save_model(model, label_encoders, "models/laptop_price_model.pkl", "models/label_encoders.pkl")
