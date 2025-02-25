import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load dataset
df = pd.read_csv("D:\My Programs\House Price Prediction\Delhi.csv")

# Data Preprocessing

df.dropna(inplace=True)
label_enc = LabelEncoder()
df['Location'] = label_enc.fit_transform(df['Location'])
X = df.drop(columns=['Price'])  # Features
y = df['Price']  # Target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[['Location', 'Area', 'No. of Bedrooms']])
X_test_scaled = scaler.transform(X_test[['Location', 'Area', 'No. of Bedrooms']])


# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save the model and encoders
joblib.dump(scaler, "scaler.pkl")

# Flask API
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "House Price Prediction API is running!", 200


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    location = label_enc.transform([data['Location']])[0]
    size = data['Area']
    bhk = data['No. of Bedrooms']
    
    # Prepare input data
    input_data = scaler.transform([[location, size, bhk]])
    predicted_price = model.predict(input_data)[0]
    
    return jsonify({"predicted_price": predicted_price})

@app.route("/data", methods=["GET"])
def data():
    df['Decoded_Location'] = label_enc.inverse_transform(df['Location'])
    training_data = df[['Decoded_Location', 'Area', 'No. of Bedrooms']].rename(columns={"Decoded_Location": "Location"}).to_dict(orient='records')
    return jsonify({"training_data": training_data})

@app.route("/evaluate", methods=["GET"])
def evaluate():
    predictions = model.predict(X_test_scaled)
    errors = y_test - predictions
    evaluation = {
        "MAE": mean_absolute_error(y_test, predictions),
        "MSE": mean_squared_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "R² Score": r2_score(y_test, predictions),
        "Max Error": np.max(errors),
        "Min Error": np.min(errors)
    }
    return jsonify(evaluation)


if __name__ == "__main__":
    app.run(debug=True,port=5001)

