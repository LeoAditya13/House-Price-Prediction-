# 🏡 House Price Prediction API – Backend Summary  

This **Flask-based Machine Learning API** predicts house prices based on **location, area, and number of bedrooms**. It processes user inputs, applies a trained model, and returns accurate price estimates.  

---

## 🚀 Backend Workflow  

### 🔹 **1. Data Processing**  
- Loads **Delhi.csv** (real estate dataset).  
- Encodes **categorical data (Location)** into numerical values.  
- Scales numerical features (**Area, Bedrooms**) for better model accuracy.  

### 🔹 **2. Model Training**  
- Uses **MLPRegressor (Neural Network)** for prediction.  
- Trained using **historical housing data**.  
- Evaluates performance using **MAE, RMSE, and R² Score**.  

### 🔹 **3. API Development (Flask)**  
- Built using **Flask**, with the following key endpoints:  

| **Endpoint** | **Method** | **Description** |
|-------------|------------|----------------|
| `/predict` | `POST` | Takes input (Location, Area, Bedrooms) & returns predicted price. |
| `/evaluate` | `GET` | Returns model accuracy metrics (MAE, RMSE, R² Score). |
| `/data` | `GET` | Provides dataset info used for training. |

### 🔹 **4. Deployment**  
- Hosted on **PythonAnywhere** for public access.  
- Uses **Gunicorn** for production-ready deployment.  
- API can be tested via **Postman, cURL, or Python requests**.  

---

## 🎉 Project Highlights  
✅ **Real-time price predictions** with high accuracy.  
✅ **Efficient ML model with optimized data processing.**  
✅ **Successfully published on NICEDT**, showcasing AI-driven real estate insights.  
