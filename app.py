import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import gradio as gr

MODEL_FILE = 'usa_housing_logistic_model.joblib'
SCALER_FILE = 'scaler.joblib'
DATA_FILE = 'USA_Housing.csv'

# =====================================================================
# 1. MODEL TRAINING & PREPARATION
# =====================================================================
def train_and_save_model():
    print("--- 1. LOADING & PREPROCESSING DATA ---")
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"'{DATA_FILE}' not found! Please check your folder.")
    
    df = pd.read_csv(DATA_FILE)
    
    # Drop address if present
    if 'Address' in df.columns:
        df = df.drop(columns=['Address'])
        
    X = df[['Avg. Area Income', 'Avg. Area House Age', 
            'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 
            'Area Population']]
    
    # --- LOGISTIC REGRESSION STEP ---
    # Convert numeric 'Price' into Binary Category (1 = Expensive, 0 = Affordable)
    # Threshold: $1.2 Million
    y = (df['Price'] > 1_200_000).astype(int)
    
    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature Scaling: Logistic Regression works best when features are on similar scales
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("--- 2. TRAINING LOGISTIC REGRESSION MODEL ---")
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model accuracy
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Trained Successfully! Accuracy: {acc * 100:.2f}%")
    
    # Save Model & Scaler
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print(f"Saved model to '{MODEL_FILE}' and scaler to '{SCALER_FILE}'\n")
    return model, scaler

# Load existing model/scaler or train new ones
if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
    print(f"Loading saved model and scaler...")
    try:
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
    except Exception:
        print("Error loading existing files. Re-training model...")
        model, scaler = train_and_save_model()
else:
    model, scaler = train_and_save_model()

# =====================================================================
# 2. PREDICTION FUNCTION FOR GRADIO
# =====================================================================
def predict_house_category(income, age, rooms, bedrooms, population):
    input_df = pd.DataFrame([{
        'Avg. Area Income': income,
        'Avg. Area House Age': age,
        'Avg. Area Number of Rooms': rooms,
        'Avg. Area Number of Bedrooms': bedrooms,
        'Area Population': population
    }])
    
    # 1. Scale the user's input data using the trained scaler
    input_scaled = scaler.transform(input_df)
    
    # 2. Predict Probability & Class
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    
    expensive_prob = probabilities[1] * 100
    affordable_prob = probabilities[0] * 100
    
    if prediction == 1:
        result = f"🏰 **Expensive House** (Confidence: {expensive_prob:.1f}%)"
    else:
        result = f"🏡 **Affordable House** (Confidence: {affordable_prob:.1f}%)"
        
    return result

# =====================================================================
# 3. GRADIO USER INTERFACE (UI)
# =====================================================================
interface = gr.Interface(
    fn=predict_house_category,
    inputs=[
        gr.Number(label="Avg. Area Income ($)", value=65000),
        gr.Slider(minimum=1.0, maximum=15.0, value=6.0, step=0.1, label="Avg. Area House Age (Years)"),
        gr.Slider(minimum=1.0, maximum=12.0, value=6.0, step=0.1, label="Avg. Number of Rooms"),
        gr.Slider(minimum=1.0, maximum=10.0, value=4.0, step=0.1, label="Avg. Number of Bedrooms"),
        gr.Number(label="Area Population", value=32000)
    ],
    outputs=gr.Markdown(label="House Classification"),
    title="🏠 USA House Tier Classifier (Logistic Regression)",
    description="Enter details to classify if the house is likely **Expensive** (Over $1.2M) or **Affordable**."
)

if __name__ == "__main__":
    interface.launch(share=True)