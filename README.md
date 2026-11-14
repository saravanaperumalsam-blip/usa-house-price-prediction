# 🏠 USA House Price Prediction (Linear & Logistic Regression)

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white) ![Gradio](https://img.shields.io/badge/UI-Gradio-FF7C00) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)

Two machine learning models trained on the **USA Housing** dataset, each with an interactive **Gradio** web app:

1. **Linear Regression** predicts a house's **price** from neighbourhood features.
2. **Logistic Regression** classifies a house as **Expensive (over $1.2M)** or **Affordable**, with a confidence score.

## 📂 Dataset

`USA_Housing.csv`: 5,000 houses with these columns:

| Feature | Description |
|---|---|
| Avg. Area Income | Average income of residents in the area |
| Avg. Area House Age | Average age of houses in the area |
| Avg. Area Number of Rooms | Average number of rooms |
| Avg. Area Number of Bedrooms | Average number of bedrooms |
| Area Population | Population of the area |
| **Price** | Sale price (target) |
| Address | Dropped before training |

Prices range from about $16K to $2.47M, with a median of $1.23M.

## ⚙️ Approach

- Drop the `Address` text column and use the 5 numeric features
- 80/20 train/test split (`random_state=42`)
- **Price model**: `LinearRegression` on the raw features
- **Expensive/Affordable classifier**: target is `Price > 1,200,000` (54% of houses are "Expensive"); features scaled with `StandardScaler`, then `LogisticRegression`
- Trained models are saved with `joblib` and reloaded by the apps, so they only retrain if the model file is missing

## 📈 Results (test set, 1,000 houses)

**Linear Regression: price prediction**

| Metric | Score |
|---|---|
| R² | **0.918** |
| MAE | $80,879 |
| RMSE | $100,444 |

What each feature adds to the price, holding the others constant:

| Feature | Effect on price |
|---|---|
| +1 year of average house age | +$164,666 |
| +1 average room | +$119,624 |
| +1 average bedroom | +$2,440 |
| +$1 average area income | +$21.65 |
| +1 resident in area population | +$15.27 |

**Logistic Regression: Expensive vs. Affordable**

| Metric | Score |
|---|---|
| Accuracy | **89.7%** |

## 🖥️ Apps

| File | Model | What it does |
|---|---|---|
| `Linear-regresssion.ipynb` | Linear Regression | Trains the price model and launches a Gradio app that shows an estimated price |
| `app.py` | Logistic Regression | Trains the classifier and launches a Gradio app that labels the house **Expensive** or **Affordable** with a confidence score |

Both apps take the same inputs: average area income, house age, rooms, bedrooms and area population.

## 🚀 How to run

```bash
git clone https://github.com/saravanaperumalsam-blip/usa-house-price-prediction.git
cd usa-house-price-prediction
pip install -r requirements.txt
```

Run the classifier app:

```bash
python app.py
```

Then open the local URL printed in the terminal (usually http://127.0.0.1:7860). For the price predictor, open `Linear-regresssion.ipynb` in Jupyter and run the cell.

## 🗂 Project structure

```
├── app.py                               # Gradio app: Expensive/Affordable classifier
├── Linear-regresssion.ipynb             # Price prediction model + Gradio app
├── USA_Housing.csv                      # Dataset
├── usa_housing_model.joblib             # Saved Linear Regression model
├── usa_housing_logistic_model.joblib    # Saved Logistic Regression model
├── scaler.joblib                        # Saved StandardScaler for the classifier
├── requirements.txt
└── README.md
```

## 🛠 Tech stack

Python · pandas · NumPy · scikit-learn · joblib · Gradio · Jupyter

## 👤 Author

**Saravana Perumal**: [GitHub](https://github.com/saravanaperumalsam-blip)
