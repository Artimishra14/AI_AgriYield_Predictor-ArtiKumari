import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import warnings

# Try to import boosted trees, fall back to RF
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None
try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('crop_yield_dataset.csv')

# Drop date column if present and handle NA
df = df.drop(columns=['Date'], errors='ignore')
df = df.dropna(subset=['Crop_Yield', 'Crop_Type'])
for col in ['Soil_pH', 'Temperature', 'Humidity', 'Wind_Speed', 'N', 'P', 'K', 'Soil_Quality']:
    if col in df:
        df[col] = df[col].fillna(df[col].median())

num_features = ['Soil_pH', 'Temperature', 'Humidity', 'Wind_Speed', 'N', 'P', 'K', 'Soil_Quality']
cat_features = ['Crop_Type']  # You can add 'Soil_Type' here to compare
target = 'Crop_Yield'

X = df[num_features + cat_features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
ct = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
])

models = []
# Add XGBoost if available
if XGBRegressor:
    xgb_pipe = Pipeline([
        ('preprocess', ct),
        ('reg', XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=8, subsample=0.9, n_jobs=-1, random_state=42))
    ])
    models.append(('XGBoost', xgb_pipe))

# Add LightGBM if available
if LGBMRegressor:
    lgbm_pipe = Pipeline([
        ('preprocess', ct),
        ('reg', LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=8, subsample=0.9, n_jobs=-1, random_state=42))
    ])
    models.append(('LightGBM', lgbm_pipe))

# Always add Random Forest
rf_pipe = Pipeline([
    ('preprocess', ct),
    ('reg', RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1))
])
models.append(('RandomForest', rf_pipe))

best_score = -np.inf
best_mae = np.inf
best_model = None
best_name = ""

print("Comparing models ...")
for name, pipe in models:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"{name}: MAE={mae:.2f}, R2={r2:.4f}")
        if r2 > best_score or (r2 == best_score and mae < best_mae):
            best_score = r2
            best_mae = mae
            best_model = pipe
            best_name = name

print(f"\nBest model: {best_name} (MAE={best_mae:.2f}, R2={best_score:.4f})")
joblib.dump(best_model, 'crop_yield_best_model2.pkl')
print(" Model and preprocessors saved as crop_yield_best_model2.pkl")

# Demo prediction
first = X_test.iloc[[0]]
print("Sample prediction input:", first.to_dict(orient='records')[0])
print("Pred:", best_model.predict(first)[0], "Actual:", y_test.iloc[0])
