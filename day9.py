import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
# XGBRegressor = XGBoost for regression (predicting numbers)
# same pattern as RandomForestRegressor from yesterday

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Same steel dataset — rebuild it
np.random.seed(42)
n = 200

carbon = np.random.uniform(0.1, 0.8, n)
manganese = np.random.uniform(0.5, 2.0, n)
temp = np.random.uniform(900, 1200, n)

yield_strength = (200 +
                  300 * carbon +
                  100 * manganese -
                  0.1 * temp +
                  np.random.normal(0, 20, n))

df = pd.DataFrame({
    "Carbon_pct": carbon,
    "Manganese_pct": manganese,
    "Rolling_Temp_C": temp,
    "Yield_Strength_MPa": yield_strength
})

X = df.drop(columns=["Yield_Strength_MPa"])
y = df["Yield_Strength_MPa"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost
xgb = XGBRegressor(n_estimators=100, random_state=42)
# n_estimators=100 = 100 sequential trees
# each tree corrects errors of previous one

xgb.fit(X_train, y_train)

print("R² Train:", round(r2_score(y_train, xgb.predict(X_train)), 3))
print("R² Test:", round(r2_score(y_test, xgb.predict(X_test)), 3))


# learning_rate controls how much each tree corrects the previous one
# lower = more cautious corrections, less overfitting
# n_estimators = number of sequential trees

for lr in [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]:
    xgb = XGBRegressor(n_estimators=100, 
                       learning_rate=lr,  
                       # learning_rate = how big each correction step is
                       # 1.0 = full correction, 0.01 = tiny correction
                       random_state=42)
    xgb.fit(X_train, y_train)
    
    r2_train = round(r2_score(y_train, xgb.predict(X_train)), 3)
    r2_test = round(r2_score(y_test, xgb.predict(X_test)), 3)
    
    print(f"learning_rate={lr}  →  Train R²: {r2_train}  |  Test R²: {r2_test}")


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Random Forest — best parameters from yesterday
rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
rf_model.fit(X_train, y_train)

# XGBoost — best learning rate you just found
xgb_best = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
xgb_best.fit(X_train, y_train)

# Compare all three
models = {
    "Linear Regression": lr_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_best
}

print(f"{'Model':<20} {'Train R²':>10} {'Test R²':>10}")
print("-" * 42)
for name, model in models.items():
    r2_train = round(r2_score(y_train, model.predict(X_train)), 3)
    r2_test = round(r2_score(y_test, model.predict(X_test)), 3)
    print(f"{name:<20} {r2_train:>10} {r2_test:>10}")

# plot_importance is a built-in XGBoost visualization function
from xgboost import plot_importance

plot_importance(xgb_best, 
                importance_type="gain",
                # gain = how much each feature improves predictions
                # other options: "weight" (how often used), "cover" (how many samples affected)
                title="XGBoost Feature Importance")
plt.show()