import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
 # cross_val_score = runs cross-validation automatically
# it handles all the splitting, training, and testing for you

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRFRegressor
from xgboost import XGBRegressor

# Rebuilt steel dataset
np.random.seed(42)
n = 200
carbon = np.random.uniform(0.1, 0.8, n)
manganese = np.random.uniform(0.5, 2.0, n)
temp = np.random.uniform(900, 1200, n)
yield_strength = (200 + 300*carbon + 100*manganese - 0.1*temp + np.random.normal(0, 20, n))

df = pd.DataFrame({
    "Carbon_pct": carbon,
    "Manganese_pct": manganese,
    "Rolling_Temp_C": temp,
    "Yield_Strength_MPa": yield_strength
})

X = df.drop(columns=["Yield_Strength_MPa"])
y = df["Yield_Strength_MPa"]

# Run 5-fold cross-validation on Linear Regression
scores = cross_val_score(LinearRegression(), X, y,
                         cv=5,        # cv=5 means 5 folds
                         scoring="r2" # evaluate using R² score
                         )

print("R² per fold:", scores.round(3))
print("Mean R²:", scores.mean().round(3))
print("Std R²:", scores.std().round(3))
# std = standard deviation — how consistent is the model across folds?
# low std = stable model. high std = model is sensitive to which data it sees

# Now let's compare all three models using cross-validation:

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
}

print(f"{'Model':<20} {'Mean R²':>10} {'Std R²':>10}")
print("-" * 42)

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    # scores = array of 5 R² values, one per fold
    print(f"{name:<20} {scores.mean().round(3):>10} {scores.std().round(3):>10}")

"""
Two interesting things here. Linear Regression still wins on mean R² — you already know why from Day 9 (linear data favors linear model).
But look at the Std column. Random Forest has the lowest std (0.005) — most stable across all 5 folds.
Now here's a question — in a real engineering application like predicting springback for a safety-critical component, which would you trust more: a model with slightly higher mean R² but higher variance, or a model with slightly lower mean R² but very consistent across all folds?

model with slightly lower mean R² but very consistent across all folds

Exactly right — and that's excellent engineering judgment!
In safety-critical applications you'd rather have predictable, consistent performance than occasionally higher but unreliable results. A model that scores 0.887 every single time is more trustworthy than one that scores 0.928 sometimes but 0.870 other times.
This is why cross-validation is so valuable — a single split would have hidden this stability difference completely.
Now let's add one more powerful concept — visualizing the cross-validation scores so you can see the spread:
"""

import matplotlib.pyplot as plt

# Run CV for all three models and store scores
results = {}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    # store all 5 fold scores per model
    results[name] = scores

# Boxplot — shows distribution of scores across 5 folds
# box = middle 50% of scores
# line inside box = median
# whiskers = full range
plt.boxplot(results.values(),        # the 5 scores per model
            labels=results.keys(),   # model names on x-axis
            patch_artist=True)       # filled boxes instead of just outlines

plt.title("Cross-Validation R² — 5 Folds")
plt.ylabel("R² Score")
plt.ylim(0.8, 1.0)   # zoom in — all scores are above 0.8
plt.grid(axis="y", alpha=0.3)  # horizontal grid lines, semi-transparent
plt.show()

"""
Great plot! Look at the three boxes carefully.
Random Forest has the smallest box — most consistent. XGBoost has the widest spread.
Notice something else on the Linear Regression box — that small circle above it at 0.95. That's called an outlier in the boxplot. It means one fold scored much higher than the others.
What does that outlier tell you about Linear Regression's reliability across different data splits?
"""

"""
The outlier means one specific fold (not material) scored unusually high — meaning when that particular 20% happened to be the test set, the model performed exceptionally well.
This tells you Linear Regression got lucky on one split. If you had done only a single 80/20 split and happened to pick that lucky split, you'd think your model is better than it really is.
That's exactly why cross-validation exists — it exposes this kind of luck.

Now let's do one final and very practical thing for Day 10 — finding the best parameters automatically using cross-validation. This is called GridSearchCV.
Instead of manually trying max_depth = 1, 2, 3, 5... like you did on Day 8, GridSearchCV tries all combinations automatically and picks the best one:

"""

from sklearn.model_selection import GridSearchCV
# GridSearchCV = automatically tries all parameter combinations
# uses cross-validation to evaluate each combination fairly

param_grid = {
    "max_depth": [3, 5, 7, 10],       # 4 options for depth
    "n_estimators": [50, 100, 200]    # 3 options for trees
}
# total combinations = 4 x 3 = 12 models will be tested

rf = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(rf,           # model to tune
                           param_grid,   # parameters to try
                           cv=5,         # 5-fold CV for each combination
                           scoring="r2", # evaluate by R²
                           n_jobs=-1)    # use all CPU cores — speeds it up
grid_search.fit(X, y)

print("Best parameters:", grid_search.best_params_)
print("Best CV R²:", round(grid_search.best_score_, 3))

"""
GridSearchCV confirmed exactly what you found manually — max_depth=5 is the sweet spot!
This is a great validation. You discovered it yourself by experimentation on Day 8, and now an automated tool agrees with you. That's good scientific instinct.
The real power of GridSearchCV is when you have many parameters — imagine testing 5 values for depth, 5 for n_estimators, and 3 for another parameter. That's 75 combinations — impossible to do manually but GridSearchCV handles it in seconds.

Day 10 complete! Here's what you learned:
Concept
Key takeaway
Cross-validation Test on every sample exactly once — no lucky splits
k-fold Split data into k groups, rotate test set k times
Mean R² Average performance across all folds — more reliable than single split
Std R² How consistent the model is — low std = trustworthy
GridSearchCV Automatically finds best parameters using cross-validation
Boxplot outlier Revealed Linear Regression got lucky on one split
"""