import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor , plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Same steel dataset from Day 6
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

#split data
X = df.drop(columns=["Yield_Strength_MPa"])
y = df["Yield_Strength_MPa"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train decision Tree
tree = DecisionTreeRegressor(max_depth=3, random_state=42)
# max_depth=3 limits tree to 3 levels of splits — prevents overfitting
tree.fit(X_train, y_train)

print("R² Train:", round(r2_score(y_train, tree.predict(X_train)), 3))
print("R² Test:", round(r2_score(y_test, tree.predict(X_test)), 3))


plt.figure(figsize=(20,8))
plot_tree(tree,
          feature_names=X.columns,  # show actual feature names instead of numbers
          filled= True,  # color nodes by prediction value
          rounded=True,  # rounded boxes look cleaner
          fontsize=10)
plt.title("Decision Tree — Steel Yield Strength")
plt.show()


from sklearn.ensemble import RandomForestRegressor

# n_estimators=100 means 100 trees
# random_state=42 for reproducibility
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

print("R² Train:", round(r2_score(y_train, rf.predict(X_train)), 3))
print("R² Test:", round(r2_score(y_test, rf.predict(X_test)), 3))

# feature_importances_ gives a score per feature
# higher score = more important for predictions
importances = rf.feature_importances_

plt.bar(X.columns, importances, color="steelblue")
plt.title("Feature Importance — Random Forest")
plt.xlabel("Feature")
plt.ylabel("Importance Score")
plt.show()

# Try different max_depth values and record results
# max_depth controls how many levels of splits the tree can make

for depth in [1, 2, 3, 5, 10, 20, None]:
    # None means no limit — tree grows as deep as it wants
    tree = DecisionTreeRegressor(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    
    r2_train = round(r2_score(y_train, tree.predict(X_train)), 3)
    r2_test = round(r2_score(y_test, tree.predict(X_test)), 3)
    
    # f-string formats the output neatly
    print(f"max_depth={str(depth):>4}  →  Train R²: {r2_train}  |  Test R²: {r2_test}")

# n_estimators = number of trees in the forest
# more trees = more stable but slower to train

for n_trees in [1, 5, 10, 50, 100, 200]:
    rf = RandomForestRegressor(n_estimators=n_trees, random_state=42)
    rf.fit(X_train, y_train)
    
    r2_train = round(r2_score(y_train, rf.predict(X_train)), 3)
    r2_test = round(r2_score(y_test, rf.predict(X_test)), 3)
    
    print(f"n_estimators={str(n_trees):>4}  →  Train R²: {r2_train}  |  Test R²: {r2_test}")