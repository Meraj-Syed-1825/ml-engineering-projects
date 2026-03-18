"""
Welcome to Day 11! 🚀
Before we touch any code — PCA is one of those concepts that makes much more sense if you understand the why before the how.
Think about your steel dataset. You have 3 features: Carbon, Manganese, Temperature. But imagine a real materials database with 50 features — composition of 20 elements, 10 process parameters, 20 microstructural measurements.
Two problems arise with 50 features:

Many features are probably correlated — remember your Day 3 heatmap? Correlated features give the model redundant information
Training becomes slow and overfitting risk increases — too many inputs, not enough data

PCA solves this by compressing many features into fewer, while keeping most of the information.
Here's a question to build intuition before coding:
From your Day 3 heatmap, Yield Strength and Strength-to-Weight were highly correlated. If you already know Yield Strength, how much new information does Strength-to-Weight actually add?
"""

"""
Strength_to_Weight = Yield_Strength / Density

Exactly! So Strength-to-Weight adds zero new information if you already have Yield Strength and Density. It's completely derived from them.
That's the core idea behind PCA — if features are correlated or derivable from each other, they're redundant. PCA finds a smarter set of combined features that captures the same information in fewer dimensions.
Let's make this visual first. Create day11.py and run this:
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# StandardScaler = rescales all features to same range
# CRITICAL for PCA — explained below

from sklearn.decomposition import PCA
# PCA = Principal Component Analysis — the dimensionality reduction tool

# Rebuild steel dataset — but with MORE features this time
np.random.seed(42)
n = 200

carbon = np.random.uniform(0.1, 0.8, n)
manganese = np.random.uniform(0.5, 2.0, n)
temp = np.random.uniform(900, 1200, n)
silicon = np.random.uniform(0.1, 0.5, n)      # new feature
chromium = np.random.uniform(0.0, 0.3, n)     # new feature

yield_strength = (200 + 300*carbon + 100*manganese - 
                  0.1*temp + 50*silicon + 
                  np.random.normal(0, 20, n))

df = pd.DataFrame({
    "Carbon_pct": carbon,
    "Manganese_pct": manganese,
    "Rolling_Temp_C": temp,
    "Silicon_pct": silicon,
    "Chromium_pct": chromium,
    "Yield_Strength_MPa": yield_strength
})

X = df.drop(columns=["Yield_Strength_MPa"])
y = df["Yield_Strength_MPa"]

print(df.shape)
print(df.head())

"""
Run it — we now have 5 input features instead of 3. Before applying PCA, why do you think we need to rescale all features to the same range first?

Think about your 5 features and their ranges:

Carbon: 0.1 to 0.8 (range of ~0.7)
Manganese: 0.5 to 2.0 (range of ~1.5)
Temperature: 900 to 1200 (range of ~300)

PCA works by finding directions of maximum variation in your data. 
If Temperature has values in the hundreds and Carbon has values below 1 — which feature will PCA think is most important, just based on the numbers?

Exactly! Temperature would dominate purely because its numbers are 1000x larger — not because it's actually more important.
That's why we standardize first — rescale every feature to have mean=0 and standard deviation=1. Then PCA judges importance fairly based on actual variation, not units.
Run this:
"""

# Step 1: Standardize — rescale all features to same scale
scaler = StandardScaler()
# StandardScaler subtracts mean and divides by std for each column
# result: every feature has mean=0, std=1

X_scaled = scaler.fit_transform(X)
# fit_transform = learn the mean/std from data, then apply the scaling
# result is a numpy array (not DataFrame)

print("Original Temperature range:", X["Rolling_Temp_C"].min().round(1), "to", X["Rolling_Temp_C"].max().round(1))
print("Scaled Temperature range:", X_scaled[:, 2].min().round(2), "to", X_scaled[:, 2].max().round(2))

print("\nOriginal Carbon range:", X["Carbon_pct"].min().round(2), "to", X["Carbon_pct"].max().round(2))
print("Scaled Carbon range:", X_scaled[:, 0].min().round(2), "to", X_scaled[:, 0].max().round(2))

"""
Both features now have the same range — roughly -1.7 to +1.7. Temperature went from 900-1200 down to that same small range as Carbon.
Now PCA can fairly compare them. Let's apply PCA:
"""

# Step 2: Apply PCA
pca = PCA()
# PCA() with no arguments = keep all components first
# we'll decide how many to keep after seeing the results

X_pca = pca.fit_transform(X_scaled)
# fit_transform = learn the principal components, then project data onto them
# result: same shape as input (200 x 5) but in new coordinate system

# explained_variance_ratio_ = how much of total information each component captures
print("Variance explained per component:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"  Component {i+1}: {var:.1%}")

print(f"\nTotal with 2 components: {pca.explained_variance_ratio_[:2].sum():.1%}")
print(f"Total with 3 components: {pca.explained_variance_ratio_[:3].sum():.1%}")

"""
Interesting result! The variance is spread almost equally across all 5 components — each captures roughly 20%.

What does that tell you about the relationships between your 5 features? Think back to the Day 3 heatmap concept.

one is higher and lower at some points but is mostly same

You're noticing the right thing! Let me make it sharper.
When PCA finds that variance is evenly spread across all components, it means none of the components can be dropped without losing significant information.
Compare this to what would happen if Carbon and Manganese were highly correlated — say Carbon = 2 × Manganese always. How many independent pieces of information would you actually have from those two features?
"""

"""
Just one piece of information — they move together so they're really one variable pretending to be two!
That's exactly the point. When features are independent (not correlated), PCA can't compress them — every component carries unique information. Our 5 features happen to be mostly independent of each other.
Let's visualize this with a scree plot — the standard way to decide how many components to keep:
"""

plt.figure(figsize=(8, 5))

# cumulative sum adds up variance step by step
# [0.225, 0.217, 0.206...] → [0.225, 0.442, 0.648...]
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

plt.plot(range(1, 6), cumulative_variance, 
         marker="o",        # circle marker at each point
         color="steelblue",
         linewidth=2)

# draw a horizontal line at 80% — common threshold in practice
plt.axhline(y=0.8, color="red", linestyle="--", label="80% threshold")

plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance Explained")
plt.title("Scree Plot — How Many Components to Keep?")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

"""
You need 4 components to cross the 80% threshold — capturing 84% of the information.
This confirms what the numbers told us — because our 5 features are mostly independent, you can't compress them much without losing significant information.

Now let's see what happens when you actually use PCA-reduced features to train a model — does reducing from 5 to 4 components hurt performance?
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

# Compare: full 5 features vs PCA-reduced 4 components
results = {}

# Model 1: all 5 original features
rf_full = RandomForestRegressor(n_estimators=50, random_state=42)
scores_full = cross_val_score(rf_full, X_scaled, y, cv=5, scoring="r2")
results["5 features (original)"] = scores_full.mean()

# Model 2: PCA reduced to 4 components (captures 84% variance)
pca_4 = PCA(n_components=4)
# n_components=4 = keep only first 4 components
X_pca_4 = pca_4.fit_transform(X_scaled)
# fit_transform = learn components from data, project onto 4 dimensions

rf_pca4 = RandomForestRegressor(n_estimators=50, random_state=42)
scores_pca4 = cross_val_score(rf_pca4, X_pca_4, y, cv=5, scoring="r2")
results["4 components (PCA)"] = scores_pca4.mean()

# Model 3: PCA reduced to 2 components (captures only 44% variance)
pca_2 = PCA(n_components=2)
X_pca_2 = pca_2.fit_transform(X_scaled)

rf_pca2 = RandomForestRegressor(n_estimators=50, random_state=42)
scores_pca2 = cross_val_score(rf_pca2, X_pca_2, y, cv=5, scoring="r2")
results["2 components (PCA)"] = scores_pca2.mean()

print(f"{'Setup':<25} {'Mean R²':>10}")
print("-" * 37)
for name, score in results.items():
    print(f"{name:<25} {score.round(3):>10}")

"""
Big drop! From 0.902 → 0.748 with 4 components, and 0.173 with just 2.
This perfectly confirms what the scree plot told you. You said it yourself earlier — because the 5 features are mostly independent, dropping any of them loses real information.
This is actually the most important lesson about PCA:

PCA is powerful when features are correlated. It's useless when features are independent.

In your steel dataset, Carbon, Manganese, Temperature, Silicon, Chromium were all generated independently — so PCA can't help.
But think about a real materials database where you measure:

Tensile strength
Yield strength
Ultimate strength
Elongation at break

These are all related to the same underlying property — ductility vs brittleness. 
PCA would compress them beautifully into 1-2 components without losing much.
"""