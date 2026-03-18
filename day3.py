# Day - 3
# Today you're going to visualize the materials data you built over the last two days. This is where your data starts telling a story.

# why do you think visualization matters in ML? What could you learn from a plot that you can't easily see in a table of numbers?

# Interpret results — a scatter plot shows relationships between variables instantly that would take minutes to spot in a table
# Spot anomalies — outliers, weird distributions, missing patterns jump out visually

#In ML this is called Exploratory Data Analysis (EDA) — and every experienced data scientist does it before touching any model. You never want to train a model on data you don't visually understand.

import pandas as pd
import matplotlib.pyplot as plt

data = {
    "Material": ["Steel", "Aluminum", "Titanium", "Copper"],
    "Yield_Strength_MPa": [250,270,880,70],
    "Density_gcm3": [7.85, 2.70, 4.43, 8.96],
    "Youngs_Modulus_GPa": [210, 69, 114, 128]
}

df = pd.DataFrame(data)
df["Strength_to_weight"] = df["Yield_Strength_MPa"] / df["Density_gcm3"]

# First plot

plt.bar(df["Material"], df["Yield_Strength_MPa"], color="steelblue")
plt.title("Yield Strength by Material")
plt.xlabel("Material")
plt.ylabel("Yield Strength (MPa)")
plt.show()

colors = ["steelblue", "silver", "gray", "orange"]
plt.bar(df["Material"], df["Yield_Strength_MPa"], color=colors)
plt.title("Yield Strength by Material")
plt.xlabel("Material")
plt.ylabel("Yield Strength (MPa)")
plt.show()

# You're describing exactly the limitation — a bar chart shows one variable at a time but can't show relationships between two variables.
# For example — you can't see from that chart whether denser materials tend to be stiffer, or whether strength and modulus are correlated.
# That's where a scatter plot comes in. Try this:

plt.scatter(df["Density_gcm3"], df["Yield_Strength_MPa"], color= "steelblue", s=100)
# add materials to each scatter point
for i, row in df.iterrows():
    plt.annotate(row["Material"], (row["Density_gcm3"], row["Yield_Strength_MPa"]))

plt.title("Yield Strength vs Density")
plt.xlabel("Density (g/cm3)")
plt.ylabel("Yield STrength (MPa)")
plt.show()

# adding third variable

plt.scatter(df["Density_gcm3"], df["Yield_Strength_MPa"], 
            s=df["Youngs_Modulus_GPa"]*3,   # size = modulus
            color="steelblue", alpha=0.6)

for i, row in df.iterrows():
    plt.annotate(row["Material"], (row["Density_gcm3"],row["Yield_Strength_MPa"]))

plt.title("Yield Strength vs Density (size = Young's Modulus)")
plt.xlabel("Density (g/cm³)")
plt.ylabel("Yield STrength (MPa)")
plt.show()

# a correlation heatmap.

import numpy as np
# only numericla colums
corr = df[["Yield_Strength_MPa", "Density_gcm3", "Youngs_Modulus_GPa", "Strength_to_weight"]].corr()

fig, ax = plt.subplots()
im = ax.imshow(corr, cmap="coolwarm")

ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticklabels(corr.columns)

plt.colorbar(im)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

"""
the heatmap is actually showing correlation, not strength-to-weight directly.
The color scale goes from -1 to +1:

Red → two variables move together (when one goes up, the other goes up)
Blue → two variables move oppositely (when one goes up, the other goes down)
White/neutral → no relationship

Strength-to-Weight = Yield Strength / Density

"""

"""
Day 3 done! Here's what you covered:

Bar charts for single variable comparison
Scatter plots for relationships between two variables
Bubble charts for three variables in one plot
Correlation heatmaps for spotting relationships across all variables

"""