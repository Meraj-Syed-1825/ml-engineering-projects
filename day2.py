# Yesterday you worked with NumPy arrays — raw numerical matrices. Today we're adding labels and structure on top of that with Pandas.

import pandas as pd

data = {
    "Material": ["Steel", "Aluminum", "Titanium", "Copper"],
    "Yield_Strength_MPa": [250, 270, 880, 70],
    "Density_gcm3": [7.85, 2.70, 4.43, 8.96],
    "Young_Modulus_GPa": [210, 69, 114, 128]
}

df = pd.DataFrame(data) # DataFrame — exactly right! df is just the standard shorthand every data scientist uses for DataFrame.
print(df)

# NumPy → raw numbers, optimized for math and computation
# Pandas → structured table with labels, optimized for data handling

print(df.shape)
print(df.dtypes) # dtypes is telling you the data type of each column — object means text (like material names), int64 means whole numbers, float64 means decimals.

print(df.head(2)) # hows the first 2 rows. Useful when your dataset has thousands of rows and you just want a quick peek at the structure.
print(df.info()) # shows a summary: column names, how many non-null values, and data types. Great for spotting missing data.
print(df.describe()) # shows statistics for every numerical column: mean, min, max, standard deviation. First thing every data scientist runs on a new dataset.

#  In ML, df.describe() is how you get a feel for your data before building any model — are the values in a sensible range? Are there outliers?

# get one column
print(df["Yield_Strength_MPa"])

# grt one row by index
print(df.iloc[1]) # iloc stands for integer location — it selects rows by their position number (0, 1, 2...).
"""
            Think of it like NumPy indexing from yesterday — A[1] gave you row 1. df.iloc[1] does the same for a DataFrame.
            Pandas actually has two ways to select rows:

            iloc → by position (integer)
            loc → by label (name)
            """

# get a specific value
print(df["Yield_Strength_MPa"][2])


print(df.loc[1])
print(df.iloc[1])

df2 = df[df["Yield_Strength_MPa"] > 100]
print(df2)
print("---")
print(df2.iloc[0])   # first row by position
print("---")
print(df2.loc[0])    # row with label 0


# add a new column: strength to weight ratio
df["Strength_to_weight"] = df["Yield_Strength_MPa"] / df["Density_gcm3"]
print(df)

# one more new column
df["Stiffness_to_weight"] = df["Young_Modulus_GPa"] / df["Density_gcm3"]
print(df)

print(df[["Material", "Stiffness_to_weight","Strength_to_weight"]])

# You just did real Pandas work — filtering, adding computed columns, comparing materials. Let's do one last thing for today — handling missing data.

import numpy as np

df.loc[1, "Yield_strength_MPa"] = np.nan
df.loc[3, "Density_gcm3"] = np.nan

print(df)
print("\nMissing values:")
print(df.isnull().sum()) # isnull().sum() counted the missing values per column — it found 1 missing in Density_gcm3 (Copper's density).


mean_density = df["Density_gcm3"].mean( )
df["Density_gcm3"].fillna(mean_density, inplace=True)
print(df["Density_gcm3"])


"""
Copper got filled with 4.993 — the average of the other three densities (7.85 + 2.70 + 4.43 / 3).
Not perfect, but better than losing the row entirely. In real datasets you might use a smarter fill — like the average of similar materials only.

That's Day 2 done! Here's what you covered today:

Creating DataFrames from dictionaries
head(), info(), describe() for quick data exploration
iloc vs loc for row selection
Filtering rows with conditions
Adding computed columns — strength-to-weight, stiffness-to-weight
Handling missing data with fillna()

"""