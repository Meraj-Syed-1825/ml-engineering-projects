"""
First, quick review. Without looking at your code, can you answer these:

1. What's the difference between * and @ in NumPy?
2. What does df.iloc[0] do?
3. What's the difference between regression and classification?
4. What does R² = 0.95 mean in plain language?
5. Why do we split data into train and test sets?
"""

"""
1. * is for multiplication of elemets, @ for matrix multiplication
2. it says about the position in length
3. Regression vs Classification — partially right. Better way to think about it:

    Regression → predicts a number (yield strength = 350 MPa)
    Classification → predicts a category (this part is defective / not defective)

4. R² = 0.95 — not quite. It means the model explains 95% of the variation in the data. 1.0 is perfect, 0.0 means the model learned nothing. 
0.95 is actually a very good model!
5. Train/test split — this is an important one to correct. It's not about computing power. 
The real reason is: we need unseen data to honestly evaluate how the model performs on new inputs it has never seen before. 
Otherwise we'd just be testing if the model memorized the training data. 
"""

"""
Now the write-up. Open a Word document or a text file and write a 1-page project summary using this structure:
1. Problem Statement (2 sentences)
   → What were you trying to predict and why?

2. Data (2 sentences)
   → What data did you use? How many samples, what features?

3. Method (3 sentences)
   → What did you do step by step?

4. Results (2 sentences)
   → What R² did you get? What did the model learn?

5. Conclusion (2 sentences)
   → What does this mean? What would you do next?


   The problem was to predeict the yield strength of steel, we had density but it was not enough to predict, so taking this into account can save testing period. over 200 samples we considered 3 main driving factors carbon, manganese, and rolling temp. with these we made a scatter plot to predict how these factors are really influencing and tained a regression model to showcase this. this data was compared to r2 and model was very close to carbon real, them to conclude sucessfully predictied yield strength


   Two small improvements for when you use this in a portfolio or interview:
Be specific with numbers — instead of "very close to carbon real", say "model learned carbon coefficient of 295 vs actual 300 — within 2% error". Numbers make engineers sound credible.
Add the R² score explicitly — "R² = 0.953 on unseen test data" is a concrete result that speaks for itself.

Week 1 is complete! 🎉
Here's what you covered in 7 days:

NumPy — arrays, matrix math, slicing
Pandas — DataFrames, filtering, computed columns, missing data
Matplotlib — bar charts, scatter plots, heatmaps
Linear Regression — train/test split, R² score
Classification — logistic regression, confusion matrix, precision, recall
Full ML pipeline on a steel manufacturing dataset
"""

"""
Here's the complete Week 1 code, all in one place with every line explained:Ran 2 commands, viewed a file, created a fileRan 2 commands, viewed a file, created a fileYour complete Week 1 code reference is ready! Here's what's inside:
7 chapters, one per day, each containing every line of code with inline comments explaining exactly what it does and why — plus:

📖 Blue explanation boxes — what each code block does in plain English
💡 Green key concept boxes — the most important ideas to remember (float precision, iloc vs loc, fit vs predict, etc.)
Quick reference table at the end — all 16 most-used functions in one place

"""
