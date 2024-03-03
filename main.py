import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

# Reading csv data from Codecademy content
df = pd.read_csv("https://content.codecademy.com/programs/data-science-path/linear_regression/honeyproduction.csv")

# Creating X with unique years and calculate mean honey production for each year
X = np.array(df['year'].unique()).reshape(-1, 1)
y = df.groupby('year')['totalprod'].mean().values

# Plotting the scatter plot
plt.scatter(X, y)

my_regression_model = linear_model.LinearRegression()
my_regression_model.fit(X, y)

# Predicting values using the fitted model
y_pred = my_regression_model.predict(X)

# Plotting the regression line
plt.plot(X, y_pred, color='red')
plt.xlabel('Year')
plt.ylabel('Average Honey Production')
plt.title('Linear Regression of Honey Production Over Years')
plt.show()
