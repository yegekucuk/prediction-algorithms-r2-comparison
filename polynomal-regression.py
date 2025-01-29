import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# - Importing dataset
x = pd.read_csv("./dataset/x.csv")
y = pd.read_csv("./dataset/y.csv")

# - Applying polynomal features
poly = PolynomialFeatures(degree=4)
x_poly = poly.fit_transform(x)

# - Creating and fitting a regression model
regressor = LinearRegression()
regressor.fit(x_poly, y)

# - Visualization
df = pd.DataFrame()
df["Original"] = y.values.flatten()
df["Prediction"] = regressor.predict(x_poly)
print(df)

# - Plotting
plt.scatter(
    df['Original'], 
    df['Prediction'], 
    alpha=0.7,
    c='blue',
    label='Predictions'
)
min_val = min(df['Original'].min(), df['Prediction'].min())
max_val = max(df['Original'].max(), df['Prediction'].max())
plt.plot([min_val, max_val], [min_val, max_val], 
         '--r', 
         label='Perfect Prediction (y=x)')
plt.xlabel('Original Values', fontsize=12)
plt.ylabel('Predicted Values', fontsize=12)
plt.title('Original vs Predicted Values', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
buffer = 0.1*(max_val - min_val)
plt.xlim(min_val - buffer, max_val + buffer)
plt.ylim(min_val - buffer, max_val + buffer)

# - Adding R2 score to plot
from sklearn.metrics import r2_score
r2 = r2_score(df['Original'], df['Prediction'])
plt.text(0.05, 0.9, f'R² = {r2:.2f}', transform=plt.gca().transAxes, fontsize=12)
plt.show()
