import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from google.colab import files

uploaded = files.upload()

data = pd.read_csv("polynomial_dataset.csv")

print("Dataset:\n")
print(data)

X = data[["Experience"]]
y = data["Salary"]

linear_model = LinearRegression()
linear_model.fit(X, y)
linear_pred = linear_model.predict(X)

poly = PolynomialFeatures(degree=2)

X_poly = poly.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)
poly_pred = poly_model.predict(X_poly)

print("\nLinear Regression R2 Score:")
print(r2_score(y, linear_pred))

print("\nPolynomial Regression R2 Score:")
print(r2_score(y, poly_pred))

new_sample = [[6]]

linear_salary = linear_model.predict(new_sample)
poly_salary = poly_model.predict(poly.transform(new_sample))

print("\nLinear Regression Prediction:", linear_salary[0])
print("Polynomial Regression Prediction:", poly_salary[0])
