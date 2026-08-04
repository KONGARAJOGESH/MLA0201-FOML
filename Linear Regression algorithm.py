import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from google.colab import files

uploaded = files.upload()

data = pd.read_csv("linear_regression_dataset.csv")

print("Dataset:\n")
print(data)

X = data[["StudyHours"]]
y = data["Marks"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nPredicted Marks:")
print(y_pred)

print("\nActual Marks:")
print(y_test.values)

print("\nMean Squared Error:")
print(mean_squared_error(y_test, y_pred))

print("\nR2 Score:")
print(r2_score(y_test, y_pred))

new_sample = [[6.5]]

prediction = model.predict(new_sample)

print("\nPredicted Marks for StudyHours = 6.5:")
print(prediction[0])
