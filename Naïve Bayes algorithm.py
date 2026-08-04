import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from google.colab import files

uploaded = files.upload()

data = pd.read_csv("naive_bayes_dataset.csv")

print("Dataset:\n")
print(data)

encoder = LabelEncoder()

for column in data.columns:
    data[column] = encoder.fit_transform(data[column])

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

model = GaussianNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nPredicted Values:")
print(y_pred)

print("\nActual Values:")
print(y_test.values)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred) * 100)
