import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from google.colab import files

uploaded = files.upload()

data = pd.read_csv("knn_dataset.csv")

print("Training Dataset:\n")
print(data)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

model = KNeighborsClassifier(n_neighbors=3)

model.fit(X, y)

new_sample = [[168, 62]]

prediction = model.predict(new_sample)

print("\nPrediction for New Sample (Height=168, Weight=62):")
print(prediction[0])
