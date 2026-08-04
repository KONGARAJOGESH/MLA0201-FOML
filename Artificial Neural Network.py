import pandas as pd
from sklearn.neural_network import MLPClassifier
from google.colab import files

uploaded = files.upload()

data = pd.read_csv("ann_dataset.csv")

print("Training Dataset:\n")
print(data)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

model = MLPClassifier(hidden_layer_sizes=(4,),
                      activation='logistic',
                      solver='sgd',
                      learning_rate_init=0.5,
                      max_iter=5000,
                      random_state=1)

model.fit(X, y)

print("\nTraining Completed")

predictions = model.predict(X)

print("\nActual Output:")
print(list(y))

print("\nPredicted Output:")
print(list(predictions))

new_sample = [[1, 0]]
prediction = model.predict(new_sample)

print("\nPrediction for New Sample [1,0]:", prediction[0])
