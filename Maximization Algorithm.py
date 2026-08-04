import pandas as pd
from sklearn.mixture import GaussianMixture
from google.colab import files

uploaded = files.upload()

data = pd.read_csv("em_dataset.csv")

print("Dataset:\n")
print(data)

X = data.values

model = GaussianMixture(n_components=2, random_state=1)

model.fit(X)

labels = model.predict(X)

print("\nCluster Labels:")
print(labels)

print("\nMeans of Clusters:")
print(model.means_)

new_sample = [[168, 62]]

prediction = model.predict(new_sample)

print("\nPredicted Cluster for New Sample:", prediction[0])
