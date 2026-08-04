import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
from google.colab import files

uploaded = files.upload()

data = pd.read_csv("play_tennis.csv")

print("Training Dataset:\n")
print(data)

le = LabelEncoder()

X = data.iloc[:, :-1].apply(le.fit_transform)
y = LabelEncoder().fit_transform(data.iloc[:, -1])

model = DecisionTreeClassifier(criterion="entropy")
model.fit(X, y)

print("\nDecision Tree Rules:\n")
print(export_text(model, feature_names=list(data.columns[:-1])))

new_sample = pd.DataFrame({
    "Outlook": ["Sunny"],
    "Temperature": ["Cool"],
    "Humidity": ["High"],
    "Wind": ["Strong"]
})

new_encoded = pd.DataFrame()

for col in new_sample.columns:
    encoder = LabelEncoder()
    encoder.fit(data[col])
    new_encoded[col] = encoder.transform(new_sample[col])

prediction = model.predict(new_encoded)

if prediction[0] == 1:
    print("\nPrediction for New Sample: Yes")
else:
    print("\nPrediction for New Sample: No")
