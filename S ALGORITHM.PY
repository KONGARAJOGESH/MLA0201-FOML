import pandas as pd
from google.colab import files

uploaded = files.upload()

data = pd.read_csv("training_data.csv")

hypothesis = ['0'] * (len(data.columns) - 1)

for i in range(len(data)):
    if data.iloc[i, -1] == "Yes":
        for j in range(len(hypothesis)):
            if hypothesis[j] == '0':
                hypothesis[j] = data.iloc[i, j]
            elif hypothesis[j] != data.iloc[i, j]:
                hypothesis[j] = '?'

print("Training Dataset:\n")
print(data)

print("\nFinal Most Specific Hypothesis:")
print(hypothesis)
