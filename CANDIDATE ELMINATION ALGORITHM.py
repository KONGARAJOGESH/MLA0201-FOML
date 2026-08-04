import pandas as pd
import numpy as np
from google.colab import files

uploaded = files.upload()

data = pd.read_csv("training_data.csv")

concepts = np.array(data.iloc[:, :-1])
target = np.array(data.iloc[:, -1])

def learn(concepts, target):
    specific_h = concepts[0].copy()
    general_h = [["?" for i in range(len(specific_h))] for i in range(len(specific_h))]

    for i, h in enumerate(concepts):
        if target[i] == "Yes":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x] = "?"
                    general_h[x][x] = "?"
        else:
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = "?"

    general_h = [g for g in general_h if g != ["?"] * len(specific_h)]

    return specific_h, general_h

s_final, g_final = learn(concepts, target)

print("Training Data:\n")
print(data)

print("\nFinal Specific Hypothesis:")
print(s_final)

print("\nFinal General Hypothesis:")
for g in g_final:
    print(g)
