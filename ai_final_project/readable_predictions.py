import pandas as pd

test = pd.read_csv("test.csv", low_memory=False)
preds = pd.read_csv("predictions.csv")

merged = test[["Id", "Store", "Date"]].merge(preds, on="Id", how="left")

merged.to_csv("predictions_readable.csv", index=False)

print("predictions_readable.csv created.")
