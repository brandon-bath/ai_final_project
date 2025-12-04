import pandas as pd
from sklearn.ensemble import RandomForestRegressor

train = pd.read_csv("train.csv", low_memory=False)
store = pd.read_csv("store.csv")
test = pd.read_csv("test.csv", low_memory=False)

train_merged = train.merge(store, on="Store")
test_merged = test.merge(store, on="Store")

train_merged = train_merged[(train_merged["Open"] == 1) & (train_merged["Sales"] > 0)]

train_merged["Date"] = pd.to_datetime(train_merged["Date"])
test_merged["Date"] = pd.to_datetime(test_merged["Date"])

train_merged["Year"] = train_merged["Date"].dt.year
train_merged["Month"] = train_merged["Date"].dt.month
train_merged["Day"] = train_merged["Date"].dt.day

test_merged["Year"] = test_merged["Date"].dt.year
test_merged["Month"] = test_merged["Date"].dt.month
test_merged["Day"] = test_merged["Date"].dt.day

categorical_cols = ["StoreType", "Assortment", "StateHoliday", "PromoInterval"]

combined = pd.concat(
    [train_merged[categorical_cols], test_merged[categorical_cols]],
    axis=0
)

for col in categorical_cols:
    combined[col] = combined[col].astype("category")
    codes = combined[col].cat.codes
    train_merged[col] = codes.iloc[: len(train_merged)].values
    test_merged[col] = codes.iloc[len(train_merged):].values

feature_cols = [
    "DayOfWeek", "Promo", "SchoolHoliday",
    "CompetitionDistance", "CompetitionOpenSinceMonth",
    "CompetitionOpenSinceYear", "Promo2", "Promo2SinceWeek",
    "Promo2SinceYear", "Year", "Month", "Day",
    "StoreType", "Assortment", "StateHoliday", "PromoInterval"
]

X_train = train_merged[feature_cols]
y_train = train_merged["Sales"]

X_test = test_merged[feature_cols]

rf = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
test_pred = rf.predict(X_test)

pred_df = pd.DataFrame({
    "Id": test["Id"],
    "Sales_Prediction": test_pred
})

pred_df.to_csv("predictions.csv", index=False)
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv("train.csv", low_memory=False)
store = pd.read_csv("store.csv")
test = pd.read_csv("test.csv", low_memory=False)

train = train.merge(store, on="Store", how="left")
test = test.merge(store, on="Store", how="left")

train = train[(train["Open"] == 1) & (train["Sales"] > 0)]

train["Date"] = pd.to_datetime(train["Date"])
test["Date"] = pd.to_datetime(test["Date"])

for df in [train, test]:
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day

cat_cols = ["StoreType", "Assortment", "StateHoliday", "PromoInterval"]

for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

feature_cols = [
    "DayOfWeek", "Promo", "SchoolHoliday",
    "CompetitionDistance", "CompetitionOpenSinceMonth",
    "CompetitionOpenSinceYear", "Promo2", "Promo2SinceWeek",
    "Promo2SinceYear", "Year", "Month", "Day",
    "StoreType", "Assortment", "StateHoliday", "PromoInterval"
]

train[feature_cols] = train[feature_cols].fillna(0)
test[feature_cols] = test[feature_cols].fillna(0)

X = train[feature_cols]
y = train["Sales"]
X_test = test[feature_cols]

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X, y)

test_preds = model.predict(X_test)

output = pd.DataFrame({
    "Id": test["Id"],
    "Sales": test_preds
})

output.to_csv("predictions.csv", index=False)
