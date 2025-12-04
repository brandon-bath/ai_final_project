import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

train = pd.read_csv("train.csv", low_memory=False)
store = pd.read_csv("store.csv")

df = pd.merge(train, store, on="Store", how="left")
df = df[(df["Open"] == 1) & (df["Sales"] > 0)]
df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day

label_cols = ["StoreType", "Assortment", "StateHoliday", "PromoInterval"]
encoder_dict = {}
for col in label_cols:
    le = LabelEncoder()
    df[col] = df[col].astype(str)
    df[col] = le.fit_transform(df[col])
    encoder_dict[col] = le

feature_cols = [
    "DayOfWeek", "Promo", "SchoolHoliday",
    "CompetitionDistance", "CompetitionOpenSinceMonth",
    "CompetitionOpenSinceYear", "Promo2", "Promo2SinceWeek",
    "Promo2SinceYear",
    "Year", "Month", "Day",
    "StoreType", "Assortment", "StateHoliday", "PromoInterval"
]
df[feature_cols] = df[feature_cols].fillna(0)
X = df[feature_cols]
y = df["Sales"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

linreg = LinearRegression()
linreg.fit(X_train_scaled, y_train)
y_pred_lin = linreg.predict(X_val_scaled)
mae_lin = mean_absolute_error(y_val, y_pred_lin)
mse_lin = mean_squared_error(y_val, y_pred_lin)
rmse_lin = mse_lin ** 0.5
r2_lin = r2_score(y_val, y_pred_lin)

print("Linear Regression Results")
print(f"MAE:  {mae_lin:,.2f}")
print(f"MSE:  {mse_lin:,.2f}")
print(f"RMSE: {rmse_lin:,.2f}")
print(f"R²:   {r2_lin:.4f}")

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_val)
mae_rf = mean_absolute_error(y_val, y_pred_rf)
mse_rf = mean_squared_error(y_val, y_pred_rf)
rmse_rf = mse_rf ** 0.5
r2_rf = r2_score(y_val, y_pred_rf)

print("\nRandom Forest Results")
print(f"MAE:  {mae_rf:,.2f}")
print(f"MSE:  {mse_rf:,.2f}")
print(f"RMSE: {rmse_rf:,.2f}")
print(f"R²:   {r2_rf:.4f}")

print("\nModel Comparison")
print(f"Linear Regression - RMSE: {rmse_lin:,.2f}, MAE: {mae_lin:,.2f}, R²: {r2_lin:.4f}")
print(f"Random Forest     - RMSE: {rmse_rf:,.2f}, MAE: {mae_rf:,.2f}, R²: {r2_rf:.4f}")

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.title("Random Forest Feature Importance")
plt.bar(range(len(feature_cols)), importances[indices])
plt.xticks(range(len(feature_cols)), [feature_cols[i] for i in indices], rotation=45, ha="right")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

n_points = min(5000, len(y_val))
y_val_sample = y_val[:n_points]
y_pred_sample = y_pred_rf[:n_points]
plt.figure(figsize=(6, 6))
plt.scatter(y_val_sample, y_pred_sample, alpha=0.3)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales (Random Forest)")
max_val = max(y_val_sample.max(), y_pred_sample.max())
plt.plot([0, max_val], [0, max_val], "r--")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.close()
