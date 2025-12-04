import os
import pandas as pd
import matplotlib.pyplot as plt


def load_data(train_file="train.csv", store_file="store.csv"):
    train = pd.read_csv(train_file)
    store = pd.read_csv(store_file)
    return train, store


def merge_and_engineer(train, store):
    df = pd.merge(train, store, on="Store", how="left")
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["DayOfWeekNum"] = df["Date"].dt.dayofweek
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df = df[(df["Open"] == 1) & (df["Sales"] > 0)]
    return df


def check_missing_values(df):
    missing = df.isna().sum()
    print(missing[missing > 0])


def basic_descriptives(df):
    print(df["Sales"].describe())
    if "Customers" in df.columns:
        print(df["Customers"].describe())


def analyze_promo_effect(df):
    promo_sales = df.groupby("Promo")["Sales"].mean()
    print(promo_sales)
    plt.figure()
    promo_sales.plot(kind="bar")
    plt.title("Average Sales: Promo vs Non-Promo")
    plt.xlabel("Promo (0 = No, 1 = Yes)")
    plt.ylabel("Average Sales")
    plt.tight_layout()
    plt.savefig("promo_effect.png")
    plt.close()


def analyze_day_of_week(df):
    dow_sales = df.groupby("DayOfWeekNum")["Sales"].mean().sort_index()
    print(dow_sales)
    plt.figure()
    dow_sales.plot(kind="bar")
    plt.title("Average Sales by Day of Week")
    plt.xlabel("DayOfWeekNum (0=Mon ... 6=Sun)")
    plt.ylabel("Average Sales")
    plt.tight_layout()
    plt.savefig("day_of_week.png")
    plt.close()


def analyze_store_type(df):
    if "StoreType" not in df.columns:
        return
    storetype_sales = df.groupby("StoreType")["Sales"].mean().sort_values(ascending=False)
    print(storetype_sales)
    plt.figure()
    storetype_sales.plot(kind="bar")
    plt.title("Average Sales by StoreType")
    plt.xlabel("StoreType")
    plt.ylabel("Average Sales")
    plt.tight_layout()
    plt.savefig("store_type.png")
    plt.close()


def analyze_monthly_seasonality(df):
    month_sales = df.groupby("Month")["Sales"].mean().sort_index()
    print(month_sales)
    plt.figure()
    month_sales.plot(kind="bar")
    plt.title("Average Sales by Month")
    plt.xlabel("Month (1-12)")
    plt.ylabel("Average Sales")
    plt.tight_layout()
    plt.savefig("monthly_seasonality.png")
    plt.close()


def save_cleaned_data(df, output_file="train_merged_basic.csv"):
    df.to_csv(output_file, index=False)


def main():
    train, store = load_data()
    df = merge_and_engineer(train, store)
    check_missing_values(df)
    basic_descriptives(df)
    analyze_promo_effect(df)
    analyze_day_of_week(df)
    analyze_store_type(df)
    analyze_monthly_seasonality(df)
    save_cleaned_data(df)