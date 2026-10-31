    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    print("===== PHASE 2: EXPLORATORY DATA ANALYSIS =====")

    # --------------------------------------------------
    # 1. LOAD PROCESSED DATASET
    # --------------------------------------------------

    DATA_PATH = "data/processed/rossmann_processed.csv"
    OUTPUT_PATH = "outputs/plots"

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    print("\nLoading processed dataset...")

    df = pd.read_csv(DATA_PATH)

    # Convert Date back to datetime
    df["Date"] = pd.to_datetime(df["Date"])

    print("\nDataset Shape:", df.shape)

    print("\nDataset Columns:")
    print(df.columns.tolist())

    # --------------------------------------------------
    # 2. BASIC SALES STATISTICS
    # --------------------------------------------------

    print("\n===== SALES STATISTICS =====")

    print(df["Sales"].describe())

    # --------------------------------------------------
    # 3. DAILY SALES TREND
    # --------------------------------------------------

    print("\nCreating daily sales trend...")

    daily_sales = df.groupby("Date")["Sales"].sum()

    plt.figure(figsize=(12, 6))
    plt.plot(daily_sales.index, daily_sales.values)

    plt.title("Daily Sales Trend")
    plt.xlabel("Date")
    plt.ylabel("Total Sales")
    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_PATH}/daily_sales_trend.png"
    )

    plt.close()

    # --------------------------------------------------
    # 4. MONTHLY SALES ANALYSIS
    # --------------------------------------------------

    print("Creating monthly sales analysis...")

    monthly_sales = df.groupby(
        ["Year", "Month"]
    )["Sales"].sum().reset_index()

    monthly_sales["YearMonth"] = (
        monthly_sales["Year"].astype(str)
        + "-"
        + monthly_sales["Month"].astype(str)
    )

    plt.figure(figsize=(12, 6))

    sns.barplot(
        data=monthly_sales,
        x="YearMonth",
        y="Sales"
    )

    plt.title("Monthly Sales Trend")
    plt.xlabel("Year-Month")
    plt.ylabel("Total Sales")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_PATH}/monthly_sales_trend.png"
    )

    plt.close()

    # --------------------------------------------------
    # 5. PROMOTION IMPACT
    # --------------------------------------------------

    print("Creating promotion impact analysis...")

    promo_sales = df.groupby(
        "Promo"
    )["Sales"].mean().reset_index()

    plt.figure(figsize=(7, 5))

    sns.barplot(
        data=promo_sales,
        x="Promo",
        y="Sales"
    )

    plt.title("Average Sales: Promotion vs No Promotion")
    plt.xlabel("Promotion (0 = No, 1 = Yes)")
    plt.ylabel("Average Sales")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_PATH}/promotion_impact.png"
    )

    plt.close()

    # --------------------------------------------------
    # 6. SALES BY STORE TYPE
    # --------------------------------------------------

    print("Creating store type analysis...")

    store_sales = df.groupby(
        "StoreType"
    )["Sales"].mean().reset_index()

    plt.figure(figsize=(7, 5))

    sns.barplot(
        data=store_sales,
        x="StoreType",
        y="Sales"
    )

    plt.title("Average Sales by Store Type")
    plt.xlabel("Store Type")
    plt.ylabel("Average Sales")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_PATH}/sales_by_store_type.png"
    )

    plt.close()

    # --------------------------------------------------
    # 7. SALES BY DAY OF WEEK
    # --------------------------------------------------

    print("Creating day-of-week analysis...")

    day_sales = df.groupby(
        "DayOfWeek"
    )["Sales"].mean().reset_index()

    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=day_sales,
        x="DayOfWeek",
        y="Sales"
    )

    plt.title("Average Sales by Day of Week")
    plt.xlabel("Day of Week")
    plt.ylabel("Average Sales")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_PATH}/sales_by_day.png"
    )

    plt.close()

    # --------------------------------------------------
    # 8. FINAL MESSAGE
    # --------------------------------------------------

    print("\n===== EDA COMPLETED SUCCESSFULLY =====")

    print("\nGenerated plots:")

    plots = [
        "daily_sales_trend.png",
        "monthly_sales_trend.png",
        "promotion_impact.png",
        "sales_by_store_type.png",
        "sales_by_day.png"
    ]

    for plot in plots:
        print("-", plot)