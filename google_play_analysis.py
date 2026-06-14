# ==================================================
# Google Play Store Data Analysis
# Part 1: Import Libraries, Load Data & Cleaning
# ==================================================

# Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------------------------
# 1. Load Datasets
# --------------------------------------------------

apps_df = pd.read_csv("apps.csv")
reviews_df = pd.read_csv("user_reviews.csv")

print("Datasets Loaded Successfully!")

print("\nGoogle Play Store Dataset Shape:")
print(apps_df.shape)

print("\nFirst 5 Rows:")
print(apps_df.head())


# --------------------------------------------------
# 2. Dataset Information
# --------------------------------------------------

print("\nDataset Information:")
print(apps_df.info())


print("\nMissing Values:")
print(apps_df.isnull().sum())


# --------------------------------------------------
# 3. Remove Duplicate Apps
# --------------------------------------------------

print("\nDuplicate Records:", apps_df.duplicated().sum())

apps_df.drop_duplicates(inplace=True)

print("Duplicates Removed Successfully!")


# --------------------------------------------------
# 4. Cleaning Rating Column
# --------------------------------------------------

apps_df["Rating"] = pd.to_numeric(
    apps_df["Rating"],
    errors="coerce"
)

apps_df["Rating"].fillna(
    apps_df["Rating"].median(),
    inplace=True
)


# --------------------------------------------------
# 5. Cleaning Reviews Column
# --------------------------------------------------

apps_df["Reviews"] = pd.to_numeric(
    apps_df["Reviews"],
    errors="coerce"
)


# --------------------------------------------------
# 6. Cleaning Installs Column
# --------------------------------------------------

apps_df["Installs"] = (
    apps_df["Installs"]
    .astype(str)
    .str.replace(",", "")
    .str.replace("+", "", regex=False)
)

apps_df["Installs"] = pd.to_numeric(
    apps_df["Installs"],
    errors="coerce"
)


# --------------------------------------------------
# 7. Cleaning Price Column
# --------------------------------------------------

apps_df["Price"] = (
    apps_df["Price"]
    .astype(str)
    .str.replace("$", "", regex=False)
)

apps_df["Price"] = pd.to_numeric(
    apps_df["Price"],
    errors="coerce"
)


print("\nData Cleaning Completed Successfully!")

print("\nFinal Data Types:")
print(apps_df.dtypes)
# ==================================================
# Part 2: Category Analysis & Visualizations
# ==================================================


# --------------------------------------------------
# 8. App Category Distribution
# --------------------------------------------------

plt.figure(figsize=(12, 6))

category_count = apps_df["Category"].value_counts()

sns.barplot(
    x=category_count.index,
    y=category_count.values
)

plt.title("Number of Apps in Each Category")
plt.xlabel("Category")
plt.ylabel("Number of Apps")
plt.xticks(rotation=90)

plt.show()


# --------------------------------------------------
# 9. Rating Distribution
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    apps_df["Rating"],
    bins=20,
    kde=True
)

plt.title("Distribution of App Ratings")
plt.xlabel("Rating")
plt.ylabel("Count")

plt.show()


# --------------------------------------------------
# 10. Free vs Paid Applications
# --------------------------------------------------

plt.figure(figsize=(6, 6))

type_count = apps_df["Type"].value_counts()

plt.pie(
    type_count,
    labels=type_count.index,
    autopct="%1.1f%%"
)

plt.title("Free vs Paid Apps")

plt.show()


# --------------------------------------------------
# 11. Top 10 Categories by Average Rating
# --------------------------------------------------

top_categories = (
    apps_df.groupby("Category")["Rating"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10, 5))

sns.barplot(
    x=top_categories.index,
    y=top_categories.values
)

plt.title("Top 10 Categories by Average Rating")
plt.xlabel("Category")
plt.ylabel("Average Rating")
plt.xticks(rotation=45)

plt.show()


# --------------------------------------------------
# 12. Most Installed Apps Categories
# --------------------------------------------------

install_category = (
    apps_df.groupby("Category")["Installs"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10, 5))

sns.barplot(
    x=install_category.index,
    y=install_category.values
)

plt.title("Top Categories by Average Installs")
plt.xlabel("Category")
plt.ylabel("Average Installs")
plt.xticks(rotation=45)

plt.show()
# ==================================================
# Part 3: Advanced Analysis & Final Insights
# ==================================================


# --------------------------------------------------
# 13. Cleaning Size Column
# --------------------------------------------------

apps_df["Size"] = (
    apps_df["Size"]
    .astype(str)
    .str.replace("M", "")
    .str.replace("k", "")
    .str.replace("Varies with device", "0")
)

apps_df["Size"] = pd.to_numeric(
    apps_df["Size"],
    errors="coerce"
)


# --------------------------------------------------
# 14. App Size Distribution
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    apps_df["Size"],
    bins=30,
    kde=True
)

plt.title("Distribution of App Size")
plt.xlabel("Size")
plt.ylabel("Number of Apps")

plt.show()


# --------------------------------------------------
# 15. Price Distribution
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    apps_df["Price"],
    bins=30,
    kde=True
)

plt.title("Distribution of App Prices")
plt.xlabel("Price")
plt.ylabel("Number of Apps")

plt.show()


# --------------------------------------------------
# 16. Sentiment Analysis
# --------------------------------------------------

print("\nUser Review Sentiment Analysis:")

sentiment_count = reviews_df["Sentiment"].value_counts()

print(sentiment_count)


plt.figure(figsize=(7, 5))

sns.barplot(
    x=sentiment_count.index,
    y=sentiment_count.values
)

plt.title("User Review Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")

plt.show()


# --------------------------------------------------
# 17. Top Reviewed Apps
# --------------------------------------------------

top_reviews = (
    apps_df.sort_values(
        by="Reviews",
        ascending=False
    )
    .head(10)
)

plt.figure(figsize=(12, 5))

sns.barplot(
    x="App",
    y="Reviews",
    data=top_reviews
)

plt.title("Top 10 Most Reviewed Apps")
plt.xlabel("App Name")
plt.ylabel("Number of Reviews")

plt.xticks(rotation=90)

plt.show()


# --------------------------------------------------
# 18. Final Insights
# --------------------------------------------------

print("\n================ FINAL INSIGHTS ================")

print("""
1. The Google Play Store contains thousands of applications 
   across multiple categories.

2. Most applications are free, while only a small percentage 
   are paid applications.

3. Ratings and reviews help understand user satisfaction.

4. Categories with higher installs represent popular areas 
   among Android users.

5. Sentiment analysis provides insights into user opinions 
   and overall app experience.
""")


print("Google Play Store Data Analysis Project Completed Successfully!")