# ==========================================
# Customer Purchase Behavior Analysis
# ==========================================

# 1. Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ------------------------------------------
# 2. Load Dataset
# ------------------------------------------
df = pd.read_csv("dataset/online_retail.csv", encoding="ISO-8859-1")

print("Dataset loaded successfully")
print(df.head())

# ------------------------------------------
# 3. Data Understanding
# ------------------------------------------
print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# ------------------------------------------
# 4. Data Cleaning
# ------------------------------------------

# Remove rows with missing CustomerID
df = df.dropna(subset=["CustomerID"])

# Remove cancelled orders and free items
df = df[df["Quantity"] > 0]
df = df[df["UnitPrice"] > 0]

# Create total purchase amount
df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

# Convert date column
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

print("\nData cleaning completed")

# ------------------------------------------
# 5. RFM Feature Engineering
# ------------------------------------------

snapshot_date = df["InvoiceDate"].max()

rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
    "InvoiceNo": "count",
    "TotalAmount": "sum"
}).reset_index()

rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]

print("\nRFM features created")
print(rfm.head())

# ------------------------------------------
# 6. Exploratory Data Analysis
# ------------------------------------------

plt.figure(figsize=(6,4))
sns.histplot(rfm["Monetary"], bins=40)
plt.title("Customer Spending Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.scatterplot(x="Frequency", y="Monetary", data=rfm)
plt.title("Frequency vs Monetary Value")
plt.show()

# ------------------------------------------
# 7. Customer Segmentation (K-Means)
# ------------------------------------------

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

kmeans = KMeans(n_clusters=3, random_state=42)
rfm["CustomerSegment"] = kmeans.fit_predict(rfm_scaled)

plt.figure(figsize=(6,4))
sns.scatterplot(
    x="Frequency",
    y="Monetary",
    hue="CustomerSegment",
    data=rfm,
    palette="Set2"
)
plt.title("Customer Segmentation")
plt.show()

print("\nCustomer Segments:")
print(rfm["CustomerSegment"].value_counts())

# ------------------------------------------
# 8. Create Target Variable (Simple Logic)
# ------------------------------------------

# If customer purchased more than 3 times → Repeat customer
rfm["RepeatCustomer"] = rfm["Frequency"].apply(lambda x: 1 if x > 3 else 0)

print("\nRepeat Customer Distribution:")
print(rfm["RepeatCustomer"].value_counts())

# ------------------------------------------
# 9. Prepare Data for Machine Learning
# ------------------------------------------

X = rfm[["Recency", "Frequency", "Monetary"]]
y = rfm["RepeatCustomer"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------
# 10. Train Random Forest Model
# ------------------------------------------

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

print("\nModel training completed")

# ------------------------------------------
# 11. Model Evaluation
# ------------------------------------------

predictions = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# ------------------------------------------
# 12. Display Predicted Repeat Customers
# ------------------------------------------

# Create a results DataFrame
results = X_test.copy()
results["CustomerID"] = rfm.loc[X_test.index, "CustomerID"]
results["PredictedRepeat"] = predictions

# Filter customers predicted to repeat
repeat_customers_predicted = results[results["PredictedRepeat"] == 1]

print("\nList of Customers Predicted to Repeat:")
print(repeat_customers_predicted[["CustomerID", "Recency", "Frequency", "Monetary"]].head(10))

print(f"\nTotal customers predicted to repeat: {len(repeat_customers_predicted)}")

repeat_customers_predicted.to_csv(
    "predicted_repeat_customers.csv", index=False
)
print("Saved predicted repeat customers to predicted_repeat_customers.csv")


# ------------------------------------------
# 13. Feature Importance
# ------------------------------------------

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# ------------------------------------------
# 14. Simple Business Insights
# ------------------------------------------

print("\nBusiness Insights:")
print("- Customers who purchase frequently are more likely to return")
print("- High spending customers contribute most of the revenue")
print("- Customer segmentation helps target marketing strategies")

print("\nProject completed successfully!")
