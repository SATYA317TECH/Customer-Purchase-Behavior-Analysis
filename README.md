# Customer Purchase Behavior Analysis

## 📌 Project Overview

This project analyzes **customer purchase behavior** using real-world e-commerce transaction data.  
The main objective is to understand how customers behave, segment them based on their purchasing patterns, and build a simple machine learning model to **predict which customers are likely to make repeat purchases**.

This project is created as part of my learning journey as an **aspiring Data Scientist (Fresher)** and demonstrates a complete beginner-level data science workflow.

---

## 🧠 Problem Statement

Businesses want to answer questions such as:
- Which customers are valuable?
- Which customers are likely to return?
- How can customers be grouped for better marketing strategies?

This project addresses these questions using data analysis, clustering, and machine learning techniques.

---

## 📂 Project Structure

```
Customer-Purchase-Behavior-Analysis/
│
├── customer_purchase_analysis.py
├── requirements.txt
│
├── dataset/
│ ├── README.md
│
├── outputs/
│ ├── spending_distribution.png
│ ├── customer_segmentation.png
│ ├── feature_importance.png
│
├── Predicted list/
│ ├── predicted_repeat_customers.csv
│ └── Predicted_Customers.png
│
└── README.md
```

---

## 📥 Dataset Information

The dataset is not included in this repository because it exceeds GitHub’s file size limit.
You can download the dataset from Kaggle and place it inside the `dataset/` folder before running the code.

---

## 🛠️ Tools and Technologies Used

- **Python**
- **Pandas & NumPy** – Data manipulation and analysis
- **Matplotlib & Seaborn** – Data visualization
- **Scikit-learn** – Machine learning
- **K-Means Clustering** – Customer segmentation
- **Random Forest Classifier** – Repeat customer prediction

---

## 🧪 Project Workflow

1. **Data Loading**
   - Loaded real-world e-commerce transaction data

2. **Data Cleaning**
   - Removed missing customer IDs
   - Removed cancelled orders and invalid records
   - Created a total purchase amount feature

3. **Exploratory Data Analysis (EDA)**
   - Visualized customer spending behavior
   - Analyzed relationships between purchase frequency and spending

4. **Feature Engineering (RFM Analysis)**
   - Recency: Days since last purchase
   - Frequency: Number of purchases
   - Monetary: Total amount spent

5. **Customer Segmentation**
   - Grouped customers into segments using K-Means clustering

6. **Machine Learning Model**
   - Built a Random Forest model to predict repeat customers

7. **Prediction Output**
   - Displayed and saved a list of customers predicted to repeat purchases

---

## 📈 Key Insights

- Customers with higher purchase frequency are more likely to return
- High-value customers contribute a significant portion of revenue
- Customer segmentation helps identify loyal, occasional, and at-risk customers
- Predicting repeat customers can help businesses improve retention strategies



