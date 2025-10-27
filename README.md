# customer-churn-prediction

## 1\. Project Objective 

The goal of this project is to build a machine learning model that can predict customer churn (i.e., when a customer stops using the company's services).

This is a critical business problem for telecom companies, as the cost of acquiring a new customer is significantly higher than retaining an existing one. By identifying customers who are at high risk of churning, the company can proactively target them with retention strategies (like special discounts or service upgrades) to reduce churn and save costs.

This project solves a **binary classification problem**, where the model will predict "Yes" (will churn) or "No" (will not churn).

## 2\. Dataset 

The dataset used is the **Telco Customer Churn** dataset, which is publicly available on Kaggle.

  * **Link:** [Kaggle Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
  * **Content:** The dataset contains 7,043 customer records with 20 features (columns) describing their demographics, account information, and the services they use.

## 3\. Project Workflow 

My machine learning workflow consisted of the following steps:

1.  **Data Cleaning:**

      * Loaded the dataset.
      * Identified and handled missing values: The `TotalCharges` column had 11 missing values, which were filled with `0` (assuming they were new customers with no charges yet).
      * Converted `TotalCharges` from an 'object' to a 'numeric' (float) data type.
      * Dropped the `customerID` column as it is a unique identifier and provides no predictive value.

2.  **Data Preprocessing:**

      * **One-Hot Encoding:** Converted all categorical columns (e.g., `PaymentMethod`, `Contract`, `InternetService`) into numerical "dummy" variables so the model could understand them.
      * **Target Variable:** Encoded the `Churn` column from "Yes"/"No" to `1`/`0`.
      * **Feature Scaling:** Applied `StandardScaler` to all numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`). This ensures all features are on the same scale, which is essential for models like Logistic Regression.

3.  **Train-Test Split:**

      * Split the final, processed data into training (80%) and testing (20%) sets.
      * Used `stratify=y` during the split to ensure that both the training and testing sets had the same proportion of churners as the original dataset.

4.  **Model Training & Evaluation:**

      * Trained two different classification models to compare performance.
      * **Model 1: Logistic Regression** (a great baseline model for classification).
      * **Model 2: Random Forest Classifier** (a more powerful, ensemble-based model).
      * Evaluated models using a **Confusion Matrix**, **Accuracy**, **Precision**, **Recall**, and **F1-Score**.

## 4\. Results & Model Performance 

Accuracy alone is a poor metric for this problem because the dataset is **imbalanced** (most customers did not churn). Therefore, **Recall for Class 1 (Churn='Yes')** is the most important metric. We want to find as many actual churners as possible (minimize false negatives).

Here is a comparison of the results on the test set:

| Model | Accuracy | F1-Score (Class 1) | Precision (Class 1) | Recall (Class 1) |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | `0.81` | `0.60` | `0.66` | `0.56` |
| **Random Forest** | `0.79` | `0.56` | `0.64` | `0.49` |

*(Note: These are placeholder values. **Update them with your model's scores** from the `classification_report`)*

## 5\. Conclusion & Key Insights 

Both models performed similarly, with the Logistic Regression model having a slight edge in identifying customers who will churn (higher Recall for Class 1).

  * **Model Choice:** I would choose the **Logistic Regression** model. While its overall accuracy is similar to the Random Forest, its higher recall for the "Churn" class means it's better at the primary business goal: **finding the customers who are at risk of leaving**.

  * **Actionable Insights:** A business could use this model to generate a daily or weekly list of "high-risk" customers. The marketing team could then target these specific customers with retention offers, effectively allocating their budget and reducing churn.

*(**Bonus Insight**: If you also run a feature importance analysis on the Random Forest, you can add this):*

  * **Top Churn Predictors:** The analysis showed that the most important features for predicting churn were:

    1.  `Contract` (month-to-month)
    2.  `tenure` (how long they've been a customer)
    3.  `TotalCharges`

    This suggests the business should **focus on converting month-to-month customers to long-term contracts**.

## 6\. How to Run This Project 

To run this project, you can use the `Customer Churn Project.ipynb` notebook in Google Colab or any local Jupyter environment.

1.  Clone this repository.
2.  Install the required libraries:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```
3.  Upload the `WA_Fn-UseC_-Telco-Customer-Churn.csv` dataset.
4.  Run the cells in the notebook from top to bottom.
