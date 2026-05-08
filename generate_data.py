import pandas as pd
import numpy as np

np.random.seed(42)
n = 500

ages = np.random.randint(18, 70, n)
incomes = np.random.randint(20000, 150000, n)
spending_scores = np.random.randint(1, 100, n)
purchase_frequency = np.random.randint(1, 52, n)
avg_order_value = np.round(np.random.uniform(20, 500, n), 2)
genders = np.random.choice(["Male", "Female"], n)
regions = np.random.choice(["North", "South", "East", "West"], n)
customer_ids = [f"CUST{str(i).zfill(4)}" for i in range(1, n + 1)]

df = pd.DataFrame({
    "CustomerID": customer_ids,
    "Age": ages,
    "Annual_Income": incomes,
    "Spending_Score": spending_scores,
    "Purchase_Frequency": purchase_frequency,
    "Avg_Order_Value": avg_order_value,
    "Gender": genders,
    "Region": regions
})

df.to_csv("data/customers.csv", index=False)
print(f"Dataset created: {df.shape[0]} rows, {df.shape[1]} columns")
