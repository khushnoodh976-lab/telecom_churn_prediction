# telecom_churn_prediction
#----------Telecom Churn Prediction Project----------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

SEED = 42
#----------1.Load----------
df = pd.read_csv("telecom_churn_raw.csv")
print("Raw shape:", df.shape)
print("Missing:\n", df.isna().sum())
print("Duplicates:", df.duplicated().sum())
print(df["contract_type"].value_counts(dropna=False))
print("Max changes:", df["monthly_charges"].max())
#----------2.Clean----------
df = df.drop_duplicates().reset_index(drop=True)
#Fix Contract Text
df["contract_type"] = df["contract_type"].str.strip().str.title()
df["contract_type"] = df["contract_type"].replace({"Month_to_Month":"Month_to_Month"})
#Outlies > 150 = wrong, make it missing
df.loc[df["monthly_charges"] > 150, "monthly_charges"] = np.nan
#Fill missing with median(median is safe,mean is affected by outlier)
df["tenure_months"] = df["tenure_months"].fillna(df["tenure_months"].median())
df["monthly_charges"] = df["monthly_charges"].fillna(df["monthly_charges"].median())
print("\nClean shape:", df.shape)
print("Missing after clean:", df.isna().sum().sum())
print("Churn %:\n", df["churn"].value_counts(normalize=True) * 100)
#----------3.EDA (Answer in numbers)----------
print("\n---Churn by contract(&)---")
print(df.groupby("contract_type")["churn"].apply(lambda s: (s=="Yes").mean()*100))
print("\n--- Average by Churn ---")
print(df.groupby("churn")[["tenure_months","monthly_charges"]].mean())
print("\n---Churn by support tickets(&)---")
df["tickets_group"] = df["support_tickets"].apply(lambda x: "3+" if x >= 3 else "0-2")
print(df.groupby("tickets_group")["churn"].apply(lambda s: (s=="Yes").mean()*100))
print("\n--- Churn by Tenure Group (%) ---")
df["tenure_group"] = pd.cut(df["tenure_months"], bins=[0,12,24,60], labels=["0-12","13-24","25-60"])
print(df.groupby("tenure_group", observed=True)["churn"].apply(lambda s: (s=="Yes").mean()*100))
#----------4.Visualisation(Charts)----------
#Chart 1
plt.figure()
df["churn"].value_counts().plot(kind="bar", color=["blue","red"])
plt.title("Churn Distribution")
plt.ylabel("Customers")
plt.tight_layout(); plt.savefig("01_churn.png"); plt.close()
#Chart 2
rate = df.groupby("contract_type")["churn"].apply(lambda s: (s=="Yes").mean()*100)
plt.figure()
rate.plot(kind="bar", color="#4C78A8")
plt.title("Churn Rate by Contract Type")
plt.ylabel("Churn rate (%)")
plt.tight_layout(); plt.savefig("02_churn_by_contract.png"); plt.close()
#Chart 3
plt.figure()
df.groupby("tickets_group")["churn"].apply(lambda s: (s=="Yes").mean()*100).plot(kind="bar", color="orange")
plt.title("Churn Rate by Support Tickets")
plt.ylabel("Churn rate (%)")
plt.tight_layout(); plt.savefig("03_churn_by_tickets.png"); plt.close()
#Chart 4
plt.figure(); df.groupby("tickets_group")["churn"].apply(lambda s: (s=="Yes").mean()*100).plot(kind="bar", color="orange")
plt.title("04 - Churn Rate by Support Tickets"); plt.ylabel("Churn rate (%)")
plt.tight_layout(); plt.savefig("04_churn_by_tickets.png"); plt.close()
#Chart 5
plt.figure(); df.groupby("tenure_group", observed=True)["churn"].apply(lambda s: (s=="Yes").mean()*100).plot(kind="bar", color="green")
plt.title("05 - Churn Rate by Tenure Group"); plt.ylabel("Churn rate (%)")
plt.tight_layout(); plt.savefig("05_churn_by_tenure_group.png"); plt.close()
#Chart 6
plt.figure()
for label, color in [("No","blue"),("Yes","red")]:
    sub = df[df.churn==label]
    plt.scatter(sub["tenure_months"], sub["monthly_charges"], label=label, alpha=0.7, color=color)
plt.title("06 - Tenure vs Monthly Charges by Churn"); plt.xlabel("Tenure"); plt.ylabel("Charges"); plt.legend()
plt.tight_layout(); plt.savefig("06_tenure_vs_charges.png"); plt.close()
print("\nCharts saved: 01, 02, 03, 04, 05, 06")
#----------5.Model----------
# Convert text to numbers
X = pd.get_dummies(df[["contract_type","tenure_months","monthly_charges","support_tickets"]],
                   columns=["contract_type"], drop_first=True, dtype=float)
y = (df["churn"] == "Yes").astype(int)  # Yes=1, No=0

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=SEED, stratify=y)

print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
#----------6.Train + Evaluate----------
def evaluate(name, model):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"\n{name}")
    print(f"Accuracy: {accuracy_score(y_test, pred):.3f}")
    print(f"Precision: {precision_score(y_test, pred, zero_division=0):.3f}")
    print(f"Recall: {recall_score(y_test, pred, zero_division=0):.3f}")
    print(f"F1: {f1_score(y_test, pred, zero_division=0):.3f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
    return model

m1 = evaluate("Logistic Regression", LogisticRegression(max_iter=1000))
m2 = evaluate("Random Forest", RandomForestClassifier(n_estimators=100, random_state=SEED))
# ----------7.Conclusion----------
print("\nCONCLUSION: Month-to-Month + many support tickets + short tenure = highest churn risk.")
print("Retention team should contact these customers first.")
