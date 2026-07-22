import pandas as pd
import tabfm
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------------------------
# 1. Load Titanic dataset
# -------------------------------------------------
df = pd.read_csv("tested.csv")

# -------------------------------------------------
# 2. Remove unnecessary columns
# -------------------------------------------------
df = df.drop(columns=[
    "PassengerId",
    "Name",
    "Ticket",
    "Cabin"
])

# -------------------------------------------------
# 3. Handle missing values
# -------------------------------------------------

# Numerical column
df["Age"] = df["Age"].fillna(df["Age"].median())

# Categorical column
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# -------------------------------------------------
# 4. Split Features and Target
# -------------------------------------------------
X = df.drop(columns=["Survived"])
y = df["Survived"]

# -------------------------------------------------
# 5. Use last 10 rows as test data
# -------------------------------------------------
X_train = X.iloc[:-10]
y_train = y.iloc[:-10]

X_test = X.iloc[-10:]
y_test = y.iloc[-10:]

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# -------------------------------------------------
# 6. Load TabFM
# -------------------------------------------------
model = tabfm.tabfm_v1_0_0_pytorch.load(
    model_type="classification"
)

clf = tabfm.TabFMClassifier(model=model)

# -------------------------------------------------
# 7. Give training examples to TabFM
# -------------------------------------------------
clf.fit(X_train, y_train)

# -------------------------------------------------
# 8. Predict
# -------------------------------------------------
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)

# -------------------------------------------------
# 9. Show results
# -------------------------------------------------
print("\nClasses:")
print(clf.classes_)

print("\nPrediction Results")

for i in range(len(X_test)):
    print("-" * 60)
    print(f"Passenger Index : {X_test.index[i]}")
    print(f"Actual          : {y_test.iloc[i]}")
    print(f"Prediction      : {predictions[i]}")
    print(f"Probability     : {probabilities[i]}")

# -------------------------------------------------
# 10. Evaluate
# -------------------------------------------------
print("\nAccuracy:")
print(accuracy_score(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))