import pandas as pd
import tabfm

from sklearn.model_selection import train_test_split


# Load CSV
df = pd.read_csv("demo.csv")

# Target column
X = df.drop(columns=["churn"])
y = df["churn"]


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42  # it will pick the same rows every time you run it
)


# Load TabFM
model = tabfm.tabfm_v1_0_0_pytorch.load(
    model_type="classification"
)

clf = tabfm.TabFMClassifier(model=model)


# Train
clf.fit(X_train, y_train)


# Predict
predictions = clf.predict(X_test)

probabilities = clf.predict_proba(X_test)


print("Classes:")
print(clf.classes_)

print("\ny test")
print(y_test)


print("\nx test")
print(X_test)

print("\nPredictions:")
print(predictions)

print("\nProbabilities:")
print(probabilities)