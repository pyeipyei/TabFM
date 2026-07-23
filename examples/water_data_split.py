import pandas as pd
import pandas as pd
import tabfm

# Load dataset
df = pd.read_excel("water_network_leak_dataset.xlsx")

# Separate classes
class_0 = df[df["Leak_Class"] == 0]
class_1 = df[df["Leak_Class"] == 1]

# Shuffle the data (recommended)
class_0 = class_0.sample(frac=1, random_state=42)
class_1 = class_1.sample(frac=1, random_state=42)

# Select required rows
train_class_0 = class_0.iloc[:1280]
train_class_1 = class_1.iloc[:320]

test_class_0 = class_0.iloc[1280:1600]
test_class_1 = class_1.iloc[320:400]

# Combine train and test datasets
train_data = pd.concat([train_class_0, train_class_1])
test_data = pd.concat([test_class_0, test_class_1])

# Shuffle again after combining
train_data = train_data.sample(frac=1, random_state=42)
test_data = test_data.sample(frac=1, random_state=42)

# Split features (X) and labels (y)
x_train = train_data.drop(columns=["Leak_Class", "Pipe_ID"])
y_train = train_data["Leak_Class"]

x_test = test_data.drop(columns=["Leak_Class", "Pipe_ID"])
y_test = test_data["Leak_Class"]

# Check sizes
print("x_train:", x_train.shape)
print("y_train:", y_train.shape)
print("x_test:", x_test.shape)
print("y_test:", y_test.shape)

# Check class distribution
print("\nTraining distribution:")
print(y_train.value_counts())

print("\nTesting distribution:")
print(y_test.value_counts())

#############################################

model = tabfm.tabfm_v1_0_0_pytorch.load(
    model_type="classification", device = "cuda"
)

clf = tabfm.TabFMClassifier(model=model)

clf.fit(x_train, y_train)

predictions = clf.predict(x_test)
probabilities = clf.predict_proba(x_test)


print("\nClasses:")
print(clf.classes_)

print("\nPrediction Results")

for i in range(len(x_test)):
    print("-" * 60)
    print(f"Pipe Index : {x_test.index[i]}")
    print(f"Actual          : {y_test.iloc[i]}")
    print(f"Prediction      : {predictions[i]}")
    print(f"Probability     : {probabilities[i]}")