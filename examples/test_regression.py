# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0

"""Example showing how to run regression with TabFM v1.0.0 using CSV."""

import numpy as np
import pandas as pd
import tabfm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def run_example(csv_file="demo2.csv", model=None) -> np.ndarray:
  """Loads CSV data and runs regression."""

  if model is None:
    # Option A: JAX Backend
    # model = tabfm.tabfm_v1_0_0_jax.load(model_type="regression")

    # Option B: PyTorch Backend
    model = tabfm.tabfm_v1_0_0_pytorch.load(model_type="regression")

  # Initialize regressor
  reg = tabfm.TabFMRegressor(model=model)

  # Load external CSV
  df = pd.read_csv(csv_file)

  # Separate features and target
  X_train = df.drop(columns=["house_price"])
  y_train = df["house_price"].values

  # Example test data
  X_test = pd.DataFrame({
    "area_sqft": [1900, 1000],
    "bedrooms": [4, 2],
    "bathrooms": [3, 1],
    "location": ["City", "Town"],
    "age_years": [5, 25],
    "garage_spaces": [2, 1],
})
  # Train
  reg.fit(X_train, y_train)

#   print("Classes:", reg.classes_)

  # Predict
  preds = reg.predict(X_test)

  return preds


if __name__ == "__main__":
  print(
      "Running TabFM regression model..."
      " (first execution may take a few minutes)"
  )

  predictions = run_example("demo2.csv")

  print("Regression predictions:\n", predictions)