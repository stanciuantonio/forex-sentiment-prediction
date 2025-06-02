import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("data/processed/eurusd_final_processed.csv")

# Define features and target
features = ['gdelt_sentiment', 'log_return', 'fwd_return']
target = 'label'

X = df[features].values
y = df[target].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

kernel = C(1.0, constant_value_bounds=(1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))

gpr = GPR(kernel=kernel, normalize_y=True)

gpr.fit(X_train, y_train)

y_pred, y_std = gpr.predict(X_test, return_std=True)

print("Test R^2 score:", gpr.score(X_test, y_test))