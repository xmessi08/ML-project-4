import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor

housing = fetch_california_housing(as_frame=True)
df = housing.frame

TARGET_COL = "MedHouseVal"

X = df.select_dtypes(include=[np.number]).drop(columns=[TARGET_COL])
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

base_model = LinearRegression()
base_model.fit(X_train, y_train)
y_pred_base = base_model.predict(X_test)

base_metrics = {
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_base)),
    'MAE': mean_absolute_error(y_test, y_pred_base),
    'R2': r2_score(y_test, y_pred_base)
}

vif_df = pd.DataFrame()
vif_df["Feature"] = X_train.columns
vif_df["VIF"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

print("--- VIF (Мультиколлинеарность) ---")
print(vif_df.sort_values(by="VIF", ascending=False))

cols_to_drop = ['AveBedrms', 'Latitude', 'Longitude']

X_train_red = X_train.drop(columns=cols_to_drop)
X_test_red = X_test.drop(columns=cols_to_drop)

red_model = LinearRegression()
red_model.fit(X_train_red, y_train)
y_pred_red = red_model.predict(X_test_red)

red_metrics = {
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_red)),
    'MAE': mean_absolute_error(y_test, y_pred_red),
    'R2': r2_score(y_test, y_pred_red)
}

comparison = pd.DataFrame([base_metrics, red_metrics], index=['Базовая', 'Сокращённая'])
print("\n--- Сравнение метрик ---")
print(comparison)

residuals = y_test - y_pred_red

plt.figure(figsize=(8, 5))
plt.scatter(y_pred_red, residuals, alpha=0.3, color='indigo')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Предсказанные значения")
plt.ylabel("Остатки (Residuals)")
plt.title("График остатков")
plt.grid(True)
plt.show()