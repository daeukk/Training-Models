import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

file_path = "filtered_data_kalman_scaled.xlsx"
df = pd.read_excel(file_path)

X = df.iloc[:, :4].to_numpy()
Y = df.iloc[:, 4].to_numpy()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}")

rf_model = RandomForestRegressor(
    n_estimators=300,          
    max_depth=12,              
    min_samples_split=3,       
    random_state=42,
    n_jobs=-1                  
)

rf_model.fit(X_train, Y_train)

Y_rf_pred_train = rf_model.predict(X_train)
Y_rf_pred_test = rf_model.predict(X_test)

def evaluate_metrics(y_true, y_pred, label=""):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"\n{label} Results:")
    print(f"R² Score: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    return r2, mse, rmse, mae

r2_train, mse_train, rmse_train, mae_train = evaluate_metrics(Y_train, Y_rf_pred_train, "Train Set")
r2_test, mse_test, rmse_test, mae_test = evaluate_metrics(Y_test, Y_rf_pred_test, "Test Set")

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 12})
plt.figure(figsize=(8, 5))
plt.scatter(Y_test, Y_rf_pred_test, alpha=0.5, color="blue", label="Predictions")
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], 
         color="red", linestyle="dashed", label="Ideal Line")

plt.xlabel("Actual", fontsize=14, fontfamily='Times New Roman')
plt.ylabel("Predicted", fontsize=14, fontfamily='Times New Roman')
plt.legend(fontsize=12, loc="upper left", frameon=True, prop={'family': 'Times New Roman', 'size': 12})
plt.xticks(fontsize=12, fontfamily='Times New Roman')
plt.yticks(fontsize=12, fontfamily='Times New Roman')

r2_x_position = min(Y_test) + (max(Y_test) - min(Y_test)) * 0.5 
r2_y_position = max(Y_rf_pred_test) - (max(Y_rf_pred_test) - min(Y_rf_pred_test)) * 0.2
plt.text(r2_x_position, r2_y_position, 
         f"$R^2$ = {r2_test:.4f}", 
         fontsize=14, fontfamily='Times New Roman')
plt.show()