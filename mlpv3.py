import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

file_path = "filtered_data_kalman_scaled.xlsx"
df = pd.read_excel(file_path)

X = df.iloc[:, :4].values
Y = df.iloc[:, 4].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

nn_model = tf.keras.Sequential([
    tf.keras.layers.Dense(1024, activation=tf.keras.activations.swish, input_shape=(X_train.shape[1],)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1), 

    tf.keras.layers.Dense(512, activation=tf.keras.activations.swish),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Dense(256, activation=tf.keras.activations.swish),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(128, activation=tf.keras.activations.swish),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(64, activation=tf.keras.activations.swish),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(32, activation=tf.keras.activations.swish),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(1)
])

nn_model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-6), 
                 loss='huber', metrics=['mae'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=200, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=100, min_lr=1e-6)

history = nn_model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
                        epochs=1000, batch_size=16, callbacks=[early_stopping, reduce_lr])

eval_loss_test, eval_mae_test = nn_model.evaluate(X_test, Y_test, verbose=1)
Y_nn_pred_test = nn_model.predict(X_test)

eval_loss_train, eval_mae_train = nn_model.evaluate(X_train, Y_train, verbose=1)
Y_nn_pred_train = nn_model.predict(X_train)

r2_train = r2_score(Y_train, Y_nn_pred_train)
mse_train = mean_squared_error(Y_train, Y_nn_pred_train)
rmse_train = np.sqrt(mse_train)
mae_train = mean_absolute_error(Y_train, Y_nn_pred_train)

r2_test = r2_score(Y_test, Y_nn_pred_test)
mse_test = mean_squared_error(Y_test, Y_nn_pred_test)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(Y_test, Y_nn_pred_test)

print("\nTrain Set Results:")
print(f"R2 Score: {r2_train}")
print(f"MSE: {mse_train}")
print(f"RMSE: {rmse_train}")
print(f"MAE: {mae_train}")

print("\nTest Set Results:")
print(f"R2 Score: {r2_test}")
print(f"MSE: {mse_test}")
print(f"RMSE: {rmse_test}")
print(f"MAE: {mae_test}")

plt.figure(figsize=(8, 5))
plt.scatter(Y_test, Y_nn_pred_test, alpha=0.5, color="blue", label="Predictions")
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color="red", linestyle="dashed", label="Ideal Line")
plt.xlabel("Actual", fontsize=14, fontfamily='Times New Roman')
plt.ylabel("Predicted", fontsize=14, fontfamily='Times New Roman')
plt.legend(fontsize=12, loc="upper left", frameon=True, prop={'family': 'Times New Roman', 'size': 12})
plt.xticks(fontsize=12, fontfamily='Times New Roman')
plt.yticks(fontsize=12, fontfamily='Times New Roman')

r2_x_position = min(Y_test) + (max(Y_test) - min(Y_test)) * 0.5
r2_y_position = max(Y_nn_pred_test) - (max(Y_nn_pred_test) - min(Y_nn_pred_test)) * 0.2
plt.text(r2_x_position, r2_y_position, f"$R^2$ = {r2_test:.4f}", fontsize=14, fontfamily='Times New Roman')
plt.show()

converter = tf.lite.TFLiteConverter.from_keras_model(nn_model)
tflite_model = converter.convert()

with open("final_model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved successfully.")