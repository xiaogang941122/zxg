import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Conv1D, GRU, BatchNormalization, Dropout, Dense, concatenate
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Example Data (replace with actual data)
train_data = np.random.rand(1000, 24, 1)  # Shape: (samples, time_steps, features)
val_data = np.random.rand(200, 24, 1)
test_data = np.random.rand(200, 24, 1)
train_labels = np.random.rand(1000)  # Shape: (samples,)
val_labels = np.random.rand(200)
test_labels = np.random.rand(200)

# Data normalization
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data.reshape(-1, train_data.shape[-1])).reshape(train_data.shape)
val_data = scaler.transform(val_data.reshape(-1, val_data.shape[-1])).reshape(val_data.shape)
test_data = scaler.transform(test_data.reshape(-1, test_data.shape[-1])).reshape(test_data.shape)

# Model Inputs (adjust according to your dataset)
input1 = Input(shape=(train_data.shape[1], 1))  # shape: (time_steps, features)
input2 = Input(shape=(train_data.shape[1], 1))  # Another input for additional data if applicable

# CNN for feature extraction (Input1)
conv_out1_1 = Conv1D(filters=64, kernel_size=round(train_data.shape[1] / 2), activation='relu')(input1)
conv_out1_1 = BatchNormalization()(conv_out1_1)
conv_out1_1 = Dropout(0.2)(conv_out1_1)

# CNN for feature extraction (Input2, if needed, else use same input1 as input2)
conv_out2_1 = Conv1D(filters=64, kernel_size=round(train_data.shape[1] / 2), activation='relu')(input2)
conv_out2_1 = BatchNormalization()(conv_out2_1)
conv_out2_1 = Dropout(0.2)(conv_out2_1)

# GRU layers
gru_out1 = GRU(64, activation='tanh', recurrent_dropout=0.2, return_sequences=False)(conv_out1_1)
gru_out2 = GRU(64, activation='tanh', recurrent_dropout=0.2, return_sequences=False)(conv_out2_1)

# Concatenate the outputs
combined = concatenate([gru_out1, gru_out2])

# Dense output layer
output = Dense(1)(combined)

# Define the model
model = Model(inputs=[input1, input2], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Fit the model
history = model.fit([train_data, train_data], train_labels,
                    epochs=100, batch_size=32,
                    validation_data=([val_data, val_data], val_labels),
                    callbacks=[early_stopping])

# Predictions on the test data
predicted_labels = model.predict([test_data, test_data])

# Evaluate model performance
mse = mean_squared_error(test_labels, predicted_labels)
mae = mean_absolute_error(test_labels, predicted_labels)
rmse = np.sqrt(mse)
print(f'MSE: {mse}, MAE: {mae}, RMSE: {rmse}')

# Plotting training & validation loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

# Plotting real vs predicted values on the test set
plt.figure(figsize=(12, 6))
plt.plot(test_labels, label='Real Values', color='blue')
plt.plot(predicted_labels, label='Predicted Values', color='orange')
plt.title('Real vs Predicted Values')
plt.ylabel('Power Load')
plt.xlabel('Sample Index')
plt.legend(loc='upper right')
plt.show()
