import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set up CPU threading for efficient parallelism
tf.config.threading.set_intra_op_parallelism_threads(64)  # Maximize intra-op parallelism across 64 CPUs
tf.config.threading.set_inter_op_parallelism_threads(64)  # Maximize inter-op parallelism across 64 CPUs

# Load the data
X = pd.read_csv('X_imputed.csv')
y = pd.read_csv('y_data.csv')

# Replace missing values in y with NaN
y = y.replace("?", np.nan)  # If missing values are marked with "?" in the dataset
y_numeric = y.drop(columns=['statecounty']) # Remove 'statecounty'

# Normalize X (remove the 'statecounty' column)
X_numeric = X.drop(columns=['statecounty'])  # Remove 'statecounty'
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_numeric), columns=X_numeric.columns)

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_numeric, test_size=0.2, random_state=42)

# Custom masked loss function to handle missing values in y
def masked_loss(y_true, y_pred):
    # Create a mask for NaN values in y_true (target values)
    mask = tf.math.is_finite(y_true)  # Mask where y_true is finite (not NaN)

    # Apply the mask to both y_true and y_pred
    y_true_masked = tf.where(mask, y_true, 0.0)  # Replace NaNs in y_true with 0.0 for loss calculation
    y_pred_masked = tf.where(mask, y_pred, 0.0)  # Align y_pred with masked y_true

    # Compute mean squared error for the masked values
    return tf.reduce_mean(tf.square(y_true_masked - y_pred_masked))

# Define an MLP architecture
def build_model():
    model = tf.keras.models.Sequential()

    # Input layer
    model.add(tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)))

    # First hidden layer
    model.add(tf.keras.layers.Dense(units=256, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.3))

    # Second hidden layer
    model.add(tf.keras.layers.Dense(units=256, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.3))

    # Output layer for multi-output regression
    model.add(tf.keras.layers.Dense(y_train.shape[1], activation='linear'))

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=masked_loss,
                  metrics=['mae'])

    return model

# Build the model
model = build_model()

# Train the model
model.fit(X_train, y_train, epochs=50, validation_split=0.2, batch_size=64)

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test MAE: {mae}")

# Save the model
model.save('best_mlp_model.h5')
