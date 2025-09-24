import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Dropout, Lambda, Add
from sklearn.preprocessing import MinMaxScaler
import requests
import json

# --- Configuration ---
# ThingSpeak Details (will be read from GitHub Secrets)
CHANNEL_ID = os.environ.get('THINGSPEAK_CHANNEL_ID', '2968337')
READ_API_KEY = os.environ.get('THINGSPEAK_READ_API_KEY', 'JKW1ZRWYQDM0IEVM')
RESULTS = 8000 # Max results per request from ThingSpeak

DATASET_PATH = "dataset.csv"
TFLITE_MODEL_PATH = "tcn_model_quant_9_features.tflite"
MODEL_HEADER_PATH = 'tcn_model.h'
TARGET_VARIABLE = 'soil_moisture'
WINDOW_SIZE = 24

def residual_block(x, dilation_rate, nb_filters, kernel_size, dropout_rate=0.1):
    """Create a residual block for TCN."""
    # Dilated convolution
    conv = Conv1D(filters=nb_filters,
                  kernel_size=kernel_size,
                  dilation_rate=dilation_rate,
                  padding='causal',
                  activation='relu')(x)
    conv = Dropout(dropout_rate)(conv)
    
    # Second convolution
    conv = Conv1D(filters=nb_filters,
                  kernel_size=kernel_size,
                  dilation_rate=dilation_rate,
                  padding='causal',
                  activation='relu')(conv)
    conv = Dropout(dropout_rate)(conv)
    
    # Residual connection
    if x.shape[-1] != nb_filters:
        # Match dimensions for residual connection
        x = Conv1D(filters=nb_filters, kernel_size=1, padding='same')(x)
    
    return Add()([x, conv])

def create_tcn_model(input_shape, nb_filters=24, kernel_size=3, dilations=[1, 2, 4], dropout_rate=0.1):
    """Create a TCN model using standard Keras layers."""
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    
    # Initial convolution
    x = Conv1D(filters=nb_filters, kernel_size=1, padding='same')(x)
    
    # Stack residual blocks with different dilation rates
    for dilation in dilations:
        x = residual_block(x, dilation, nb_filters, kernel_size, dropout_rate)
    
    # Global temporal pooling
    x = Lambda(lambda x: x[:, -1, :])(x)  # Take last timestep
    
    # Output layer
    outputs = Dense(1)(x)
    
    return tf.keras.Model(inputs, outputs)

def fetch_thingspeak_data():
    """Fetches data from ThingSpeak and saves it to a CSV file."""
    print("Fetching data from ThingSpeak...")
    url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?api_key={READ_API_KEY}&results={RESULTS}"
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for bad status codes
        data = response.json()
        
        # Convert to DataFrame
        df = pd.DataFrame(data['feeds'])
        
        # Rename columns to match your dataset
        df = df.rename(columns={
            'created_at': 'timestamp',
            'field1': 'soil_moisture',
            'field2': 'soil_temp',
            'field3': 'ec',
            'field4': 'ph',
            'field5': 'ambient_temp',
            'field6': 'ambient_hum'
        })

        # Keep only the necessary columns (and drop others like entry_id, field7, field8)
        required_columns = ['timestamp', 'soil_moisture', 'soil_temp', 'ec', 'ph', 'ambient_temp', 'ambient_hum']
        df = df[required_columns]

        # Convert data types to numeric, coercing errors
        for col in required_columns[1:]: # Skip timestamp
             df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.to_csv(DATASET_PATH, index=False)
        print(f"Data saved successfully to {DATASET_PATH}")
        return True

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from ThingSpeak: {e}")
        return False
    except (KeyError, TypeError) as e:
        print(f"Error processing ThingSpeak JSON response: {e}")
        return False


def main():
    """Main function to run the complete training and conversion pipeline."""
    # Step 1: Fetch Data
    if not fetch_thingspeak_data():
        print("Halting process due to data fetching error.")
        return

    # Step 2: Load and Prepare Data
    print("Loading and preparing data...")
    df = pd.read_csv(DATASET_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    
    # Step 3: Feature Engineering
    print("Applying feature engineering...")
    df['hour'] = df.index.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['temp_humidity_interaction'] = df['ambient_temp'] * df['ambient_hum']
    df.drop('hour', axis=1, inplace=True)

    MODEL_FEATURES = [
        'soil_moisture', 'soil_temp', 'ec', 'ph', 'ambient_temp', 'ambient_hum',
        'hour_sin', 'hour_cos', 'temp_humidity_interaction'
    ]
    df_model = df[MODEL_FEATURES].copy()
    df_model.ffill(inplace=True)
    df_model.bfill(inplace=True)

    # Step 4: Data Splitting & Scaling
    print("Splitting and scaling data...")
    train_size = int(len(df_model) * 0.7)
    val_size = int(len(df_model) * 0.15)
    
    train_df = df_model.iloc[:train_size]
    val_df = df_model.iloc[train_size:train_size + val_size]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df)
    val_scaled = scaler.transform(val_df)
    
    # --- For MCU: Print scaler min/max values ---
    print("\n--- Scaler values for MCU ---")
    print("const float feature_min[NUM_FEATURES] = {")
    print(','.join([f'    {val:.6f}f' for val in scaler.data_min_]))
    print("};")
    print("\nconst float feature_max[NUM_FEATURES] = {")
    print(','.join([f'    {val:.6f}f' for val in scaler.data_max_]))
    print("};")

    # Step 5: Create Windowed Datasets
    print("\nCreating windowed datasets...")
    TARGET_COLUMN_INDEX = df_model.columns.get_loc(TARGET_VARIABLE)
    
    def create_dataset(data, window_size, target_index):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:(i + window_size)])
            y.append(data[i + window_size, target_index])
        return np.array(X), np.array(y)

    X_train, y_train = create_dataset(train_scaled, WINDOW_SIZE, TARGET_COLUMN_INDEX)
    X_val, y_val = create_dataset(val_scaled, WINDOW_SIZE, TARGET_COLUMN_INDEX)
    
    # Step 6: Build and Train TCN Model
    print("Building and training TCN model...")
    model = create_tcn_model(
        input_shape=(WINDOW_SIZE, X_train.shape[2]),
        nb_filters=24,
        kernel_size=3,
        dilations=[1, 2, 4],
        dropout_rate=0.1
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
        loss='mean_squared_error'
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )

    # Step 7: Convert to TFLite (Quantized)
    print("Converting to TensorFlow Lite...")
    def representative_dataset_gen():
        for i in range(100):
            yield [X_train[i:i+1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_model_quant = converter.convert()

    with open(TFLITE_MODEL_PATH, 'wb') as f:
        f.write(tflite_model_quant)
    print(f"Quantized TFLite model saved to: {TFLITE_MODEL_PATH}")

    # Step 8: Convert to C Header File
    print("Converting TFLite model to C header file...")
    os.system(f"xxd -i {TFLITE_MODEL_PATH} > {MODEL_HEADER_PATH}")
    
    with open(MODEL_HEADER_PATH, 'r') as f:
        content = f.read()
    
    array_name = 'g_tcn_model_data'
    new_declaration = f"const unsigned char {array_name}[] __attribute__((aligned(16))) = {{"
    content = content.replace(f"unsigned char {TFLITE_MODEL_PATH.replace('.', '_')}[] = {{", new_declaration)
    
    with open(MODEL_HEADER_PATH, 'w') as f:
        f.write(content)
    
    print(f"C header file '{MODEL_HEADER_PATH}' created successfully.")


if __name__ == "__main__":
    main()
