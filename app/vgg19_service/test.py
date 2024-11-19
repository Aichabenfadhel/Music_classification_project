import os
import sys
import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

# Set encoding environment variable
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)

# Path to JSON file
DATA_PATH = "app/vgg19_service/data_10.json"

def load_data(data_path):
    """Loads training dataset from JSON file.
       :param data_path (str): Path to JSON file containing data
       :return X (ndarray): Inputs
       :return y (ndarray): Targets
    """
    try:
        with open(data_path, "r", encoding="utf-8", errors="replace") as fp:
            data = json.load(fp)

        # Convert lists to numpy arrays
        X = np.array(data["mfcc"])
        y = np.array(data["labels"])

        print("Data successfully loaded!")

        return X, y
    except FileNotFoundError:
        print(f"The file {data_path} was not found.")
    except json.JSONDecodeError:
        print("Error decoding JSON from the file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Load data
    X, y = load_data(DATA_PATH)

    if X is not None and y is not None:
        # Create train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        # Build network topology
        model = keras.Sequential([

            # Input layer
            keras.layers.InputLayer(input_shape=(X.shape[1], X.shape[2])),
            keras.layers.Flatten(),

            # 1st dense layer
            keras.layers.Dense(512, activation='relu'),

            # 2nd dense layer
            keras.layers.Dense(256, activation='relu'),

            # 3rd dense layer
            keras.layers.Dense(64, activation='relu'),

            # Output layer
            keras.layers.Dense(10, activation='softmax')
        ])

        # Compile model
        optimiser = keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=optimiser,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.summary()

        # Train model
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=50)
        # Sauvegarder le modèle
        model.save('vgg_model.keras')
        print("model saved successfully")
