import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def run_neural_network(data_path='data/large_dataset.csv'):
    # Load the dataset
    data = pd.read_csv(data_path)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Build the neural network model
    model = Sequential([
        Dense(64, activation='relu', input_dim=X_train.shape[1]),
        Dense(64, activation='relu'),
        Dense(1, activation='linear')
    ])

    model.compile(optimizer='adam', loss='mse')

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Make predictions
    y_pred = model.predict(X_test)

    # Visualizing the results
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)  # Perfect prediction line
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Neural Network Predictions vs True Values')

    # Save plot to image folder
    plt.savefig('image/neural_network_results.png')
    print("Neural Network results saved to 'image/neural_network_results.png'.")

if __name__ == "__main__":
    run_neural_network()