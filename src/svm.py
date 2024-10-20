import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def run_svm(data_path='data/large_dataset.csv'):
    # Load the dataset
    data = pd.read_csv(data_path)
    X = data.iloc[:, :-1]  # Features
    y = data.iloc[:, -1]   # Target

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the SVM model
    model = SVR(kernel='linear')
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Model evaluation
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)  # Perfect prediction line
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('SVM Predictions vs True Values')

    # Save plot to image folder
    plt.savefig('image/svm_results.png')
    print("SVM results saved to 'image/svm_results.png'.")

if __name__ == "__main__":
    run_svm()