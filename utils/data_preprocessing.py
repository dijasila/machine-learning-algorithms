import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(data_path):
    data = pd.read_csv(data_path)

    # Example preprocessing steps
    # 1. Fill missing values
    data.fillna(data.mean(), inplace=True)

    # 2. Encoding categorical variables
    le = LabelEncoder()
    data['categorical_column'] = le.fit_transform(data['categorical_column'])

    # 3. Feature scaling
    scaler = StandardScaler()
    features = data.iloc[:, :-1]
    scaled_features = scaler.fit_transform(features)

    # Return the preprocessed data
    return pd.DataFrame(scaled_features), data.iloc[:, -1]

if __name__ == "__main__":
    X, y = preprocess_data('data/large_dataset.csv')
    print(X.head())