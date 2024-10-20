import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

def perform_feature_engineering(data_path, k=5):
    data = pd.read_csv(data_path)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Feature selection
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)

    # Return the transformed features
    return pd.DataFrame(X_new)

if __name__ == "__main__":
    engineered_features = perform_feature_engineering('data/large_dataset.csv')
    print(engineered_features.head())