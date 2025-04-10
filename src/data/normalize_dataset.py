import pandas as pd
from sklearn.preprocessing import StandardScaler

def normalize(train_path, test_path, out_train_path, out_test_path):
    X_train = pd.read_csv(train_path)
    X_test = pd.read_csv(test_path)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv(out_train_path, index=False)
    pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv(out_test_path, index=False)

if __name__ == "__main__":
    normalize(
        "data/processed/X_train.csv",
        "data/processed/X_test.csv",
        "data/processed/X_train_scaled.csv",
        "data/processed/X_test_scaled.csv",
    )
