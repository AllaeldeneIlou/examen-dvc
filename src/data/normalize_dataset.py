import pandas as pd
from sklearn.preprocessing import StandardScaler

def normalize():
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")

    numeric_cols = X_train.select_dtypes(include=["number"]).columns

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled = scaler.transform(X_test[numeric_cols])

    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=numeric_cols)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=numeric_cols)

    X_train_scaled_df.to_csv("data/processed/X_train_scaled.csv", index=False)
    X_test_scaled_df.to_csv("data/processed/X_test_scaled.csv", index=False)

if __name__ == "__main__":
    normalize()
