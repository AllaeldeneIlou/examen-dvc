import pandas as pd
from sklearn.model_selection import train_test_split
import os

def main():
    # Load dataset
    df = pd.read_csv("data/raw/raw.csv")

    # Features (X) and target (y)
    X = df.drop(columns=["silica_concentrate"])
    y = df["silica_concentrate"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Ensure output folder exists
    os.makedirs("data/processed", exist_ok=True)

    # Save splits
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    print("Dataset split and saved in data/processed/")

if __name__ == "__main__":
    main()
