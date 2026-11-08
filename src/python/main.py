"""Predict the survival of passengers on the Titanic using logistic regression."""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Resolve the data directory relative to this file
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Columns to drop because they are not useful for prediction
DROP_COLS = ["PassengerId", "Name", "Ticket", "Cabin"]

# Set pandas display options for better readability
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)


def section(title):
    """Print a section header."""
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def load_data(path):
    """Load a CSV file."""
    df = pd.read_csv(path)
    print(f"Loaded {path.name}: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def explore(df):
    """Print missing values and survival rates."""
    print("\nFirst 5 rows:")
    print(df.head())

    missing = df.isnull().sum()
    print("\nMissing values per column:")
    print(missing[missing > 0])

    print(f"\nOverall survival rate: {df['Survived'].mean():.1%}")
    print("\nSurvival rate by sex:")
    print(df.groupby("Sex")["Survived"].mean().round(3))
    print("\nSurvival rate by passenger class:")
    print(df.groupby("Pclass")["Survived"].mean().round(3))


def get_fill_values(df):
    """Compute fill values for missing data from the training set."""
    fill_values = {
        "Age": float(df["Age"].median()),
        "Embarked": df["Embarked"].mode()[0],
        "Fare": float(df["Fare"].median()),
    }
    print(f"Fill values (computed from training data only): {fill_values}")
    return fill_values


def preprocess(df, fill_values):
    """Clean and encode a raw Titanic DataFrame."""
    df = df.copy()

    df = df.drop(columns=DROP_COLS)
    print(f"Dropped columns: {DROP_COLS}")

    for col, value in fill_values.items():
        n_missing = df[col].isnull().sum()
        df[col] = df[col].fillna(value)
        print(f"Filled {n_missing} missing '{col}' values with {value}")

    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    print("Encoded 'Sex': male -> 0, female -> 1")

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    print("Created 'FamilySize' = SibSp + Parch + 1")

    df = pd.get_dummies(df, columns=["Embarked"], prefix="Embarked", dtype=int)
    new_cols = [c for c in df.columns if c.startswith("Embarked_")]
    print(f"One-hot encoded 'Embarked' -> {new_cols}")

    print(f"Columns after preprocessing: {list(df.columns)}")
    return df


def main():
    section("STEP 1: Load training data")
    train_raw = load_data(DATA_DIR / "train.csv")

    section("STEP 2: Explore training data")
    explore(train_raw)

    section("STEP 3: Clean and adjust training data")
    fill_values = get_fill_values(train_raw)
    train = preprocess(train_raw, fill_values)

    section("STEP 4: Train logistic regression")
    X = train.drop(columns="Survived")
    y = train["Survived"]
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    print(f"Trained on {X.shape[0]} passengers using features: {list(X.columns)}")
    coefs = pd.Series(model.coef_[0], index=X.columns).sort_values(ascending=False)
    print("\nCoefficients (positive = higher survival chance):")
    print(coefs.round(3))

    section("STEP 5: Accuracy on training set")
    train_acc = accuracy_score(y, model.predict(X))
    print(f"Training accuracy: {train_acc:.2%}")

    # Using train_test_split to create a hold-out validation set from the training data
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    val_model = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
    val_acc = accuracy_score(y_val, val_model.predict(X_val))
    print(f"Hold-out validation accuracy (80/20 split of train.csv): {val_acc:.2%}")

    section("STEP 6: Load and preprocess test data")
    test_raw = load_data(DATA_DIR / "test.csv")
    test = preprocess(test_raw, fill_values)
    X_test = test.reindex(columns=X.columns, fill_value=0)
    print(f"Aligned test columns to training features ({X_test.shape[1]} columns)")

    section("STEP 7: Predict on test set")
    test_preds = model.predict(X_test)
    n_survived = int(test_preds.sum())
    print(f"Predicted survivors: {n_survived} / {len(test_preds)}")
    print(f"Predicted deaths:    {len(test_preds) - n_survived} / {len(test_preds)}")

    # test.csv has no labels; compare against the gender baseline as a proxy
    baseline = load_data(DATA_DIR / "gender_submission.csv")
    compare = test_raw[["PassengerId"]].assign(Predicted=test_preds)
    compare = compare.merge(baseline, on="PassengerId")
    agreement = (compare["Predicted"] == compare["Survived"]).mean()
    print(
        f"Agreement with gender_submission.csv baseline: {agreement:.2%}\n"
        "(Note: true test labels are not public, so this is NOT true test "
        "accuracy - see the hold-out validation accuracy above.)"
    )
    

if __name__ == "__main__":
    main()