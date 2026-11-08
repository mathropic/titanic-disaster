# titanic-disaster

Predict passenger survival in the Titanic disaster using the Kaggle Titanic dataset.
The repository contains two self-contained Docker pipelines, one in **Python** and one in **R**. Each pipeline loads and cleans the data, trains a model, and predicts survival on the test set. Every step prints its results to the terminal.

> **Short version:** Install Docker, clone this repo, put `train.csv`, `test.csv`, and `gender_submission.csv` in `src/data/`, then run the two `docker build` / `docker run` pairs below from the repo root.

---

## Repository structure

```
titanic-disaster/
├── README.md
├── .gitignore               # excludes src/data/, virtual environments, OS files
├── CODEOWNERS
└── src/
    ├── data/                # NOT in the repo - you create it and add the CSVs (Step 2)
    ├── python/
    │   ├── Dockerfile           # python:3.12-slim image
    │   ├── requirements.txt     # pinned Python packages
    │   └── main.py              # Python pipeline (logistic regression)
    └── r/
        ├── Dockerfile           # rocker/r-ver:4.6.1 image
        ├── install_packages.R   # installs pinned R packages
        └── main.R               # R pipeline (logistic regression + LightGBM)
```

---

## Prerequisites

Make sure you have Git installed, Docker Desktop installed and running, and a Kaggle account for downloading the data (Python and R are not needed locally - Docker provides both).

---

## Step 1 - Clone the repository

```bash
git clone https://github.com/mathropic/titanic-disaster.git
cd titanic-disaster
```

**All remaining commands must be run from this `titanic-disaster` folder (the repo root).**

---

## Step 2 - Download the data

The dataset is not stored in this repository. Download it from Kaggle:

1. Go to https://www.kaggle.com/competitions/titanic/data and sign in.
2. If prompted, click **Join Competition** / **I Understand and Accept** to accept the competition rules (required before downloading).
3. Click **Download All**. This downloads `titanic.zip`.
4. Create the data folder inside the repo:

   ```bash
   mkdir -p src/data
   ```
5. Unzip the download into that folder (adjust the path if your browser saves to a different location):

   ```bash
   unzip ~/Downloads/titanic.zip -d src/data
   ```

   On Windows, right-click `titanic.zip` → **Extract All…** and choose the `src\data` folder inside the repo.

<details>
<summary>Alternative: download with the Kaggle command-line tool</summary>

If you already have the Kaggle CLI set up (`pip install kaggle` and an API token in `~/.kaggle/kaggle.json`), and you have accepted the competition rules on the website:

```bash
mkdir -p src/data
kaggle competitions download -c titanic -p src/data
unzip src/data/titanic.zip -d src/data
```

</details>

**Check:** the folder must contain exactly these three files (the `.zip` may also be there; that is fine):

```bash
ls src/data
```

```
gender_submission.csv  test.csv  train.csv
```

---

## Step 3 - Run the Python pipeline

Build the image (the first build downloads the Python base image and packages, about 1 minute):

```bash
docker build -t titanic-python src/python
```

Run the container:

```bash
docker run --rm -v "$(pwd)/src/data:/app/data" titanic-python
```

What these commands do:

- `docker build -t titanic-python src/python` builds an image named `titanic-python` from the Dockerfile in `src/python/`.
- `-v "$(pwd)/src/data:/app/data"` makes your local `src/data` folder available inside the container at `/app/data`, where the script looks for the CSVs. The data is **mounted, not copied** into the image, so it never ends up in the image or the repo.
- `--rm` removes the container after it finishes.

---

## Step 4 - Run the R pipeline

Build the image (the first build downloads the R base image and packages, about 1 minute):

```bash
docker build -t titanic-r src/r
```

Run the container:

```bash
docker run --rm -v "$(pwd)/src/data:/app/data" titanic-r
```

The flags work exactly as in Step 3.

---

## Expected output

Each pipeline prints seven labelled sections (`STEP 1` through `STEP 7`): data loading, exploration, each cleaning step, model training, training accuracy, test data preprocessing, and test predictions.

The key numbers you should see are:

|                                            | Python: logistic regression | R: logistic regression | R: LightGBM |
| ------------------------------------------ | --------------------------- | ---------------------- | ----------- |
| Training accuracy                          | 80.02%                      | 79.80%                 | 90.91%      |
| Hold-out validation accuracy (80/20 split) | 80.45%                      | 75.98%                 | 83.24%      |
| Predicted survivors in `test.csv`        | 155 / 418                   | 156 / 418              | 142 / 418   |
| Agreement with `gender_submission.csv`   | 94.02%                      | 94.26%                 | 87.08%      |

Results are reproducible: random seeds are fixed in both scripts.

---

## What the pipelines do

Both scripts follow the same steps:

1. **Load** `train.csv` and print its shape and first few rows.
2. **Explore**: Check missing values by column, the overall survival rate (38.4%), and survival rates by sex (74% of women vs. 19% of men) and passenger class (63% in 1st class vs. 24% in 3rd class).
3. **Clean and adjust**:
   - Drop `PassengerId`, `Name`, and `Ticket` (identifiers / free text) and `Cabin` (77% missing).
   - Fill missing `Age` (median), `Embarked` (mode), and `Fare` (median). Imputation values are computed **from the training set only** and reused on the test set to avoid data leakage. `train.csv` has no missing `Fare` values, but `test.csv` has one.
   - Encode `Sex` as 0 (male) / 1 (female) and one-hot encode `Embarked`.
4. **Train** a logistic regression model and, in the R pipeline, a LightGBM gradient-boosted tree model for comparison.
5. **Evaluate** training accuracy and accuracy on an 80/20 hold-out split of `train.csv` to estimate performance on unseen data.
6. **Load** `test.csv` and apply the exact same preprocessing.
7. **Predict** survival for all 418 test passengers.

### Differences between the Python and R versions

- **Features:** Python adds `FamilySize = SibSp + Parch + 1`. R leaves it out because it is an exact linear combination of `SibSp` and `Parch`. R's unregularized `glm` cannot estimate a separate coefficient for it (it returns `NA`), while scikit-learn's default L2 regularization can handle the collinearity.
- **Encoding:** Python creates three `Embarked` dummy columns; R uses a factor, so `glm` creates two and treats port C as the reference level.
- **Models:** R additionally trains a LightGBM model. It fits the training data more closely (90.9% vs. 79.8%) and scored higher on this validation split (83.2% vs. 76.0%), but the gap between its training and validation accuracy suggests some overfitting.
- **Validation split:** Python and R draw different random 80/20 splits, so their validation accuracies are not directly comparable. With only 179 validation passengers, a single split can shift accuracy by a few percentage points.

### A note on "test accuracy"

Kaggle's `test.csv` does **not** include the `Survived` column; the true outcomes are hidden for the competition. Therefore, actual test accuracy cannot be computed locally. Instead, the scripts:

- Report **hold-out validation accuracy** on 20% of `train.csv` as an estimate of performance on unseen data.
- Report **agreement with `gender_submission.csv`**, Kaggle's example submission, which predicts that every woman survives and every man dies. This is a sanity check, not an accuracy metric. That baseline itself scores about 77% on Kaggle.
