# Predict Titanic survival with logistic regression and LightGBM (R).

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(lightgbm)
})

# Locate src/data relative to this script (works locally and in Docker)
args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args[grep("--file=", args)])
DATA_DIR <- file.path(dirname(dirname(normalizePath(script_path))), "data")

DROP_COLS <- c("PassengerId", "Name", "Ticket", "Cabin")
EMBARKED_LEVELS <- c("C", "Q", "S")
options(width = 200)


#' Print a section header.
section <- function(title) {
  line <- strrep("=", 60)
  cat("\n", line, "\n", title, "\n", line, "\n", sep = "")
}


#' Load a CSV file.
load_data <- function(path) {
  df <- read_csv(path, show_col_types = FALSE)
  cat(sprintf("Loaded %s: %d rows x %d columns\n", basename(path), nrow(df), ncol(df)))
  df
}


#' Print missing values and survival rates.
explore <- function(df) {
  cat("\nFirst 5 rows:\n")
  print(as.data.frame(head(df, 5)))

  missing <- colSums(is.na(df))
  cat("\nMissing values per column:\n")
  print(missing[missing > 0])

  cat(sprintf("\nOverall survival rate: %.1f%%\n", 100 * mean(df$Survived)))
  cat("\nSurvival rate by sex:\n")
  print(as.data.frame(df %>% group_by(Sex) %>% summarise(rate = round(mean(Survived), 3))))
  cat("\nSurvival rate by passenger class:\n")
  print(as.data.frame(df %>% group_by(Pclass) %>% summarise(rate = round(mean(Survived), 3))))
}


#' Compute fill values for missing data from the training set.
get_fill_values <- function(df) {
  fill_values <- list(
    Age = median(df$Age, na.rm = TRUE),
    Embarked = names(which.max(table(df$Embarked))),
    Fare = median(df$Fare, na.rm = TRUE)  # train has no gaps, but test.csv has 1
  )
  cat(sprintf("Fill values (computed from training data only): Age=%s, Embarked=%s, Fare=%s\n",
              fill_values$Age, fill_values$Embarked, fill_values$Fare))
  fill_values
}


#' Clean and encode a raw Titanic data frame.
preprocess <- function(df, fill_values) {
  df <- df %>% select(-all_of(DROP_COLS))
  cat("Dropped columns:", paste(DROP_COLS, collapse = ", "), "\n")

  for (col in names(fill_values)) {
    n_missing <- sum(is.na(df[[col]]))
    df[[col]] <- coalesce(df[[col]], fill_values[[col]])
    cat(sprintf("Filled %d missing '%s' values with %s\n", n_missing, col, fill_values[[col]]))
  }

  df <- df %>% mutate(Sex = ifelse(Sex == "female", 1, 0))
  cat("Encoded 'Sex': male -> 0, female -> 1\n")

  df <- df %>% mutate(Embarked = factor(Embarked, levels = EMBARKED_LEVELS))
  cat("Converted 'Embarked' to a factor with fixed levels:", paste(EMBARKED_LEVELS, collapse = ", "), "\n")

  cat("Kept SibSp and Parch separately (no FamilySize: it equals SibSp + Parch + 1,",
      "which makes glm coefficients undefined)\n")
  cat("Columns after preprocessing:", paste(names(df), collapse = ", "), "\n")
  as.data.frame(df)
}


#' Convert features to a numeric matrix (one-hot encodes Embarked) for LightGBM.
to_matrix <- function(df) {
  model.matrix(~ . , data = df)[, -1, drop = FALSE]
}


#' Fit a LightGBM binary classifier.
fit_lgbm <- function(X, y) {
  params <- list(
    objective = "binary",
    learning_rate = 0.05,
    num_leaves = 15,
    min_data_in_leaf = 20,
    seed = 42,
    deterministic = TRUE,
    num_threads = 1,
    verbose = -1
  )
  lgb.train(params = params, data = lgb.Dataset(X, label = y), nrounds = 200, verbose = -1)
}


accuracy <- function(pred, truth) mean(pred == truth)


main <- function() {
  section("STEP 1: Load training data")
  train_raw <- load_data(file.path(DATA_DIR, "train.csv"))

  section("STEP 2: Explore training data")
  explore(train_raw)

  section("STEP 3: Clean and adjust training data")
  fill_values <- get_fill_values(train_raw)
  train <- preprocess(train_raw, fill_values)
  y <- train$Survived
  X_df <- train %>% select(-Survived)

  section("STEP 4: Train models")
  logit <- glm(Survived ~ ., data = train, family = binomial)
  cat("Logistic regression coefficients (positive = higher survival chance):\n")
  print(round(sort(coef(logit), decreasing = TRUE), 3))

  X <- to_matrix(X_df)
  lgbm <- fit_lgbm(X, y)
  cat("\nLightGBM trained (200 trees). Feature importance (share of total gain):\n")
  imp <- lgb.importance(lgbm)
  print(as.data.frame(imp[, c("Feature", "Gain")]), digits = 3)

  section("STEP 5: Accuracy on training set")
  logit_train_pred <- ifelse(predict(logit, type = "response") > 0.5, 1, 0)
  lgbm_train_pred <- ifelse(predict(lgbm, X) > 0.5, 1, 0)
  cat(sprintf("Logistic regression training accuracy: %.2f%%\n", 100 * accuracy(logit_train_pred, y)))
  cat(sprintf("LightGBM training accuracy: %.2f%%\n", 100 * accuracy(lgbm_train_pred, y)))

  # Honest estimate on unseen data: same 80/20 split for both models
  set.seed(42)
  idx <- sample(nrow(train), floor(0.8 * nrow(train)))
  logit_v <- glm(Survived ~ ., data = train[idx, ], family = binomial)
  logit_val_pred <- ifelse(predict(logit_v, newdata = train[-idx, ], type = "response") > 0.5, 1, 0)
  lgbm_v <- fit_lgbm(X[idx, ], y[idx])
  lgbm_val_pred <- ifelse(predict(lgbm_v, X[-idx, ]) > 0.5, 1, 0)
  cat(sprintf("\nHold-out validation accuracy (80/20 split of train.csv):\n"))
  cat(sprintf("Logistic regression: %.2f%%\n", 100 * accuracy(logit_val_pred, y[-idx])))
  cat(sprintf("LightGBM: %.2f%%\n", 100 * accuracy(lgbm_val_pred, y[-idx])))

  section("STEP 6: Load and preprocess test data")
  test_raw <- load_data(file.path(DATA_DIR, "test.csv"))
  test <- preprocess(test_raw, fill_values)
  X_test <- to_matrix(test)
  cat(sprintf("Test feature matrix: %d rows x %d columns (same columns as training: %s)\n",
              nrow(X_test), ncol(X_test), identical(colnames(X_test), colnames(X))))

  section("STEP 7: Predict on test set")
  logit_test_pred <- ifelse(predict(logit, newdata = test, type = "response") > 0.5, 1, 0)
  lgbm_test_pred <- ifelse(predict(lgbm, X_test) > 0.5, 1, 0)
  n <- length(logit_test_pred)
  cat(sprintf("Logistic regression predicted survivors: %d / %d\n", sum(logit_test_pred), n))
  cat(sprintf("LightGBM predicted survivors: %d / %d\n", sum(lgbm_test_pred), n))
  cat(sprintf("The two models agree on %.2f%% of test passengers\n",
              100 * accuracy(logit_test_pred, lgbm_test_pred)))

  # test.csv has no labels; compare against the gender baseline as a proxy
  baseline <- load_data(file.path(DATA_DIR, "gender_submission.csv"))
  base_y <- baseline$Survived[match(test_raw$PassengerId, baseline$PassengerId)]
  cat(sprintf("Agreement with gender_submission.csv baseline:\n"))
  cat(sprintf("Logistic regression: %.2f%%\n", 100 * accuracy(logit_test_pred, base_y)))
  cat(sprintf("LightGBM: %.2f%%\n", 100 * accuracy(lgbm_test_pred, base_y)))
  cat("(Note: true test labels are not public, so this is NOT true test accuracy -",
      "see the hold-out validation accuracy above.)\n")
}

main()