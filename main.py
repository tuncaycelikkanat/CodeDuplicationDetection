import os
import random

from sklearn.model_selection import train_test_split

from models.linear_svm import build_linear_svm
from models.xgboost import build_xgboost
from preprocessing.tokenizer import tokenize, normalize_tokens
from vectorization.tfidf import build_tfidf_vectorizer
from pairing.pair_generator import generate_pairs
from models.random_forest import build_random_forest

from utils.experiment_logger import (
    generate_experiment_name,
    save_experiment
)

# <----------> CONFIG <---------->
DATASET_PATH = "data/poj104"
NUM_PAIRS = 400_000
TEST_SIZE = 0.2
RANDOM_STATE = 42

random.seed(RANDOM_STATE)

# <----------> LOAD DATA <---------->
print("---> Loading dataset...")

all_codes = []
labels = []

for label in sorted(os.listdir(DATASET_PATH)):
    class_path = os.path.join(DATASET_PATH, label)
    if not os.path.isdir(class_path):
        continue

    for file in os.listdir(class_path):
        if file.endswith(".txt"):
            with open(os.path.join(class_path, file), "r", errors="ignore") as f:
                all_codes.append(f.read())
                labels.append(label)

print(f"---> Total codes: {len(all_codes)}")


# <----------> PREPROCESS <---------->
print("---> Preprocessing codes...")

processed_codes = []
for code in all_codes:
    tokens = tokenize(code)
    norm_tokens = normalize_tokens(tokens)
    processed_codes.append(" ".join(norm_tokens))


# <----------> TF-IDF <---------->
print("---> Vectorizing with TF-IDF...")

vectorizer = build_tfidf_vectorizer()
X = vectorizer.fit_transform(processed_codes)

print("TF-IDF shape:", X.shape)
print("Feature count:", len(vectorizer.vocabulary_))


# <----------> PAIRS <---------->
print("---> Generating pairs...")

pairs_X, pairs_y = generate_pairs(X, labels, NUM_PAIRS)

print("Pair matrix:", pairs_X.shape)


# <----------> SPLIT <---------->
X_train, X_test, y_train, y_test = train_test_split(
    pairs_X,
    pairs_y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)


# <----------> MODEL <---------->
''' RANDOM FOREST '''
#print("---> Training Random Forest...")
#model = build_random_forest(RANDOM_STATE)
#model_name = "RandomForest"

''' LINEAR SVM '''
#print("---> Training Linear SVM...")
#model = build_linear_svm(RANDOM_STATE)
#model_name = "LinearSVM"

''' XGBOOST '''
print("---> Training model")
model = build_xgboost(RANDOM_STATE)
model_name = "XGBoost"

model.fit(X_train, y_train)

# <----------> EVALUATION <---------->
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


# <----------> SAVE EXPERIMENT <---------->
exp_name = generate_experiment_name(
    model_name=model_name,
    pair_count=NUM_PAIRS
)

save_experiment(
    exp_name=exp_name,
    model_name=model_name,
    model=model,
    vectorizer=vectorizer,
    pair_count=NUM_PAIRS,
    X_train=X_train,
    y_train=y_train,
    y_train_pred=y_train_pred,
    X_test=X_test,
    y_test=y_test,
    y_test_pred=y_test_pred
)
